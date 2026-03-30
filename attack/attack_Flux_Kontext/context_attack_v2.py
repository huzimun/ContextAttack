#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Context Suppression + Semantic Alignment (v2, fixed)

相较于原始 DeContext，本版本在“上下文抑制”目标之外，额外加入语义对齐项，
用于让对抗后的条件图像在若干特征空间内仍与参考图像/参考概念保持一致。

本修订版主要修复了原始 context_attack_v2.py 中的几个关键问题：
1. 原代码只优化语义对齐项，遗漏了 DeContext 原本的上下文抑制损失；
2. velocity / attention 参考分支只在第一次迭代计算一次，后续 timestep / noise 改变后参考目标失效；
3. perturbed 分支和 reference 分支共用同一个 transformer hook，导致 attention 对齐可能退化为“自己和自己对齐”；
4. 原 attention_align_loss 实际比较的是 q/k 归一化特征，而不是 attention map；本版改为比较真实 attention 分布；
5. 添加了更详细的中文注释，便于后续继续修改和做消融实验。
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.models.embeddings import apply_rotary_emb
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

logger = get_logger(__name__)


# ============================================================
# 数据集定义
# ============================================================
class AdversarialDatasetV2(Dataset):
    """
    读取条件图像与可选参考图像的数据集。

    参数说明：
        condition_images_dir: 待攻击的条件图像目录。
        reference_images_dir: 参考图像目录。若为 None，则默认以原条件图像本身作为参考。
        resolution: 输入图像统一缩放到的分辨率。

    返回字段：
        condition_image: 归一化到 [-1, 1] 的条件图像张量。
        cond_path: 条件图像原始路径。
        reference_image: 参考图像张量（若提供）。
        ref_path: 参考图像原始路径（若提供）。
    """

    def __init__(
        self,
        condition_images_dir: str,
        reference_images_dir: Optional[str] = None,
        resolution: int = 512,
    ) -> None:
        self.condition_images_dir = Path(condition_images_dir)
        self.reference_images_dir = Path(reference_images_dir) if reference_images_dir else None
        self.resolution = resolution

        self.cond_files = sorted(
            [f for f in self.condition_images_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        )
        if len(self.cond_files) == 0:
            raise ValueError(f"未在 {self.condition_images_dir} 中找到图像文件")

        if self.reference_images_dir is not None:
            self.ref_files = sorted(
                [f for f in self.reference_images_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            )
            if len(self.cond_files) != len(self.ref_files):
                raise ValueError("参考图像数量需与条件图像数量一致")
        else:
            self.ref_files = None

        # 与 FLUX 输入一致的基础预处理：缩放、中心裁剪、转张量、归一化到 [-1, 1]
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.cond_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cond_path = self.cond_files[idx]
        cond_image = Image.open(cond_path).convert("RGB")
        cond_image = self.image_transforms(cond_image)

        sample = {
            "condition_image": cond_image,
            "cond_path": str(cond_path),
        }

        if self.ref_files is not None:
            ref_path = self.ref_files[idx]
            ref_image = Image.open(ref_path).convert("RGB")
            ref_image = self.image_transforms(ref_image)
            sample["reference_image"] = ref_image
            sample["ref_path"] = str(ref_path)

        return sample


# ============================================================
# 文本编码
# ============================================================
def encode_prompt_flux_kontext(
    text_encoders: List[torch.nn.Module],
    tokenizers: List,
    prompt,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
):
    """
    编码 FLUX Kontext 所需的双分支文本特征。

    返回：
        prompt_embeds:        T5 token-level embedding，形状 [B, L, C]
        pooled_prompt_embeds: CLIP pooled embedding，形状 [B, C]
        text_ids:             文本 token 对应的位置信息占位张量，形状 [L, 3]
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # CLIP 分支：产生全局 pooled 文本语义
    text_inputs_clip = tokenizers[0](
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        pooled_prompt_embeds = text_encoders[0](text_inputs_clip.input_ids.to(device)).pooler_output

    # T5 分支：产生 token 级别文本表示
    text_inputs_t5 = tokenizers[1](
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = text_encoders[1](text_inputs_t5.input_ids.to(device))[0]

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


# ============================================================
# 注意力 Hook：抓取归一化后的 q / k
# ============================================================
class AttentionHook:
    """
    在 transformer 的注意力层上注册 forward hook，抓取 q_norm / k_norm。

    说明：
    - DeContext 的核心是分析“目标 token 对上下文 token 的注意力分配”；
    - 因此这里只需保存注意力计算前、已经归一化的 q / k；
    - 之后再显式重构 attention map，从而同时支持：
        1) 原始的 context suppression；
        2) 新增的 attention map alignment。
    """

    def __init__(self) -> None:
        self.qkv_normalized: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

    def register_hooks(self, transformer: FluxTransformer2DModel, selected_layers: Optional[Dict[str, List[int]]] = None) -> None:
        """
        在指定层上注册 hook。

        默认行为：
        - 对所有 single transformer blocks 注册；
        - 若用户有需要，也可以额外在 double blocks 上注册。
        """
        if selected_layers is None:
            selected_layers = {
                "single_blocks": list(range(0, 26))
            }

        def create_norm_hook(layer_name: str, proj_type: str):
            def hook_fn(module, input, output):
                if output is not None and hasattr(output, "shape"):
                    # clone 保留当前前向的张量快照，避免被后续前向覆盖/原地修改
                    self.qkv_normalized[f"{layer_name}_{proj_type}"] = output.clone()
            return hook_fn

        for i in selected_layers.get("double_blocks", []):
            if i < len(transformer.transformer_blocks):
                block = transformer.transformer_blocks[i]
                if hasattr(block, "attn"):
                    for proj_type, norm_layer in [("q", "norm_q"), ("k", "norm_k")]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f"double_{i}", proj_type)
                            )
                            self.hooks.append(hook)

        for i in selected_layers.get("single_blocks", []):
            if i < len(transformer.single_transformer_blocks):
                block = transformer.single_transformer_blocks[i]
                if hasattr(block, "attn"):
                    for proj_type, norm_layer in [("q", "norm_q"), ("k", "norm_k")]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f"single_{i}", proj_type)
                            )
                            self.hooks.append(hook)

    def clear_hooks(self) -> None:
        """移除所有 hook，并清空缓存。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.qkv_normalized = {}


# ============================================================
# 从 q / k 重构 attention map 及 context proportion
# ============================================================
def _get_sequence_ranges(layer_base: str, seq_len: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    根据层类型和序列长度，给出 text / target / context 三段 token 的切片边界。

    当前实现沿用原始 DeContext 代码中的默认假设：
    - 输入分辨率为 512；
    - FLUX Kontext 的 latent pack 方式不变。

    返回：
        (text_start, text_end, target_start, target_end, context_start, context_end)
    """
    if "double_" in layer_base:
        # double block 中通常不显式拼接 text token 到同一序列，因此这里只区分 target/context
        if seq_len != 2048:
            return None
        return (0, 0, 0, 1024, 1024, 2048)

    if "single_" in layer_base:
        # single block 中顺序为 [text, target image, context image]
        if seq_len != 2560:
            return None
        return (0, 512, 512, 1536, 1536, 2560)

    return None


def compute_attention_maps(
    qkv_normalized_dict: Dict[str, torch.Tensor],
    image_rotary_emb,
) -> Dict[str, torch.Tensor]:
    """
    由 hook 保存的 q_norm / k_norm 重构每一层的 attention map。

    返回：
        attention_maps[layer_name] = [B, H, Q_target, K_all]

    这里仅保留“目标图像 token 作为 query”的那一部分 attention，
    因为无论是上下文抑制还是 attention 对齐，我们关心的都是：
        目标输出到底在多大程度上依赖 context image。
    """
    attention_maps: Dict[str, torch.Tensor] = {}

    for layer_name in qkv_normalized_dict:
        if not layer_name.endswith("_q"):
            continue

        layer_base = layer_name[:-2]
        q_key = f"{layer_base}_q"
        k_key = f"{layer_base}_k"
        if q_key not in qkv_normalized_dict or k_key not in qkv_normalized_dict:
            continue

        q_norm = qkv_normalized_dict[q_key]
        k_norm = qkv_normalized_dict[k_key]
        _, seq_len, _, dim_head = q_norm.shape

        ranges = _get_sequence_ranges(layer_base, seq_len)
        if ranges is None:
            continue

        _, _, target_start, target_end, _, _ = ranges

        # 将 RoPE 重新施加到 q / k，使手工重构的 attention 与模型真实计算保持一致
        q_with_rope = apply_rotary_emb(q_norm, image_rotary_emb, sequence_dim=1)
        k_with_rope = apply_rotary_emb(k_norm, image_rotary_emb, sequence_dim=1)

        # 只取 target 对所有 key 的注意力
        q_target = q_with_rope[:, target_start:target_end].permute(0, 2, 1, 3)  # [B, H, Q, D]
        k_all = k_with_rope.permute(0, 2, 1, 3)                                  # [B, H, K, D]

        attn_scores = torch.matmul(q_target, k_all.transpose(-2, -1)) / math.sqrt(dim_head)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attention_maps[layer_base] = attn_weights

    return attention_maps


def compute_context_proportion(
    qkv_normalized_dict: Dict[str, torch.Tensor],
    image_rotary_emb,
) -> torch.Tensor:
    """
    计算 DeContext 原始目标中使用的 context proportion。

    定义：
        r_ctx = 平均而言，target query 分配给 context key 的注意力质量占比。

    返回：
        标量张量。值越大，说明模型越依赖 context image；值越小，说明上下文越被抑制。
    """
    attention_maps = compute_attention_maps(qkv_normalized_dict, image_rotary_emb)
    proportions = []

    for layer_base, attn_weights in attention_maps.items():
        seq_len = attn_weights.shape[-1]
        ranges = _get_sequence_ranges(layer_base, seq_len)
        if ranges is None:
            continue

        _, _, _, _, context_start, context_end = ranges
        context_weights = attn_weights[..., context_start:context_end]
        context_prop_per_query = context_weights.sum(dim=-1)
        proportions.append(context_prop_per_query.mean())

    if len(proportions) == 0:
        # 与前向同设备、同 dtype 保持一致
        any_tensor = next(iter(qkv_normalized_dict.values()))
        return torch.zeros((), device=any_tensor.device, dtype=any_tensor.dtype)

    return torch.stack(proportions).mean()


# ============================================================
# 语义对齐损失
# ============================================================
def compute_latent_align_loss(
    vae: AutoencoderKL,
    perturbed_img: torch.Tensor,
    reference_img: torch.Tensor,
    shift: float,
    scale: float,
) -> torch.Tensor:
    """
    VAE latent 对齐损失。

    作用：
        约束扰动后的条件图像在 VAE latent 空间中靠近参考图像，
        防止对抗优化把图像语义完全破坏掉。

    说明：
        reference_img 分支不需要梯度，因此放在 no_grad 中。
    """
    with torch.no_grad():
        ref_latent = vae.encode(reference_img).latent_dist.mode()
        ref_latent = (ref_latent - shift) * scale

    pert_latent = vae.encode(perturbed_img).latent_dist.mode()
    pert_latent = (pert_latent - shift) * scale
    return F.mse_loss(pert_latent, ref_latent)



def compute_velocity_align_loss(model_pred: torch.Tensor, ref_pred: torch.Tensor) -> torch.Tensor:
    """
    Flow matching velocity 对齐损失。

    要点：
        必须保证 model_pred 与 ref_pred 对应同一个 prompt、同一个 timestep、同一个 noisy target。
        否则 velocity 差异里会同时混入“随机采样差异”，对齐目标就不成立。
    """
    return F.mse_loss(model_pred, ref_pred)



def compute_attention_align_loss(
    pert_qkv_normalized: Dict[str, torch.Tensor],
    ref_qkv_normalized: Dict[str, torch.Tensor],
    image_rotary_emb,
) -> torch.Tensor:
    """
    Attention map 对齐损失。

    注意：
    - 原 context_attack_v2.py 名义上写的是 "attention map loss"，
      但实际比较的是 q_norm / k_norm 本身，而不是 attention map；
    - 本实现首先重构真实 attention map，再进行逐层 MSE 对齐，
      才与“attention map 对齐”这一表述一致。
    """
    pert_attn_maps = compute_attention_maps(pert_qkv_normalized, image_rotary_emb)
    ref_attn_maps = compute_attention_maps(ref_qkv_normalized, image_rotary_emb)

    losses = []
    for layer_name, pert_map in pert_attn_maps.items():
        if layer_name in ref_attn_maps:
            losses.append(F.mse_loss(pert_map, ref_attn_maps[layer_name]))

    if len(losses) == 0:
        any_tensor = next(iter(pert_qkv_normalized.values()))
        return torch.zeros((), device=any_tensor.device, dtype=any_tensor.dtype)

    return torch.stack(losses).mean()


# ============================================================
# 辅助：采样 prompt 池
# ============================================================
def get_prompt_pool() -> List[str]:
    """编辑指令池。为避免对单一 prompt 过拟合，每步随机采样一个 prompt。"""
    return [
        "Make this person smile happily.",
        "Make this person look angry.",
        "Make this person look surprised.",
        "Make this person look sad.",
        "Give this person a serious expression.",
        "Make this person laugh joyfully.",
        "Make this person look worried.",
        "Make this person look excited.",
        "Make this person look tired.",
        "Make this person look confused.",
        "Add glasses to this person.",
        "Add sunglasses to this person.",
        "Add a hat to this person.",
        "Add a scarf to this person.",
        "Add a necklace to this person.",
        "Add earrings to this person.",
        "Add a cap to this person.",
        "Add a headband to this person.",
        "Add a watch to this person.",
        "Add a tie to this person.",
        "Make this person wave their hand.",
        "Make this person cross their arms.",
        "Make this person point forward.",
        "Make this person give a thumbs up.",
        "Make this person put hands on hips.",
        "Make this person cover their mouth.",
        "Make this person hold up a peace sign.",
        "Make this person shrug their shoulders.",
        "Make this person touch their chin.",
        "Make this person salute.",
        "Make this person look older.",
        "Make this person look younger.",
        "Add a beard to this person.",
        "Add a mustache to this person.",
        "Make this person's hair longer.",
        "Make this person's hair shorter.",
        "Add makeup to this person.",
        "Add freckles to this person.",
        "Make this person's skin tanned.",
        "Add stubble to this person.",
        "Place this person on a tropical beach with palm trees and ocean waves.",
        "Put this person in a snowy mountain landscape with pine trees.",
        "Move this person to a bustling city street with skyscrapers.",
        "Put this person in a beach setting.",
        "Put this person in a snowy mountain.",
        "Put this person in a city street.",
        "Put this person in a forest.",
        "Put this person in a coffee shop.",
        "Put this person in a library.",
        "Put this person in a park.",
        "Make this person smile and wave.",
        "Add glasses and make this person look serious.",
        "Make this person look older with a beard.",
        "Add a hat and make this person point forward.",
        "Make this person look surprised with hands on face.",
        "Add sunglasses and make this person give a thumbs up.",
        "Make this person look younger and smile happily.",
        "Add a scarf and make this person cross their arms.",
        "Make this person look tired and rub their eyes.",
        "Add earrings and make this person look confident.",
    ]


# ============================================================
# 单次前向：得到 velocity / attention / context proportion 所需中间量
# ============================================================
def forward_with_hooks(
    transformer: FluxTransformer2DModel,
    vae: AutoencoderKL,
    condition_img: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    guidance: torch.Tensor,
    shift: float,
    scale: float,
    weight_dtype: torch.dtype,
    device: torch.device,
):
    """
    对给定条件图像执行一次 transformer 前向，并返回：
        model_pred            : flow matching velocity 预测
        qkv_normalized        : 供 attention map 重构使用的 q/k 缓存
        combined_latent_ids   : 图像 token 的位置/模态 id
        image_rotary_emb      : 当前 text+image 序列对应的 RoPE

    设计原因：
        perturbed 分支和 reference 分支都需要完全一致的随机变量（prompt/timestep/noise），
        因此将它们统一封装，确保两条分支仅在 condition image 上不同。
    """
    bsz = condition_img.shape[0]

    # 将条件图像编码到 VAE latent 空间，作为 context image latent
    cond_latents = vae.encode(condition_img).latent_dist.mode()
    cond_latents = (cond_latents - shift) * scale

    # target image token 的位置/模态 id
    latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
        bsz,
        cond_latents.shape[2] // 2,
        cond_latents.shape[3] // 2,
        device,
        weight_dtype,
    )

    # context image token 的位置/模态 id
    cond_latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
        cond_latents.shape[0],
        cond_latents.shape[2] // 2,
        cond_latents.shape[3] // 2,
        device,
        weight_dtype,
    )
    cond_latent_ids[..., 0] = 1

    combined_latent_ids = torch.cat([latent_image_ids, cond_latent_ids], dim=0)

    packed_noisy_latents = FluxKontextPipeline._pack_latents(
        noisy_latents,
        bsz,
        noisy_latents.shape[1],
        noisy_latents.shape[2],
        noisy_latents.shape[3],
    )
    packed_cond = FluxKontextPipeline._pack_latents(
        cond_latents,
        bsz,
        cond_latents.shape[1],
        cond_latents.shape[2],
        cond_latents.shape[3],
    )
    packed_input = torch.cat([packed_noisy_latents, packed_cond], dim=1)

    hook = AttentionHook()
    hook.register_hooks(transformer)

    timesteps_expanded = timesteps.expand(bsz).to(packed_noisy_latents.dtype)
    model_pred = transformer(
        hidden_states=packed_input,
        timestep=timesteps_expanded / 1000,
        guidance=guidance.expand(bsz),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=combined_latent_ids,
        return_dict=False,
    )[0]

    # RoPE 的 token 顺序必须与真实前向中的 token 拼接顺序一致
    ids = torch.cat([text_ids, latent_image_ids, cond_latent_ids], dim=0)
    image_rotary_emb = transformer.pos_embed(ids)

    # 注意：这里要把 hook 中保存的张量拷出来，再清理 hook，避免后续 reference forward 覆盖。
    qkv_normalized = {k: v for k, v in hook.qkv_normalized.items()}
    hook.clear_hooks()

    return model_pred, qkv_normalized, combined_latent_ids, image_rotary_emb


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./example")
    parser.add_argument("--reference_images_dir", type=str, default=None, help="参考图像目录；若为空，则使用原条件图像作为参考")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./perturbed")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # PGD 参数
    parser.add_argument("--alpha", type=float, default=0.005, help="每一步 PGD 更新步长")
    parser.add_argument("--eps", type=float, default=0.1, help="L_inf 扰动半径")
    parser.add_argument("--attack_steps", type=int, default=800, help="总优化步数")

    # 上下文抑制损失权重
    parser.add_argument("--context_loss_weight", type=float, default=1.0, help="DeContext 原始上下文抑制项权重")

    # 语义对齐损失配置
    parser.add_argument(
        "--align_loss_types",
        type=str,
        default="latent",
        help="启用的语义对齐损失类型，逗号分隔，可选：latent,velocity,attention",
    )
    parser.add_argument(
        "--align_loss_weights",
        type=str,
        default="1.0,1.0,1.0",
        help="语义对齐损失对应权重，顺序与 align_loss_types 一一对应",
    )

    return parser.parse_args()


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # prompt 池：每一步随机采样一个 prompt，提高扰动在不同编辑指令下的泛化性
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts in the pool")

    # ------------------------------------------------------------
    # 加载模型组件
    # ------------------------------------------------------------
    tokenizer_clip = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_t5 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 模型移动到设备上，并冻结参数；优化对象仅为输入图像像素
    text_encoder_clip.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    text_encoder_clip.eval()
    text_encoder_t5.eval()
    vae.eval()
    transformer.eval()

    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # 使用 gradient checkpointing 以节省显存
    transformer.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    # ------------------------------------------------------------
    # 数据集与超参数准备
    # ------------------------------------------------------------
    dataset = AdversarialDatasetV2(
        condition_images_dir=args.condition_images_dir,
        reference_images_dir=args.reference_images_dir,
        resolution=args.resolution,
    )
    logger.info(f"Loaded {len(dataset)} condition images")

    vae_shift_factor = vae.config.shift_factor
    vae_scaling_factor = vae.config.scaling_factor

    # 解析语义对齐项配置
    align_types = [s.strip() for s in args.align_loss_types.split(",") if s.strip()]
    align_weights = [float(s) for s in args.align_loss_weights.split(",") if s.strip()]
    align_type2weight = {
        loss_name: align_weights[i] if i < len(align_weights) else 1.0
        for i, loss_name in enumerate(align_types)
    }

    logger.info(f"Enabled semantic alignment losses: {align_type2weight}")
    logger.info(f"Context loss weight: {args.context_loss_weight}")

    # ------------------------------------------------------------
    # 逐张图像优化
    # ------------------------------------------------------------
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample["condition_image"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample["cond_path"]

        reference_image = sample.get("reference_image", None)
        if reference_image is not None:
            reference_image = reference_image.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        else:
            # 若未显式提供参考图像，则默认让“语义保持”相对于原条件图像定义
            reference_image = condition_image.clone().detach()

        logger.info(f"Processing {idx + 1}/{len(dataset)}: {Path(cond_path).name}")

        # PGD 初始化：从原图本身开始优化
        perturbed_condition = condition_image.clone().detach()
        original_condition = condition_image.clone().detach()
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")

        # FLUX Kontext inference 中 guidance 一般固定为常数，这里沿用原代码配置
        guidance = torch.tensor([3.5], device=accelerator.device)

        # --------------------------------------------------------
        # 迭代优化
        # --------------------------------------------------------
        for step in progress_bar:
            perturbed_condition.requires_grad_(True)

            # 每一步随机采样一个 prompt / timestep / noise。
            # 对于 velocity 对齐与 attention 对齐，reference 分支必须使用同一组随机变量。
            prompt = random.choice(prompt_pool)
            text_encoders = [text_encoder_clip, text_encoder_t5]
            tokenizers = [tokenizer_clip, tokenizer_t5]
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_flux_kontext(
                text_encoders,
                tokenizers,
                prompt,
                args.max_sequence_length,
                accelerator.device,
            )

            bsz = 1

            # 按 DeContext 论文设置，重点采样高 timestep（早期去噪 / 高噪声阶段）
            timesteps = torch.randint(
                980,
                scheduler.config.num_train_timesteps,
                (bsz,),
                device=accelerator.device,
            ).long()

            # DeContext 中 target image 用随机噪声近似即可，无需真实目标图像
            dummy_target_shape = (bsz, 16, args.resolution // 8, args.resolution // 8)
            noisy_latents = torch.randn(dummy_target_shape, device=accelerator.device, dtype=weight_dtype)

            losses: Dict[str, torch.Tensor] = {}

            # ----------------------------------------------------
            # 1) 语义对齐：latent
            # ----------------------------------------------------
            if "latent" in align_type2weight:
                losses["latent"] = compute_latent_align_loss(
                    vae,
                    perturbed_condition,
                    reference_image,
                    vae_shift_factor,
                    vae_scaling_factor,
                )

            # ----------------------------------------------------
            # 2) perturbed 分支前向
            # ----------------------------------------------------
            pert_model_pred, pert_qkv_normalized, _, pert_image_rotary_emb = forward_with_hooks(
                transformer=transformer,
                vae=vae,
                condition_img=perturbed_condition,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                guidance=guidance,
                shift=vae_shift_factor,
                scale=vae_scaling_factor,
                weight_dtype=weight_dtype,
                device=accelerator.device,
            )

            # ----------------------------------------------------
            # 3) 计算 DeContext 原始上下文抑制项
            # ----------------------------------------------------
            # 按论文定义，L_DeContext = 1 - r_ctx，并通过梯度上升进行优化。
            context_prop = compute_context_proportion(pert_qkv_normalized, pert_image_rotary_emb)
            context_loss = 1.0 - context_prop
            losses["context"] = context_loss

            # ----------------------------------------------------
            # 4) 参考分支前向（仅在需要 velocity / attention 对齐时计算）
            # ----------------------------------------------------
            ref_model_pred = None
            ref_qkv_normalized = None
            if "velocity" in align_type2weight or "attention" in align_type2weight:
                with torch.no_grad():
                    ref_model_pred, ref_qkv_normalized, _, ref_image_rotary_emb = forward_with_hooks(
                        transformer=transformer,
                        vae=vae,
                        condition_img=reference_image,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        guidance=guidance,
                        shift=vae_shift_factor,
                        scale=vae_scaling_factor,
                        weight_dtype=weight_dtype,
                        device=accelerator.device,
                    )

                # 对于同一 prompt / timestep / noise，text_ids 与 image token 结构一致，
                # perturbed 分支与 reference 分支的 RoPE 是一致的；因此此处直接使用 pert_image_rotary_emb 即可。
                _ = ref_image_rotary_emb

            # ----------------------------------------------------
            # 5) 语义对齐：velocity / attention
            # ----------------------------------------------------
            if "velocity" in align_type2weight:
                losses["velocity"] = compute_velocity_align_loss(pert_model_pred, ref_model_pred)

            if "attention" in align_type2weight:
                losses["attention"] = compute_attention_align_loss(
                    pert_qkv_normalized,
                    ref_qkv_normalized,
                    pert_image_rotary_emb,
                )

            # ----------------------------------------------------
            # 6) 总损失组合
            # ----------------------------------------------------
            # 关键说明：
            # - 上下文抑制项 context_loss = 1 - r_ctx，希望“越大越好”；
            # - 语义对齐项 latent/velocity/attention 都是 MSE，希望“越小越好”；
            # - 本代码仍使用 PGD 的“梯度上升”形式更新图像，因此总目标应写成：
            #       L_total = w_ctx * L_context - Σ_i w_i * L_align_i
            #   这样在做 ascent 时，既会增大 context suppression，又会减小 semantic loss。
            semantic_align_total = torch.zeros((), device=accelerator.device, dtype=perturbed_condition.dtype)
            for loss_name in ["latent", "velocity", "attention"]:
                if loss_name in losses:
                    semantic_align_total = semantic_align_total + align_type2weight.get(loss_name, 1.0) * losses[loss_name]

            total_objective = args.context_loss_weight * losses["context"] - semantic_align_total

            # ----------------------------------------------------
            # 7) 反向传播并执行 PGD 更新
            # ----------------------------------------------------
            transformer.zero_grad()
            vae.zero_grad()
            if perturbed_condition.grad is not None:
                perturbed_condition.grad.zero_()

            total_objective.backward()

            # 梯度上升：沿着 total_objective 增大的方向更新
            adv_condition = perturbed_condition + args.alpha * perturbed_condition.grad.sign()

            # 投影回以原图为中心、半径 eps 的 L_inf 球
            eta = torch.clamp(adv_condition - original_condition, min=-args.eps, max=args.eps)

            # 同时裁剪回合法像素范围 [-1, 1]
            perturbed_condition = torch.clamp(original_condition + eta, min=-1.0, max=1.0).detach()
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()

            # 可视化日志
            log_dict = {
                "ctx_prop": f"{context_prop.item():.4f}",
                "L_ctx": f"{losses['context'].item():.4f}",
                "L_sem": f"{semantic_align_total.item():.4f}",
                "L_total": f"{total_objective.item():.4f}",
                "pert": f"{perturbation_norm:.4f}",
            }
            if "latent" in losses:
                log_dict["latent"] = f"{losses['latent'].item():.4f}"
            if "velocity" in losses:
                log_dict["velocity"] = f"{losses['velocity'].item():.4f}"
            if "attention" in losses:
                log_dict["attention"] = f"{losses['attention'].item():.4f}"
            progress_bar.set_postfix(log_dict)

            # 主动清理引用，缓解显存压力
            del losses, semantic_align_total, total_objective
            del pert_model_pred, pert_qkv_normalized
            if ref_model_pred is not None:
                del ref_model_pred, ref_qkv_normalized
            torch.cuda.empty_cache()

        # --------------------------------------------------------
        # 保存最终扰动图像
        # --------------------------------------------------------
        del progress_bar, guidance
        torch.cuda.empty_cache()

        with torch.no_grad():
            output_path = os.path.join(args.output_dir, f"adversarial_{Path(cond_path).stem}.png")
            perturbed_np = perturbed_condition.detach().squeeze(0).cpu().float().numpy()
            perturbed_np = (perturbed_np + 1.0) * 127.5
            perturbed_np = np.transpose(perturbed_np, (1, 2, 0)).astype(np.uint8)
            Image.fromarray(perturbed_np).save(output_path)
            logger.info(f"Saved final result: {output_path}")

    logger.info("Attack completed!")


if __name__ == "__main__":
    main()
