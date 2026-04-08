#!/usr/bin/env python
# coding=utf-8

"""
Context Attack v3 (论文级完整实现)

核心特点：
1. 完整 FLUX latent packing（target + context）
2. 精确 token 划分（target / context / text）
3. RoPE attention 计算（与模型一致）
4. 多层语义对齐（5种loss）
5. reference 分支严格对齐
6. accelerate + wandb

优化目标：
max L_context - λ1 L_latent - λ2 L_velocity - λ3 L_attn - λ4 L_qk
"""

# =========================
# 基础库
# =========================
import argparse
import torch
import torch.nn.functional as F
import os
import math
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import time
import random
import logging
import wandb

from torchvision import transforms
from torch.utils.data import Dataset

# =========================
# Diffusion / FLUX 模块
# =========================
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    FluxKontextPipeline
)

from diffusers.models.embeddings import apply_rotary_emb

from transformers import (
    CLIPTokenizer, CLIPTextModel,
    T5TokenizerFast, T5EncoderModel
)

logger = logging.getLogger(__name__)

import signal, time, sys

def _signal_handler(signum, frame):
    print(f"[SIGNAL] received signal {signum} at {time.strftime('%H:%M:%S')}", flush=True)
    sys.exit(1)

# =========================
# 数据集
# =========================
class AdversarialDataset(Dataset):
    """
    数据集类：仅加载“条件图像（context image）”

    关键点：
    - 不需要标签
    - 每个样本是一张待攻击的输入图像
    """

    def __init__(self, image_dir, reference_images_dir=None, resolution=512):
        self.files = sorted([
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ])
        
        self.reference_images_dir = Path(reference_images_dir) if reference_images_dir is not None else None

        if self.reference_images_dir is not None:
            self.ref_files = sorted([
                f for f in self.reference_images_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
            ])
            assert len(self.files) == len(self.ref_files), "reference 图像数量必须与 condition 图像数量一致"
        else:
            self.ref_files = None

        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")

        sample = {
            "image": self.transform(img),
            "path": str(path)
        }

        if self.ref_files is not None:
            ref_path = self.ref_files[idx]
            ref_img = Image.open(ref_path).convert("RGB")
            sample["reference_image"] = self.transform(ref_img)
            sample["reference_path"] = str(ref_path)

        return sample


# =========================
# Prompt 编码
# =========================
def encode_prompt_flux(encoders, tokenizers, prompt, device):
    """
    FLUX 双编码器：

    1. CLIP → 全局语义（pooled）
    2. T5 → token级语义（用于 cross-attention）

    返回：
        prompt_embeds      : [B, L, C]
        pooled_embeddings  : [B, C]
        text_ids           : RoPE 用
    """

    prompt = [prompt]

    # CLIP
    clip_inputs = tokenizers[0](prompt, return_tensors="pt", padding="max_length", max_length=77)
    pooled = encoders[0](clip_inputs.input_ids.to(device)).pooler_output

    # T5
    t5_inputs = tokenizers[1](prompt, return_tensors="pt", padding="max_length", max_length=512)
    embeds = encoders[1](t5_inputs.input_ids.to(device))[0]

    # text ids（用于 RoPE）
    text_ids = torch.zeros(embeds.shape[1], 3).to(device)

    return embeds, pooled, text_ids


# =========================
# Attention Hook
# =========================
class AttentionHook:
    """
    hook Transformer 内部的 q_norm / k_norm

    用于：
    - context loss
    - attention alignment
    - qk alignment
    """

    def __init__(self):
        self.qkv = {}
        self.hooks = []

    def register(self, transformer, selected_layers=None):
        """
        在所有 Transformer block 上注册 hook
        """
        # 与原始 DeContext 一致：
        # 若未显式指定，则默认只使用 early-to-middle 的 single blocks
        if selected_layers is None:
            selected_layers = {
                "single_blocks": list(range(0, 26))
            }

        def make_hook(name):
            def fn(module, inp, out):
                self.qkv[name] = out.clone()
            return fn

        # ===== double blocks =====
        for i in selected_layers.get("double_blocks", []):
            if i < len(transformer.transformer_blocks):
                block = transformer.transformer_blocks[i]
                if hasattr(block, "attn"):
                    if hasattr(block.attn, "norm_q"):
                        self.hooks.append(
                            block.attn.norm_q.register_forward_hook(make_hook(f"double_{i}_q"))
                        )
                    if hasattr(block.attn, "norm_k"):
                        self.hooks.append(
                            block.attn.norm_k.register_forward_hook(make_hook(f"double_{i}_k"))
                        )

        # ===== single blocks =====
        for i in selected_layers.get("single_blocks", []):
            if i < len(transformer.single_transformer_blocks):
                block = transformer.single_transformer_blocks[i]
                if hasattr(block, "attn"):
                    if hasattr(block.attn, "norm_q"):
                        self.hooks.append(
                            block.attn.norm_q.register_forward_hook(make_hook(f"single_{i}_q"))
                        )
                    if hasattr(block.attn, "norm_k"):
                        self.hooks.append(
                            block.attn.norm_k.register_forward_hook(make_hook(f"single_{i}_k"))
                        )

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.qkv = {}
        self.hooks = []


# =========================
# Context proportion（核心）
# =========================
def compute_context_proportion(qkv, rotary_emb):
    """
    严格计算：

    target token → context token attention 占比

    步骤：
    1. q,k 加 RoPE
    2. 计算 attention
    3. slice context token
    4. 求占比
    """

    freqs_cos, freqs_sin = rotary_emb
    props = []

    for k in qkv:
        if not k.endswith("_q"):
            continue

        base = k[:-2]
        q = qkv[base+"_q"]
        k_ = qkv[base+"_k"]

        B, S, H, D = q.shape
        
        # 显式区分 single-stream 序列中的 [text | target | context]
        # 这里沿用原始 DeContext 对 FLUX single block 的 token 布局假设：
        # text:    [0, 512)
        # target:  [512, 1536)
        # context: [1536, 2560)
        #
        # 如果后续分辨率或 token 组织方式变化，这里的边界也需要同步调整。
        # 根据 block 类型显式区分 token 布局
        # 与原始 DeContext 一致：
        #   double block: [target | context]
        #   single block: [text | target | context]
        if base.startswith("double_"):
            if S != 2048:
                continue
            target_start, target_end = 0, 1024
            context_start, context_end = 1024, 2048

        elif base.startswith("single_"):
            if S != 2560:
                continue
            text_start, text_end = 0, 512
            target_start, target_end = 512, 1536
            context_start, context_end = 1536, 2560

        else:
            continue

        # reshape
        q = apply_rotary_emb(q, rotary_emb, sequence_dim=1)
        k_ = apply_rotary_emb(k_, rotary_emb, sequence_dim=1)

        q = q.permute(0,2,1,3)
        k_ = k_.permute(0,2,1,3)

        attn = torch.softmax(q @ k_.transpose(-2, -1) / math.sqrt(D), dim=-1)

        # 只保留 target token 作为 query
        target_attn = attn[:, :, target_start:target_end, :]

        # 只统计 target queries 分配给 context keys 的注意力质量
        context_attn = target_attn[:, :, :, context_start:context_end]

        prop = context_attn.sum(dim=-1).mean()
        props.append(prop)

    if len(props) == 0:
        # 保持和当前 device 一致；rotary_emb[0] 一定在正确设备上
        return torch.tensor(0.0, device=rotary_emb[0].device)

    return torch.stack(props).mean()


# =========================
# 五种 Loss
# =========================
def compute_losses(cond_latents_adv, cond_latents_ref,
                   pred, ref_pred,
                   adv_qkv, ref_qkv, rotary_emb):

    # context
    ctx_prop = compute_context_proportion(adv_qkv, rotary_emb)
    Lc = 1 - ctx_prop

    # latent loss（直接使用当前 step 的 context latent）
    Ll = F.mse_loss(cond_latents_adv, cond_latents_ref)

    # velocity
    Lv = F.mse_loss(pred, ref_pred)

    # attention map loss
    # 真正对齐的是 softmax(QK^T / sqrt(d)) 之后的注意力分布，而不是 q/k 特征本身
    # ===== attention map loss（DeContext-style：只对齐 target→context）=====
    La = 0.0
    n_attn = 0

    for name in adv_qkv:
        if not name.endswith("_q"):
            continue

        base = name[:-2]
        q_key = base + "_q"
        k_key = base + "_k"

        if q_key not in adv_qkv or k_key not in adv_qkv:
            continue
        if q_key not in ref_qkv or k_key not in ref_qkv:
            continue

        q = adv_qkv[q_key]
        k_ = adv_qkv[k_key]
        q_ref = ref_qkv[q_key]
        k_ref = ref_qkv[k_key]

        B, S, H, D = q.shape

        # ===== 根据 block 类型确定切片区间 =====
        if base.startswith("double_"):
            if S != 2048:
                continue
            target_start, target_end = 0, 1024
            context_start, context_end = 1024, 2048

        elif base.startswith("single_"):
            if S != 2560:
                continue
            target_start, target_end = 512, 1536
            context_start, context_end = 1536, 2560

        else:
            continue

        # ===== RoPE 对齐 =====
        q = apply_rotary_emb(q, rotary_emb, sequence_dim=1)
        k_ = apply_rotary_emb(k_, rotary_emb, sequence_dim=1)
        q_ref = apply_rotary_emb(q_ref, rotary_emb, sequence_dim=1)
        k_ref = apply_rotary_emb(k_ref, rotary_emb, sequence_dim=1)

        # [B, H, S, D]
        q = q.permute(0, 2, 1, 3)
        k_ = k_.permute(0, 2, 1, 3)
        q_ref = q_ref.permute(0, 2, 1, 3)
        k_ref = k_ref.permute(0, 2, 1, 3)

        # ===== attention 重建 =====
        attn = torch.softmax(q @ k_.transpose(-2, -1) / math.sqrt(D), dim=-1)
        attn_ref = torch.softmax(q_ref @ k_ref.transpose(-2, -1) / math.sqrt(D), dim=-1)

        # ===== 只取 target → context 子注意力 =====
        target_attn = attn[:, :, target_start:target_end, :]
        target_attn_ref = attn_ref[:, :, target_start:target_end, :]

        context_attn = target_attn[:, :, :, context_start:context_end]
        context_attn_ref = target_attn_ref[:, :, :, context_start:context_end]

        La = La + F.mse_loss(context_attn, context_attn_ref)
        n_attn += 1

    La = La / max(n_attn, 1)

    # qk feature loss
    # 对齐的是 attention 计算前的 q/k 特征，而不是 softmax 后的注意力分布
    Lq = 0.0
    n_qk = 0

    for name in adv_qkv:
        if name not in ref_qkv:
            continue

        # 只对 q 和 k 两类特征做约束
        if not (name.endswith("_q") or name.endswith("_k")):
            continue

        Lq = Lq + F.mse_loss(adv_qkv[name], ref_qkv[name])
        n_qk += 1

    Lq = Lq / max(n_qk, 1)

    return Lc, Ll, Lv, La, Lq


# =========================
# 编辑指令池
# =========================
# 这里构造了一组图像编辑 prompt，用来在攻击优化过程中随机采样。
# 这样做的目的不是针对单一 prompt 过拟合，而是学习一种更通用的扰动，
# 使得同一张保护图像在多种编辑指令下都更难被模型“正确利用上下文”。
# 这与论文中强调的“通用上下文破坏”是一致的。
def get_prompt_pool():
    """
    a prompt set containing 20 image editing commands
    """
    prompts = [
        
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
    

    # ===== 2. Accessory Addition (10) =====

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


    # ===== 3. Posture and Movement (10) =====

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


    # ===== 4. Style Changes (10) =====

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


    # ===== 5. Scenes Changes (10) =====

        "Place this person on a tropical beach with palm trees and ocean waves.",
        "Put this person in a snowy mountain landscape with pine trees.",
        "Move this person to a bustling city street with skyscrapers.",
        "Put this person in a beach setting.",
        "Put this person in a snowy mountain.",
        "Put this person in a city street.",
        "Put this person in a forest.",
        "Put this person in a coffee shop.",
        "Put this person in a library.",
        "Put this person in a park."


    # ===== 6. Combined Prompts (10) =====

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


    return prompts

# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        signal.signal(signal.SIGHUP, _signal_handler)
    except Exception:
        pass

    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--condition_images_dir", type=str)
    parser.add_argument("--reference_images_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--attack_steps", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # loss weights（统一接口，便于做 ablation）
    parser.add_argument("--w_c", type=float, default=1.0, help="weight for context loss")
    parser.add_argument("--w_l", type=float, default=0.5, help="weight for latent loss")
    parser.add_argument("--w_v", type=float, default=1.0, help="weight for velocity loss")
    parser.add_argument("--w_a", type=float, default=1.0, help="weight for attention-map loss")
    parser.add_argument("--w_q", type=float, default=0.2, help="weight for qk-feature loss")

    parser.add_argument("--device", type=str, default=None,
                        help="device to run on (e.g. cpu, cuda, cuda:0). Auto-select if not set")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="mixed precision mode (affects some tensors/ops)")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    
    # ===== 确保输出目录存在 =====
    os.makedirs(args.output_dir, exist_ok=True)

    # device selection: accept 'cpu', 'cuda', or 'cuda:0' style strings
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    logger.warning("Requested CUDA device but CUDA is not available. Falling back to CPU.")
                    device = torch.device("cpu")
                else:
                    # if an explicit index was provided, validate it
                    if args.device.startswith("cuda:"):
                        try:
                            idx = int(args.device.split(":", 1)[1])
                            if idx < 0 or idx >= torch.cuda.device_count():
                                logger.warning(f"CUDA device index {idx} is out of range (0..{max(0, torch.cuda.device_count()-1)}). Using default CUDA device.")
                                device = torch.device("cuda")
                        except Exception:
                            logger.warning("Could not parse CUDA device index; using default CUDA device.")
                            device = torch.device("cuda")
        except Exception:
            logger.warning("Invalid device string provided; auto-selecting device.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    # mixed precision hint (we will not force model dtype, but keep a dtype for generated tensors)
    if args.mixed_precision == "fp16":
        mp_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        mp_dtype = torch.bfloat16
    else:
        mp_dtype = torch.float32

    # wandb init (single-process)
    if not args.no_wandb:
        try:
            wandb.init(
                project="context_attack_v3",
                config=vars(args),
                name=f"wc{args.w_c}_wl{args.w_l}_wv{args.w_v}_wa{args.w_a}_wq{args.w_q}"
            )
        except Exception:
            logger.warning("wandb init failed; continuing without wandb")

    # =========================
    # 加载模型
    # =========================
    tokenizer_clip = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    text_encoder_t5 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2").to(device)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer").to(device)

    # 显式切到推理模式，避免训练态行为影响结果
    text_encoder_clip.eval()
    text_encoder_t5.eval()
    vae.eval()
    transformer.eval()
    
    # 不更新模型参数，只保留对输入图像 adv 的梯度
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # ensure models and generated tensors share dtype when using mixed precision
    try:
        text_encoder_clip = text_encoder_clip.to(mp_dtype)
        text_encoder_t5 = text_encoder_t5.to(mp_dtype)
        vae = vae.to(mp_dtype)
        transformer = transformer.to(mp_dtype)
    except Exception:
        # fallback: if device/dtype combination isn't supported, keep models in default dtype
        logger.debug("Could not cast models to mp_dtype; keeping default dtype")
        
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    dataset = AdversarialDataset(
        args.condition_images_dir,
        reference_images_dir=args.reference_images_dir
    )
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts in the pool")
    
    # 预编码全部 prompt，避免每个 attack step 反复跑文本编码器
    prompt_embeds_pool = []
    pooled_prompt_embeds_pool = []
    text_ids_pool = []

    logger.info("Pre-encoding all prompts...")
    for prompt in prompt_pool:
        with torch.no_grad():
            embeds, pooled, text_ids = encode_prompt_flux(
                [text_encoder_clip, text_encoder_t5],
                [tokenizer_clip, tokenizer_t5],
                prompt,
                device
            )
        prompt_embeds_pool.append(embeds)
        pooled_prompt_embeds_pool.append(pooled)
        text_ids_pool.append(text_ids)

    logger.info(f"Pre-encoded {len(prompt_embeds_pool)} prompts")
    
    # 如果后续不再需要在线编码新 prompt，可释放文本编码器与 tokenizer 以节省显存
    del text_encoder_clip, text_encoder_t5, tokenizer_clip, tokenizer_t5
    if "cuda" in device.type:
        torch.cuda.empty_cache()

    for sample in dataset:
        img = sample["image"].unsqueeze(0).to(device).to(mp_dtype)
        original = img.clone().to(mp_dtype)
        adv = img.clone().requires_grad_(True).to(mp_dtype)
        
        reference_image = sample["reference_image"].unsqueeze(0).to(device).to(mp_dtype)

        
        # 预先读取 VAE 的缩放参数
        vae_shift = vae.config.shift_factor
        vae_scale = vae.config.scaling_factor
        
        # import pdb; pdb.set_trace()
        # 预先缓存 reference image 分支的 latent（整个攻击过程不变）
        with torch.no_grad():
            cond_latents_ref = vae.encode(reference_image).latent_dist.mode()
            cond_latents_ref = (cond_latents_ref - vae_shift) * vae_scale
                
        # ===== cache packed_cond_ref 和 latent ids（只计算一次）=====

        # context latent ids（固定）
        cond_latent_ids = FluxKontextPipeline._prepare_latent_image_ids(
            cond_latents_ref.shape[0],
            cond_latents_ref.shape[2] // 2,
            cond_latents_ref.shape[3] // 2,
            device,
            mp_dtype
        )

        # 标记为 context
        cond_latent_ids[..., 0] = 1

        # reference context latent pack（固定）
        packed_cond_ref = FluxKontextPipeline._pack_latents(
            cond_latents_ref,
            cond_latents_ref.shape[0],
            cond_latents_ref.shape[1],
            cond_latents_ref.shape[2],
            cond_latents_ref.shape[3]
        )

        # target latent ids（固定 shape，因为 noise shape 固定）
        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            cond_latents_ref.shape[0],
            cond_latents_ref.shape[2] // 2,
            cond_latents_ref.shape[3] // 2,
            device,
            mp_dtype
        )

        # 拼接 ids（固定）
        combined_latent_ids = torch.cat([latent_image_ids, cond_latent_ids], dim=0)

        # 记录 batch size
        bsz = img.shape[0]

        hook = AttentionHook()

        for step in tqdm(range(args.attack_steps)):

            # 每一步随机采样一个编辑指令，直接读取预编码缓存
            prompt_idx = random.randint(0, len(prompt_pool) - 1)
            prompt = prompt_pool[prompt_idx]

            embeds = prompt_embeds_pool[prompt_idx]
            pooled = pooled_prompt_embeds_pool[prompt_idx]
            text_ids = text_ids_pool[prompt_idx]

            noise = torch.randn((1,16,64,64), device=device, dtype=mp_dtype)
            # 将当前对抗图像编码为 context latent
            cond_latents_adv = vae.encode(adv).latent_dist.mode()
            cond_latents_adv = (cond_latents_adv - vae_shift) * vae_scale

            # 将随机噪声视为 target noisy latent（与原始 DeContext 一致）
            noisy_latents = noise

            # pack target latent
            packed_noisy_latents = FluxKontextPipeline._pack_latents(
                noisy_latents,
                bsz,
                noisy_latents.shape[1],
                noisy_latents.shape[2],
                noisy_latents.shape[3]
            )

            # pack adv context latent
            packed_cond_adv = FluxKontextPipeline._pack_latents(
                cond_latents_adv,
                bsz,
                cond_latents_adv.shape[1],
                cond_latents_adv.shape[2],
                cond_latents_adv.shape[3]
            )

            # 拼接成 FLUX 所需输入：[target tokens | context tokens]
            packed_input_adv = torch.cat([packed_noisy_latents, packed_cond_adv], dim=1)
            packed_input_ref = torch.cat([packed_noisy_latents, packed_cond_ref], dim=1)

            # 采样高噪声区间的 timestep（与原始 DeContext 一致）
            bsz = 1
            timesteps = torch.randint(
                980,
                scheduler.config.num_train_timesteps,
                (bsz,),
                device=device
            ).long()

            # FLUX/DeContext 中实际送入 transformer 的是归一化后的时间
            timesteps_expanded = timesteps.expand(bsz).to(device=device, dtype=mp_dtype) / float(scheduler.config.num_train_timesteps)

            hook.clear()
            hook.register(
                transformer,
                selected_layers={"single_blocks": list(range(0, 26))}
            )
            
            # import pdb; pdb.set_trace()  # 调试点：检查输入输出维度、RoPE ids、attention maps 等关键变量的正确性
            guidance = torch.tensor([3.5], device=device, dtype=mp_dtype)
            
            t_pred = time.time()
            pred = transformer(
                hidden_states=packed_input_adv,
                timestep=timesteps_expanded,
                guidance=guidance.expand(bsz),
                encoder_hidden_states=embeds,
                pooled_projections=pooled,
                txt_ids=text_ids,
                img_ids=combined_latent_ids,
                return_dict=False
            )[0]
            pred_time = time.time() - t_pred
            
            # 缓存 adv 分支的 qkv，避免后续 ref forward 覆盖
            adv_qkv = {k: v.clone() for k, v in hook.qkv.items()}

            with torch.no_grad():
                ref_hook = AttentionHook()
                ref_hook.register(
                    transformer,
                    selected_layers={"single_blocks": list(range(0, 26))}
                )
                
                t_ref = time.time()
                ref_pred = transformer(
                    hidden_states=packed_input_ref,
                    timestep=timesteps_expanded,
                    guidance=guidance.expand(bsz),
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled,
                    txt_ids=text_ids,
                    img_ids=combined_latent_ids,
                    return_dict=False
                )[0]
                ref_time = time.time() - t_ref
                
                # 缓存 ref 分支的 qkv
                ref_qkv = {k: v.clone() for k, v in ref_hook.qkv.items()}

            # 按 [text tokens | target image tokens | context image tokens] 的顺序构造 RoPE ids
            ids = torch.cat([text_ids, latent_image_ids, cond_latent_ids], dim=0)
            rotary_emb = transformer.pos_embed(ids)

            if step % 10 == 0:
                hook_count = len(hook.hooks)
                ref_hook_count = len(ref_hook.hooks)

                if device.type == "cuda":
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
                    print(
                        f"[DEBUG] step={step} pred_time={pred_time:.2f}s ref_time={ref_time:.2f}s "
                        f"hook_count={hook_count} ref_hook_count={ref_hook_count} "
                        f"mem_alloc={mem_alloc:.1f}MB mem_reserved={mem_reserved:.1f}MB",
                        flush=True
                    )
                else:
                    print(
                        f"[DEBUG] step={step} pred_time={pred_time:.2f}s ref_time={ref_time:.2f}s "
                        f"hook_count={hook_count} ref_hook_count={ref_hook_count}",
                        flush=True
                    )
                    
            Lc, Ll, Lv, La, Lq = compute_losses(
                cond_latents_adv, cond_latents_ref,
                pred, ref_pred,
                adv_qkv, ref_qkv, rotary_emb
            )
            
            ref_hook.clear()
            hook.clear()
            
            if step % 50 == 0:
                print(f"[DEBUG] after clear: hook={len(hook.hooks)} ref_hook={len(ref_hook.hooks)}", flush=True)
            
            L = args.w_c * Lc - args.w_l * Ll - args.w_v * Lv - args.w_a * La - args.w_q * Lq

            transformer.zero_grad()
            vae.zero_grad()
            L.backward()

            adv = adv + args.alpha * adv.grad.sign()
            eta = torch.clamp(adv - original, -args.eps, args.eps)
            adv = torch.clamp(original + eta, -1, 1).detach().requires_grad_(True)

            if not args.no_wandb:
                try:
                    wandb.log({
                        # ===== losses =====
                        "loss/context": Lc.item(),
                        "loss/latent": Ll.item(),
                        "loss/velocity": Lv.item(),
                        "loss/attention": La.item(),
                        "loss/qk": Lq.item(),
                        "loss/total": L.item(),

                        # ===== weights（方便做 ablation 对照）=====
                        "weight/w_c": args.w_c,
                        "weight/w_l": args.w_l,
                        "weight/w_v": args.w_v,
                        "weight/w_a": args.w_a,
                        "weight/w_q": args.w_q,

                        # ===== misc =====
                        "step": step,
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                    })
                except Exception:
                    logger.debug("wandb.log failed for this step")

        # 保存结果
        with torch.no_grad():
            out = adv[0].detach().clamp(-1, 1).float().cpu().numpy().transpose(1, 2, 0)
            out = ((out + 1.0) * 127.5).astype(np.uint8)
            Image.fromarray(out).save(os.path.join(args.output_dir, Path(sample["path"]).name))

    # finish wandb run if enabled
    if not args.no_wandb:
        try:
            wandb.finish()
        except Exception:
            logger.debug("wandb.finish() failed or was not initialized")


if __name__ == "__main__":
    main()