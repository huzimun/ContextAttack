#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Context Suppression + Semantic Alignment (v2)
支持多种语义对齐损失：VAE latent、Flow velocity、Attention map。
"""

import argparse
import logging
import math
import os
from pathlib import Path

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
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel
logger = get_logger(__name__)


# =========================
# 注意力 Hook：抓取 Q/K 归一化后的表示
# =========================
# DeContext 的关键思想是：上下文图像的信息主要通过多模态注意力传播到输出。
# 因此，这里并不直接优化最终生成图像，而是“插入”到 Transformer 内部，
# 抓取每层注意力中归一化后的 query / key（q_norm / k_norm），
# 然后显式计算“目标图像 token 对上下文图像 token 的注意力占比”。
# 这样做的优点是：
#   1) 目标函数更贴近上下文传播机制；
#   2) 不必真的跑完整采样解码，也能定义稳定的攻击/防御损失；
#   3) 与论文中“削弱 context pathway”的思想一致。
class AttentionHook:
    def __init__(self):
                # 保存每一层 hook 到的 q_norm / k_norm
        self.qkv_normalized = {}
        self.hooks = []
    
    def register_hooks(self, transformer, selected_layers=None):
        if selected_layers is None:
            selected_layers = {
                'single_blocks': list(range(0, 26))
            }
        
                # 针对某一层某一投影(q/k)构造 hook，缓存其输出
        def create_norm_hook(layer_name, proj_type):
            def hook_fn(module, input, output):
                if output is not None and hasattr(output, 'shape'):
                                        # clone 一份，避免后续原地操作影响缓存内容
                    self.qkv_normalized[f"{layer_name}_{proj_type}"] = output.clone()
            return hook_fn
        
                # double_blocks: 双流结构，通常对应文本/图像交互更显式的阶段
        for i in selected_layers.get('double_blocks', []):
            if i < len(transformer.transformer_blocks):
                block = transformer.transformer_blocks[i]
                if hasattr(block, 'attn'):
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'double_{i}', proj_type)
                            )
                            self.hooks.append(hook)
        
                # single_blocks: 单流结构，文本与图像 token 已拼接到统一序列中
        for i in selected_layers.get('single_blocks', []):
            if i < len(transformer.single_transformer_blocks):
                block = transformer.single_transformer_blocks[i]
                if hasattr(block, 'attn'):
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(block.attn, norm_layer):
                            hook = getattr(block.attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'single_{i}', proj_type)
                            )
                            self.hooks.append(hook)
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
                # 保存每一层 hook 到的 q_norm / k_norm
        self.qkv_normalized = {}

# =========================
# 文本提示编码
# =========================
# FLUX Kontext 同时使用两套文本编码器：
#   - CLIP text encoder：提供 pooled embedding，通常用于全局语义引导；
#   - T5 encoder：提供 token-level embedding，供 Transformer 在跨模态注意力中使用。
# 返回值：
#   prompt_embeds          : [B, L, C]，T5 的逐 token 表示；
#   pooled_prompt_embeds   : [B, C]，CLIP 的全局语义表示；
#   text_ids               : [L, 3]，供 FLUX 位置编码/模态编码使用的文本 token 标识。
# 注意：
#   text_ids 在这里全部初始化为 0，后续与图像 latent ids 拼接后共同构造 RoPE 所需的输入 id。
def encode_prompt_flux_kontext(text_encoders, tokenizers, prompt, max_sequence_length=512, device=None):
        # 统一成 batch 形式，便于 tokenizer / encoder 处理
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    # CLIP 分支：得到 pooled 全局语义表示
    text_inputs_clip = tokenizers[0](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        pooled_prompt_embeds = text_encoders[0](text_inputs_clip.input_ids.to(device)).pooler_output
    
    # T5 分支：得到逐 token 文本表示，供多模态 Transformer 使用
    text_inputs_t5 = tokenizers[1](
        prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoders[1](text_inputs_t5.input_ids.to(device))[0]
    
    # 文本 token 的 id 张量，后续会与图像 latent id 一起送入 RoPE
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids

# =========================
# 语义对齐损失函数
# =========================
def compute_latent_align_loss(vae, perturbed_img, reference_img, shift, scale):
    # VAE latent 对齐损失
    with torch.no_grad():
        ref_latent = vae.encode(reference_img).latent_dist.mode()
        ref_latent = (ref_latent - shift) * scale
    pert_latent = vae.encode(perturbed_img).latent_dist.mode()
    pert_latent = (pert_latent - shift) * scale
    return F.mse_loss(pert_latent, ref_latent)

def compute_velocity_align_loss(model_pred, ref_pred):
    # Flow velocity 对齐损失
    return F.mse_loss(model_pred, ref_pred)

def compute_attention_align_loss(attn_hook, ref_attn_hook):
    # Attention map 对齐损失
    loss = 0.0
    n = 0
    for k in attn_hook.qkv_normalized:
        if k in ref_attn_hook.qkv_normalized:
            loss = loss + F.mse_loss(attn_hook.qkv_normalized[k], ref_attn_hook.qkv_normalized[k])
            n += 1
    return loss / n if n > 0 else torch.tensor(0.0, device=next(iter(attn_hook.qkv_normalized.values())).device)

# =========================
# context抑制损失（与v1一致）
# =========================
def compute_context_suppress_loss(attention_hook, transformer, text_ids, latent_image_ids, cond_latent_ids_pert):
    """
    计算语义抑制损失（即context attention抑制），与v1一致。
    """
    # 重新拼接ids，生成RoPE
    ids_pert = torch.cat([text_ids, latent_image_ids, cond_latent_ids_pert], dim=0)
    image_rotary_emb_pert = transformer.pos_embed(ids_pert)
    # 计算context attention占比
    context_prop = compute_context_proportion(attention_hook.qkv_normalized, image_rotary_emb_pert)
    # 目标：抑制context attention（最小化context_prop）
    return context_prop

# =========================
# 命令行参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                       default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./example")
    parser.add_argument("--reference_images_dir", type=str, default=None, help="参考图像文件夹路径")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="./perturbed")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--align_loss_types", type=str, default="latent", help="对齐损失类型,逗号分隔: latent,velocity,attention")
    parser.add_argument("--align_loss_weights", type=str, default="1.0,1.0,1.0", help="对齐损失权重,与类型顺序对应")
    return parser.parse_args()


# =========================
# 数据集定义（支持参考图像/特征）
# =========================
class AdversarialDatasetV2(Dataset):
    def __init__(self, condition_images_dir, reference_images_dir=None, prompts=None, resolution=512):
        self.condition_images_dir = Path(condition_images_dir)
        self.reference_images_dir = Path(reference_images_dir) if reference_images_dir else None
        self.prompts = prompts
        self.resolution = resolution

        self.cond_files = sorted([f for f in self.condition_images_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        if self.reference_images_dir:
            self.ref_files = sorted([f for f in self.reference_images_dir.iterdir() 
                                    if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
            assert len(self.cond_files) == len(self.ref_files), "参考图像数量需与条件图像一致"
        else:
            self.ref_files = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.cond_files)
    
    def __getitem__(self, idx):
        cond_path = self.cond_files[idx]
        cond_image = Image.open(cond_path).convert('RGB')
        cond_image = self.image_transforms(cond_image)

        sample = {
            'condition_image': cond_image,
            'cond_path': str(cond_path)
        }
        if self.ref_files:
            ref_path = self.ref_files[idx]
            ref_image = Image.open(ref_path).convert('RGB')
            ref_image = self.image_transforms(ref_image)
            sample['reference_image'] = ref_image
            sample['ref_path'] = str(ref_path)
        return sample
    
# =========================
# 主流程
# =========================
def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                       datefmt="%H:%M:%S", level=logging.INFO)
    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # prompt池
    prompt_pool = ["Make this person smile happily."]  # 可扩展
    logger.info(f"Loaded {len(prompt_pool)} prompts in the pool")

    # 加载模型
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
    transformer.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    # 数据集
    dataset = AdversarialDatasetV2(
        condition_images_dir=args.condition_images_dir,
        reference_images_dir=args.reference_images_dir,
        prompts=prompt_pool,
        resolution=args.resolution
    )
    logger.info(f"Loaded {len(dataset)} condition images")

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    # 解析对齐损失类型和权重
    align_types = [s.strip() for s in args.align_loss_types.split(",") if s.strip()]
    align_weights = [float(s) for s in args.align_loss_weights.split(",")]
    align_type2weight = {k: align_weights[i] if i < len(align_weights) else 1.0 for i, k in enumerate(align_types)}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample['cond_path']
        reference_image = sample.get('reference_image', None)
        if reference_image is not None:
            reference_image = reference_image.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        else:
            reference_image = condition_image.clone().detach()

        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")
        perturbed_condition = condition_image.clone().detach()
        perturbed_condition.requires_grad_(True)
        original_condition = condition_image.clone().detach()
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")
        guidance = torch.tensor([3.5], device=accelerator.device)

        # 预先计算参考特征
        with torch.no_grad():
            ref_latent = vae.encode(reference_image).latent_dist.mode()
            ref_latent = (ref_latent - vae_config_shift_factor) * vae_config_scaling_factor
        ref_attn_hook = None
        ref_pred = None

        for step in progress_bar:
            perturbed_condition.requires_grad = True
            # 1. VAE latent对齐损失
            losses = {}
            if 'latent' in align_type2weight:
                losses['latent'] = compute_latent_align_loss(
                    vae, perturbed_condition, reference_image, vae_config_shift_factor, vae_config_scaling_factor)

            # 2. 其余损失需前向传播
            # 注册Attention hook
            attention_hook = AttentionHook()
            attention_hook.register_hooks(transformer)
            # prompt编码
            prompt = prompt_pool[0]
            text_encoders = [text_encoder_clip, text_encoder_t5]
            tokenizers = [tokenizer_clip, tokenizer_t5]
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_flux_kontext(
                text_encoders, tokenizers, prompt, args.max_sequence_length, accelerator.device)
            bsz = 1
            timesteps = torch.randint(980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            dummy_target_shape = (bsz, 16, args.resolution // 8, args.resolution // 8)
            noise = torch.randn(dummy_target_shape, device=accelerator.device, dtype=weight_dtype)
            noisy_latents = noise
            cond_latents_pert = vae.encode(perturbed_condition).latent_dist.mode()
            cond_latents_pert = (cond_latents_pert - vae_config_shift_factor) * vae_config_scaling_factor
            latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                bsz, cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype)
            cond_latent_ids_pert = FluxKontextPipeline._prepare_latent_image_ids(
                cond_latents_pert.shape[0], cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype)
            cond_latent_ids_pert[..., 0] = 1
            combined_latent_ids_pert = torch.cat([latent_image_ids, cond_latent_ids_pert], dim=0)
            packed_noisy_latents = FluxKontextPipeline._pack_latents(
                noisy_latents, bsz, noisy_latents.shape[1], noisy_latents.shape[2], noisy_latents.shape[3])
            packed_cond_pert = FluxKontextPipeline._pack_latents(
                cond_latents_pert, bsz, cond_latents_pert.shape[1], cond_latents_pert.shape[2], cond_latents_pert.shape[3])
            packed_input_pert = torch.cat([packed_noisy_latents, packed_cond_pert], dim=1)
            attention_hook.clear_hooks()
            attention_hook.register_hooks(transformer)
            timesteps_expanded = timesteps.expand(packed_noisy_latents.shape[0]).to(packed_noisy_latents.dtype)
            model_pred = transformer(
                hidden_states=packed_input_pert,
                timestep=timesteps_expanded / 1000,
                guidance=guidance.expand(bsz),
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_latent_ids_pert,
                return_dict=False
            )[0]

            # 3. 参考velocity/attention特征（仅需一次）
            if ref_pred is None and ('velocity' in align_type2weight or 'attention' in align_type2weight):
                ref_attn_hook = AttentionHook()
                ref_attn_hook.register_hooks(transformer)
                with torch.no_grad():
                    ref_cond_latents = vae.encode(reference_image).latent_dist.mode()
                    ref_cond_latents = (ref_cond_latents - vae_config_shift_factor) * vae_config_scaling_factor
                    ref_packed_cond = FluxKontextPipeline._pack_latents(
                        ref_cond_latents, bsz, ref_cond_latents.shape[1], ref_cond_latents.shape[2], ref_cond_latents.shape[3])
                    ref_packed_input = torch.cat([packed_noisy_latents, ref_packed_cond], dim=1)
                    ref_pred = transformer(
                        hidden_states=ref_packed_input,
                        timestep=timesteps_expanded / 1000,
                        guidance=guidance.expand(bsz),
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=combined_latent_ids_pert,
                        return_dict=False
                    )[0]
                # ref_attn_hook.qkv_normalized已被填充

            # 4. velocity/attention对齐损失
            if 'velocity' in align_type2weight:
                losses['velocity'] = compute_velocity_align_loss(model_pred, ref_pred)
            if 'attention' in align_type2weight:
                losses['attention'] = compute_attention_align_loss(attention_hook, ref_attn_hook)

            # 5. context抑制损失（与v1一致，目标为最小化context attention占比）
            context_suppress_loss = compute_context_suppress_loss(attention_hook, transformer, text_ids, latent_image_ids, cond_latent_ids_pert)

            # 6. 动态加权多目标损失（可自定义权重策略，这里简单示例：对齐损失和抑制损失各0.5）
            align_loss = 0.0
            for k, v in losses.items():
                align_loss = align_loss + align_type2weight.get(k, 1.0) * v
            # 可根据实际需求动态调整权重
            suppress_weight = 0.5
            align_weight = 0.5
            L_goal = align_weight * align_loss + suppress_weight * context_suppress_loss

            # 7. 反传与PGD更新
            transformer.zero_grad()
            vae.zero_grad()
            L_goal.backward()
            adv_condition = perturbed_condition + args.alpha * perturbed_condition.grad.sign()
            eta = torch.clamp(adv_condition - original_condition, min=-args.eps, max=args.eps)
            perturbed_condition = torch.clamp(original_condition + eta, min=-1, max=1).detach_()
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()
            progress_bar.set_postfix({
                'L_goal': f'{L_goal.item():.4f}',
                'align_loss': f'{align_loss.item():.4f}',
                'suppress_loss': f'{context_suppress_loss.item():.4f}',
                'pert': f'{perturbation_norm:.4f}'
            })
            attention_hook.clear_hooks()
        del progress_bar, guidance, latent_image_ids
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