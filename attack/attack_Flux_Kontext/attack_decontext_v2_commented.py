#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Context Suppression
Minimizes attention weight proportion allocated to context image.
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
# 数据集定义
# =========================
# 该数据集仅负责读取“条件图像/上下文图像”（condition image）。
# 与传统分类数据集不同，这里没有标签；攻击的目标由后续随机采样的编辑指令 prompt 决定。
# 每个样本返回：
#   1) condition_image: 归一化到 [-1, 1] 的张量，作为待攻击输入；
#   2) cond_path: 原始图像路径，便于保存攻击结果时命名。
class AdversarialDataset(Dataset):
    def __init__(self, condition_images_dir, prompts, resolution=512):
        self.condition_images_dir = Path(condition_images_dir)
        self.prompts = prompts 
        self.resolution = resolution

                # 收集目录下全部图像文件，作为待保护/待攻击的条件输入
        self.cond_files = sorted([f for f in self.condition_images_dir.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
                # 与 FLUX 常见预处理一致：缩放、中心裁剪、转 tensor、归一化到 [-1, 1]
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
        
        return {
            'condition_image': cond_image,
            'cond_path': str(cond_path)
        }


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
# 根据离散时间步查询 scheduler 对应的 sigma
# =========================
# 这个函数是通用辅助函数：给定 timesteps，从 scheduler 中取出对应噪声强度 sigma，
# 并将其 reshape 到可与 latent 广播的维度。
# 当前版本主流程没有显式使用该函数，但它保留了 flow matching / diffusion 攻击中
# 常见的时间步-噪声幅值映射逻辑，便于后续扩展。
def get_sigmas(timesteps, scheduler, n_dim=4, dtype=torch.float32, device=None):
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if len(matches) > 0:
            step_indices.append(matches[0].item())
        else:
            step_indices.append(torch.argmin(torch.abs(schedule_timesteps - t)).item())
    
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


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
# 计算“上下文注意力占比”
# =========================
# 输入：
#   qkv_normalized_dict : 由 hook 抓取到的各层 q_norm / k_norm；
#   image_rotary_emb    : 当前 token 序列对应的旋转位置编码（RoPE）。
#
# 核心思想：
#   对每个选定层，显式重构 target query -> 全部 key 的注意力分布，
#   然后统计其中分配给“context image token”的质量占比。
#
# 若该占比越大，说明模型更依赖条件图像上下文；若越小，则说明上下文被“去耦合”得更充分。
# DeContext/本代码的优化目标正是围绕这一量展开。
def compute_context_proportion(qkv_normalized_dict, image_rotary_emb):
    """
    Compute proportion of attention weight allocated to context image.
    
    Returns:
        context_prop: Average proportion across all layers [scalar]
    """
    # RoPE 参数（在本函数中不单独展开使用，但 apply_rotary_emb 需要这一对张量）
    freqs_cos, freqs_sin = image_rotary_emb
    proportions = []
    
    for layer_name in qkv_normalized_dict:
        if not layer_name.endswith('_q'):
            continue
        
        layer_base = layer_name.replace('_q', '')
        q_key = f"{layer_base}_q"
        k_key = f"{layer_base}_k"
        
        if q_key not in qkv_normalized_dict or k_key not in qkv_normalized_dict:
            continue
        
        q_norm = qkv_normalized_dict[q_key]
        k_norm = qkv_normalized_dict[k_key]
        
        # q_norm / k_norm 形状通常为 [B, S, H, D]
        batch_size, seq_len, heads, dim_head = q_norm.shape
        
        # Determine sequence structure 
        # (default with 512*512, the sequence length needs to be adjusted with context images of different resolutions)
        # 根据层类型推断 token 序列布局。
        # 这里的切片假定输入分辨率为 512，且 FLUX 的 pack 方式不变；
        # 若分辨率变化，这些边界也应同步调整。
        if 'double_' in layer_base:
            if seq_len != 2048:
                continue
            target_start, target_end = 0, 1024
            text_start, text_end = None, None
            context_start, context_end = 1024, 2048
        elif 'single_' in layer_base:
            if seq_len != 2560:
                continue
            target_start, target_end = 512, 1536
            text_start, text_end = 0, 512
            context_start, context_end = 1536, 2560
        else:
            continue
        
        # Apply RoPE
        # 将旋转位置编码施加到 q/k，以与模型真实注意力计算保持一致
        q_with_rope = apply_rotary_emb(q_norm, image_rotary_emb, sequence_dim=1)
        k_with_rope = apply_rotary_emb(k_norm, image_rotary_emb, sequence_dim=1)
        
        # Compute full attention scores
        # 仅保留“目标图像区域”作为 query；我们关心的是输出目标如何关注上下文
        q_target_with_rope = q_with_rope[:, target_start:target_end, :, :]  # [B, 1024, H, D]
        q_target_with_rope = q_target_with_rope.permute(0, 2, 1, 3)  # [B, H, 1024, D]
        k_with_rope = k_with_rope.permute(0, 2, 1, 3)
        
        # print(q_target_with_rope.shape)
        # 这里手动重构缩放点积注意力分数
        target_attn_scores = q_target_with_rope @ k_with_rope.transpose(-2, -1) / math.sqrt(dim_head)  # [B, H, S, S]
        # print(target_attn_scores.shape)
        
        # Softmax over all keys (text + target + context)
        attn_weights = F.softmax(target_attn_scores, dim=-1)  # [B, H, 1024, S]

        # Extract context weights
        context_weights = attn_weights[:, :, :, context_start:context_end]  # [B, H, 1024, 1024]
          
        # Compute proportion: sum over context keys / total (which is 1 after softmax)
        # 因为 softmax 后每个 query 的总注意力和为 1，
        # 对 context 区间求和即可得到“分给 context 的占比”
        context_prop_per_query = context_weights.sum(dim=-1)  # [B, H, 1024]
        
        # Average over heads and queries
        layer_prop = context_prop_per_query.mean()  # scalar

        proportions.append(layer_prop)
    
    if len(proportions) == 0:
        return torch.tensor(0.0)
    
    return torch.stack(proportions).mean()



# def sample_timestep_with_bias(current_step, total_steps, num_train_timesteps, device, batch_size=1):
#     progress = current_step / total_steps  
#     min_timestep = int((1 - progress) * 0 + progress * 980) 
#     max_timestep = num_train_timesteps
    
#     timesteps = torch.randint(
#         min_timestep, 
#         max_timestep, 
#         (batch_size,), 
#         device=device
#     ).long()
    
#     return timesteps



# =========================
# 命令行参数
# =========================
# alpha : 每步 PGD 更新步长
# eps   : 最大扰动半径（L_inf 约束）
# attack_steps : PGD 迭代轮数
# mixed_precision : 推理/反传所用精度
# max_sequence_length : T5 文本最大长度
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, 
                       default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./example")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="./perturbed")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


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
# 主流程
# =========================
# 整体流程可以概括为：
#   1) 载入 FLUX Kontext 所需的 tokenizer / text encoder / VAE / Transformer / scheduler；
#   2) 预编码全部 prompt，减少攻击迭代时的重复开销；
#   3) 对每张条件图像执行 PGD：
#        - 随机采样一个 prompt；
#        - 将当前扰动图像编码到 VAE latent；
#        - 与随机 target latent/noise 拼接，模拟编辑时的联合输入；
#        - hook 注意力内部 q/k；
#        - 计算“目标 token 对 context token 的注意力占比”；
#        - 依据目标函数反传到输入图像；
#        - 在 L_inf 约束下做一次 PGD 更新；
#   4) 保存最终对抗图像。
#
# 从机制上看，这并不是传统“让最终输出分类错误”的攻击，
# 而是直接削弱上下文图像在 DiT 编辑过程中的信息流通能力。
def main():
    # 解析命令行输入
    args = parse_args()
    
    # Accelerate 用于统一设备、精度与分布式接口；此脚本默认走 GPU
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", 
                       datefmt="%H:%M:%S", level=logging.INFO)
    # import pdb; pdb.set_trace()
    if args.seed is not None:
        # 固定随机种子，尽量保证攻击过程可复现
        set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # get prompt pool
    # 构造编辑 prompt 池
    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts in the pool")
    
    # Load tokenizers and models
    tokenizer_clip = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    
    # 载入 FLUX Kontext 所需的全部组件
    logger.info("Loading models...")
    text_encoder_clip = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_t5 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Set precision
    # 根据 mixed precision 设定权重精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    text_encoder_clip.to(accelerator.device, dtype=weight_dtype)
    text_encoder_t5.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    
    # 全部模块工作在 eval 模式；我们只对输入图像求梯度，不更新模型参数
    text_encoder_clip.eval()
    text_encoder_t5.eval()
    vae.eval()
    transformer.eval()
    
    text_encoder_clip.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    
    # 即使不训练模型，也可在反传到输入图像时节省显存
    transformer.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()
    
    # Load dataset
    # 构造条件图像数据集
    dataset = AdversarialDataset(
        condition_images_dir=args.condition_images_dir,
        prompts=prompt_pool,
        resolution=args.resolution
    )
    
    logger.info(f"Loaded {len(dataset)} condition images")
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    
    text_encoders = [text_encoder_clip, text_encoder_t5]
    tokenizers = [tokenizer_clip, tokenizer_t5]
    
    # Pre-encode all prompts
    logger.info("Pre-encoding all prompts...")
    # 预编码所有 prompt，避免在每个 PGD step 重复跑文本编码器
    prompt_embeds_pool = []
    # 预编码所有 prompt，避免在每个 PGD step 重复跑文本编码器
    pooled_prompt_embeds_pool = []

    for prompt in prompt_pool:  # 离线缓存文本表示
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt_flux_kontext(
                text_encoders, tokenizers, prompt, args.max_sequence_length, accelerator.device
            )
            prompt_embeds_pool.append(prompt_embeds)
            pooled_prompt_embeds_pool.append(pooled_prompt_embeds)
    
    logger.info(f"Pre-encoded {len(prompt_embeds_pool)} prompts")
    
    # release
    # 文本编码完成后立即释放对应模块，降低显存占用
    del text_encoder_clip, text_encoder_t5, tokenizer_clip, tokenizer_t5
    # 主动释放缓存，缓解大模型攻击时的显存压力
    torch.cuda.empty_cache()
    
    attention_hook = AttentionHook()
    
    # 逐张图像执行攻击/防御扰动优化
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample['cond_path']
        
        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")
        
        # 当前可优化变量：对抗条件图像
        perturbed_condition = condition_image.clone().detach()
        perturbed_condition.requires_grad_(True)
        # 保存原始图像，用于投影回 epsilon-ball
        original_condition = condition_image.clone().detach()
        
        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")
        
        # Fixed values
        # FLUX 推理中的 guidance 强度；攻击阶段固定为常数
        guidance = torch.tensor([3.5], device=accelerator.device)
        
        # g = torch.zeros_like(original_condition)
        # mu = 0.8
        for step in progress_bar:
            # randomly select a prompt at each step
            # 每一步随机采样一个编辑指令，提升扰动的泛化性
            random_prompt_idx = torch.randint(0, len(prompt_pool), (1,)).item()
            current_prompt = prompt_pool[random_prompt_idx]
            prompt_embeds = prompt_embeds_pool[random_prompt_idx]
            pooled_prompt_embeds = pooled_prompt_embeds_pool[random_prompt_idx]
            
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=prompt_embeds.dtype)
            
            bsz = 1
            
            # Aample a high timestep
            # 重点采样高噪声/早期去噪阶段。DeContext 指出这些阶段对上下文传播更关键。
            timesteps = torch.randint(980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()

            # Alternative way
            # if step % 10 == 0:
            #     timesteps = torch.tensor([1000], device=accelerator.device).long()
            # else:
            #     重点采样高噪声/早期去噪阶段。DeContext 指出这些阶段对上下文传播更关键。
            #     timesteps = torch.randint(980, scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
            
            # sample a noise as target
            dummy_target_shape = (bsz, 16, args.resolution // 8, args.resolution // 8)  
            # 构造目标 latent 的噪声初始化；这里不需要真实目标图像，只需模拟编辑过程中的目标流
            noise = torch.randn(dummy_target_shape, device=accelerator.device, dtype=weight_dtype)
            
            t_normalized = (timesteps.float() / scheduler.config.num_train_timesteps).to(dtype=weight_dtype)
            t_normalized = t_normalized.view(-1, 1, 1, 1)
            
            # noisy latents
            # 在该实现中直接把 noise 视作 noisy target latent
            noisy_latents = noise  
            
            perturbed_condition.requires_grad = True
            
            # Encode perturbed condition
            # 将当前扰动图像编码到 VAE latent 空间，作为 context image 的 latent 表示
            cond_latents_pert = vae.encode(perturbed_condition).latent_dist.mode()
            cond_latents_pert = (cond_latents_pert - vae_config_shift_factor) * vae_config_scaling_factor
            
            # Prepare latent IDs
            # 为“目标图像 latent token”准备位置/模态 id
            latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                bsz, cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype
            )
            
            # 为“条件图像 latent token”准备位置/模态 id
            cond_latent_ids_pert = FluxKontextPipeline._prepare_latent_image_ids(
                cond_latents_pert.shape[0], cond_latents_pert.shape[2] // 2, cond_latents_pert.shape[3] // 2,
                accelerator.device, weight_dtype
            )
            # 将第 0 维 id 置为 1，用于区分 context image token 与 target image token
            cond_latent_ids_pert[..., 0] = 1
            # 拼接 target/context 两部分 token id，供 Transformer 位置编码使用
            combined_latent_ids_pert = torch.cat([latent_image_ids, cond_latent_ids_pert], dim=0)
            
            # Pack latents
            # 将 2D latent pack 成 FLUX Transformer 期望的 token 序列格式
            packed_noisy_latents = FluxKontextPipeline._pack_latents(
                noisy_latents, bsz, noisy_latents.shape[1], noisy_latents.shape[2], noisy_latents.shape[3]
            )
            
            packed_cond_pert = FluxKontextPipeline._pack_latents(
                cond_latents_pert, bsz, cond_latents_pert.shape[1], cond_latents_pert.shape[2], cond_latents_pert.shape[3]
            )
            # 最终输入序列：前半是 target/noisy latent，后半是 context latent
            packed_input_pert = torch.cat([packed_noisy_latents, packed_cond_pert], dim=1)
            
            # Register hooks
            # 每一步先清空旧 hook 与缓存，避免跨 step 污染
            attention_hook.clear_hooks()
            # 在选定层注册 hook，用于抓取 q/k
            attention_hook.register_hooks(transformer)
            
            timesteps_expanded = timesteps.expand(packed_noisy_latents.shape[0]).to(packed_noisy_latents.dtype)
            
            # Forward pass
            # 前向传播：这里只是为了触发内部注意力计算与 hook 缓存
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
            
            # Compute RoPE
            # 重新按 [text, target image, context image] 顺序拼接 ids，以生成对应的 RoPE
            ids_pert = torch.cat([text_ids, latent_image_ids, cond_latent_ids_pert], dim=0)
            image_rotary_emb_pert = transformer.pos_embed(ids_pert)
            
            # Compute context proportion
            # 计算当前内部注意力中，目标 token 分配给 context token 的平均注意力占比
            context_prop = compute_context_proportion(
                attention_hook.qkv_normalized,
                image_rotary_emb_pert
            )
            
            # Loss: minimize context proportion
            # 目标函数：当前实现写成 1 - context_prop。
            # 从语义上看，若希望“抑制上下文占比”，更直接的最小化目标通常应是 context_prop。
            # 现代码在 PGD 更新时使用 +alpha*sign(grad) 的上升方向，因此等价于在最大化 L_context。
            # 这会驱动 context_prop 下降，逻辑上是成立的。
            L_context = 1 - context_prop
            L_goal = L_context
            
            # Backward
            transformer.zero_grad()
            vae.zero_grad()
            # 将内部注意力目标反传到输入图像像素
            L_goal.backward()

            
            # print(f"Grad max: {perturbed_condition.grad.max().item():.6f}")
            # print(f"Grad min: {perturbed_condition.grad.min().item():.6f}")

           # grad = perturbed_condition.grad
           # grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-8)
           # g = mu * g + grad_norm
            
            # PGD update
            # 标准 L_inf-PGD：沿梯度符号方向更新
            adv_condition = perturbed_condition + args.alpha * perturbed_condition.grad.sign()
            # 投影回以原图为中心、半径 eps 的 L_inf 球
            eta = torch.clamp(adv_condition - original_condition, min=-args.eps, max=args.eps)
            # 同时保证像素范围仍在 [-1, 1]，并 detach 切断旧计算图
            perturbed_condition = torch.clamp(original_condition + eta, min=-1, max=1).detach_()
            perturbation_norm = (perturbed_condition - original_condition).abs().max().item()

            progress_bar.set_postfix({
                'prompt_idx': random_prompt_idx,
                'timestep': timesteps.item(),
                'ctx_prop': f'{context_prop.item():.4f}',
                # 'L_goal': f'{L_goal.item():.4f}',
                'pert': f'{perturbation_norm:.4f}'
            })
 
            del L_context, L_goal, model_pred, cond_latents_pert, packed_cond_pert
            del packed_input_pert, adv_condition, context_prop, noisy_latents, noise
            # 主动释放缓存，缓解大模型攻击时的显存压力
            torch.cuda.empty_cache()
            
            # 每一步先清空旧 hook 与缓存，避免跨 step 污染
            attention_hook.clear_hooks()
        del progress_bar, guidance, latent_image_ids
        # 主动释放缓存，缓解大模型攻击时的显存压力
        torch.cuda.empty_cache()
        
        # Save final results
        with torch.no_grad():
            # 保存最终对抗图像（从 [-1,1] 反归一化回 uint8 RGB）
            output_path = os.path.join(args.output_dir, f"adversarial_{Path(cond_path).stem}.png")
            perturbed_np = perturbed_condition.detach().squeeze(0).cpu().float().numpy()
            perturbed_np = (perturbed_np + 1.0) * 127.5
            perturbed_np = np.transpose(perturbed_np, (1, 2, 0)).astype(np.uint8)
            Image.fromarray(perturbed_np).save(output_path)
        
            logger.info(f"Saved final result: {output_path}")
    
    logger.info("Attack completed!")


if __name__ == "__main__":
    main()