#!/bin/bash

PRETRAINED_MODEL="/home/humw/Pretrains/black-forest-labs/FLUX.1-Kontext-dev"
CONDITION_IMAGES_DIR="./example"
OUTPUT_DIR="./perturbed_v3"

ALPHA=0.005
EPS=0.1
ATTACK_STEPS=800
SEED=42
MIXED_PRECISION="no"

# device to run on: set to "cuda" or "cpu"
DEVICE="cuda:3"

# set to 1 to disable wandb logging
NO_WANDB=0

# ===== 新增：loss weights =====
W_CONTEXT=1.0
W_LATENT=0.5
W_VELOCITY=1.0
W_ATTENTION=1.0
W_QK=0.2

mkdir -p "$OUTPUT_DIR"

# build command
CMD=(python -u ./attack/attack_Flux_Kontext/context_attack_v3.py
    --pretrained_model_name_or_path "$PRETRAINED_MODEL"
    --condition_images_dir "$CONDITION_IMAGES_DIR"
    --output_dir "$OUTPUT_DIR"
    --alpha "$ALPHA"
    --eps "$EPS"
    --attack_steps "$ATTACK_STEPS"
    --seed "$SEED"
    --mixed_precision "$MIXED_PRECISION"
    --device "$DEVICE"
    --w_context "$W_CONTEXT"
    --w_latent "$W_LATENT"
    --w_velocity "$W_VELOCITY"
    --w_attention "$W_ATTENTION"
    --w_qk "$W_QK"
)

if [ "$NO_WANDB" -ne 0 ]; then
    CMD+=(--no_wandb)
fi

"${CMD[@]}"