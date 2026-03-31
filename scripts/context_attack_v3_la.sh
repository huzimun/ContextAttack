#!/bin/bash

PRETRAINED_MODEL="/home/humw/Pretrains/black-forest-labs/FLUX.1-Kontext-dev"

ALPHA=0.005
EPS=0.1
ATTACK_STEPS=800
SEED=42
MIXED_PRECISION="bf16"

# shorthand for ATTACK_STEPS
STEPS="$ATTACK_STEPS"

# device to run on: set to "cuda" or "cpu"
DEVICE="cuda:3"

# set to 1 to disable wandb logging
NO_WANDB=0

# ===== loss weights =====
W_C=0.0
W_L=0.0
W_V=0.0
W_A=1.0
W_Q=0.0

CONDITION_IMAGES_DIR="./example"
REFERENCE_IMAGES_DIR="./target_image"
BASE_OUTPUT_DIR="./outputs/perturbed"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/wc-${W_C}_wl-${W_L}_wv-${W_V}_wa-${W_A}_wq-${W_Q}_eps-${EPS}_steps-${STEPS}_v1"

mkdir -p "$OUTPUT_DIR"

# build command
CMD=(python -u ./attack/attack_Flux_Kontext/context_attack_v3.py
    --pretrained_model_name_or_path "$PRETRAINED_MODEL"
    --condition_images_dir "$CONDITION_IMAGES_DIR"
    --reference_images_dir "$REFERENCE_IMAGES_DIR"
    --output_dir "$OUTPUT_DIR"
    --alpha "$ALPHA"
    --eps "$EPS"
    --attack_steps "$ATTACK_STEPS"
    --seed "$SEED"
    --mixed_precision "$MIXED_PRECISION"
    --device "$DEVICE"
    --w_c "$W_C"
    --w_l "$W_L"
    --w_v "$W_V"
    --w_a "$W_A"
    --w_q "$W_Q"
)

if [ "$NO_WANDB" -ne 0 ]; then
    CMD+=(--no_wandb)
fi

"${CMD[@]}"