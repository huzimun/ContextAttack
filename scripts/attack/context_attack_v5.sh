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
DEVICE="cuda:2"

# set to 1 to disable wandb logging
NO_WANDB=0


# ===== loss weights =====
# velocity：qk：latent：attention=10：1：0.1：0.0001，将所有损失放缩到同一级别
W_L=0.0 #10.0
W_V=0.0 # 0.1
W_A=10000.0
W_Q=1.0 # 1

# ===== prompt mode =====
PROMPT_MODE="single"  # "multi" or "single" 

CONDITION_IMAGES_DIR="./example"
# target_name="Obama"
# REFERENCE_IMAGES_DIR="./target_image/${target_name}"
REFERENCE_IMAGES_DIR=$CONDITION_IMAGES_DIR # 目标参考图像也是原始图像
BASE_OUTPUT_DIR="./outputs/perturbed"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/wl-${W_L}_wv-${W_V}_wa-${W_A}_wq-${W_Q}_eps-${EPS}_steps-${STEPS}_prompt-${PROMPT_MODE}_v4"

mkdir -p "$OUTPUT_DIR"


# ===== 可选参数：prompt后缀和attention层 =====
# 参考分支prompt后缀（可自定义）
REF_PROMPT_SUFFIX="eyes bulging and mouth wide open, with blood streaming all over the countenance"
# attention block层（如"0-25"或"0,1,2,3"）
ATTN_LAYERS="0-25"

CMD=(python -u ./attack/attack_Flux_Kontext/context_attack_v5.py
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
    --w_l "$W_L"
    --w_v "$W_V"
    --w_a "$W_A"
    --w_q "$W_Q"
    --prompt_mode "$PROMPT_MODE"
    --ref_prompt_suffix "$REF_PROMPT_SUFFIX"
    --attn_layers "$ATTN_LAYERS"
)

if [ "$NO_WANDB" -ne 0 ]; then
    CMD+=(--no_wandb)
fi

"${CMD[@]}"


#!/bin/bash

# Parameters (edit as needed)
# BASE_CONTEXT_DIR="outputs/perturbed/"
# CONTEXT_NAME="example2"
# CONTEXT_IMAGE="${BASE_CONTEXT_DIR}/${CONTEXT_NAME}"
CONTEXT_IMAGE="${BASE_OUTPUT_DIR}/wl-${W_L}_wv-${W_V}_wa-${W_A}_wq-${W_Q}_eps-${EPS}_steps-${STEPS}_prompt-${PROMPT_MODE}_v4"
# PROMPT="eyes bulging and mouth wide open, with blood streaming all over the countenance"
PROPT="A photo of this person"
MODEL_PATH="/home/humw/Pretrains/black-forest-labs/FLUX.1-Kontext-dev"
# Base output parent directory
# BASE_OUTPUT_DIR="outputs/edited"
OUTPUT_DIR="outputs/edited/wl-${W_L}_wv-${W_V}_wa-${W_A}_wq-${W_Q}_eps-${EPS}_steps-${STEPS}_prompt-${PROMPT_MODE}_v4"
GUIDANCE_SCALE=3.5
NUM_INFERENCE_STEPS=20
SEED=42
WIDTH=512
HEIGHT=512

mkdir -p "$OUTPUT_DIR"

python3 "./inference/kontext_inference.py" \
	--context_image "$CONTEXT_IMAGE" \
	--prompt "$PROMPT" \
	--model_path "$MODEL_PATH" \
	--output_dir "$OUTPUT_DIR" \
	--guidance_scale "$GUIDANCE_SCALE" \
	--num_inference_steps "$NUM_INFERENCE_STEPS" \
	--device "$DEVICE" \
	--seed "$SEED" \
	--width "$WIDTH" \
	--height "$HEIGHT"
