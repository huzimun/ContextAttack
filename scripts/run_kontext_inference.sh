#!/bin/bash

# Parameters (edit as needed)
BASE_CONTEXT_DIR="outputs/perturbed/"
CONTEXT_NAME="loss_weights_1.0_0.0_0.0_0.0_0.0_eps_0.1_steps_800_v1"
CONTEXT_IMAGE="${BASE_CONTEXT_DIR}/${CONTEXT_NAME}"
PROMPT="A photo of this person"
MODEL_PATH="/home/humw/Pretrains/black-forest-labs/FLUX.1-Kontext-dev"
# Base output parent directory
BASE_OUTPUT_DIR="outputs/edited"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONTEXT_NAME}"
GUIDANCE_SCALE=3.5
NUM_INFERENCE_STEPS=20
DEVICE="cuda:2"
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
