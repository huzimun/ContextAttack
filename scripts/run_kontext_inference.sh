#!/bin/bash

# Parameters (edit as needed)
CONTEXT_IMAGE="perturbed/adversarial_170.png"
# CONTEXT_IMAGE="example/170.jpg"
PROMPT="A photo of this person"
MODEL_PATH="/home/humw/Pretrains/black-forest-labs/FLUX.1-Kontext-dev"
OUTPUT_DIR="./outputs"
GUIDANCE_SCALE=3.5
NUM_INFERENCE_STEPS=20
DEVICE="cuda:3"
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
