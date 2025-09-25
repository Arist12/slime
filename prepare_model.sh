#!/bin/bash

# Model preparation pipeline script
# Usage: ./prepare_model.sh <model_name>
# Example: ./prepare_model.sh Qwen/Qwen3-4B

set -e  # Exit on any error

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Error: No model name provided"
    echo "Usage: $0 <model_name>"
    echo "Example: $0 Qwen/Qwen3-4B"
    exit 1
fi

MODEL_NAME="$1"
BASE_DIR="/mnt/local/hf_cache/hub"

# Extract just the model name part (after the slash) for directory naming
MODEL_DIR_NAME=$(basename "$MODEL_NAME")
LOCAL_DIR="${BASE_DIR}/${MODEL_DIR_NAME}"
TORCH_DIST_DIR="${LOCAL_DIR}_torch_dist"

echo "=========================================="
echo "Model Preparation Pipeline"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Download directory: $LOCAL_DIR"
echo "Torch dist directory: $TORCH_DIST_DIR"
echo "=========================================="

# Step 1: Download model from Hugging Face
echo "Step 1: Downloading model from Hugging Face..."
hf download "$MODEL_NAME" --local-dir "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Model download completed successfully"
else
    echo "✗ Model download failed"
    exit 1
fi

# Step 2: Convert model to Megatron format
echo "Step 2: Converting model to Megatron format..."

# Source the model configuration script
# Note: You may need to adjust this path or make it configurable
SCRIPT_PATH="scripts/models/${MODEL_DIR_NAME}.sh"

if [ -f "$SCRIPT_PATH" ]; then
    echo "Sourcing configuration from: $SCRIPT_PATH"
    source "$SCRIPT_PATH"
else
    echo "Warning: Configuration script not found at $SCRIPT_PATH"
    echo "You may need to create this script or modify the path"
    # You could add default MODEL_ARGS here if needed
    # MODEL_ARGS=("--your" "--default" "--args")
fi

# Convert model to Megatron format
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$LOCAL_DIR" \
    --save "$TORCH_DIST_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Model conversion completed successfully"
    echo "=========================================="
    echo "Pipeline completed successfully!"
    echo "Model available at: $TORCH_DIST_DIR"
    echo "=========================================="
else
    echo "✗ Model conversion failed"
    exit 1
fi
