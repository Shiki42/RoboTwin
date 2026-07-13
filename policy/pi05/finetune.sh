#!/usr/bin/env bash
set -euo pipefail

train_config_name=$1
model_name=$2
gpu_use=$3
runtime_ld_path=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

export CUDA_VISIBLE_DEVICES="$gpu_use"
printf '%s\n' "$CUDA_VISIBLE_DEVICES"
LD_LIBRARY_PATH="$runtime_ld_path" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONNOUSERSITE=1 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run scripts/train.py "$train_config_name" --exp-name="$model_name" --overwrite
