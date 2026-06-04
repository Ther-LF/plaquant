#!/bin/bash
# Optimize rotation matrices (Step 1)
# Requires Step 0 (basis) to be completed first
# Run from plaquant root directory on remote H20
cd $(dirname $0)/../..
source /vllm-workspace/plaquant/.venv/bin/activate 2>/dev/null || true

LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 \
    -m promix.quantize.optimize_rotation \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation \
    --max_steps 100 \
    --learning_rate 1.5
