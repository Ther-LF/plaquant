#!/bin/bash
# Run PTQ evaluation using ProMix pipeline
cd $(dirname $0)/../..
source /vllm-workspace/plaquant/.venv/bin/activate 2>/dev/null || true

LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
    promix/eval/ptq.py --config promix/configs/llama-3.2-1b-resq.yaml
