#!/bin/bash
# Run PTQ evaluation using ProMix pipeline
# Uses torchrun for single-GPU (ResQ's modeling code requires distributed context)
cd $(dirname $0)/../..
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 \
    -m promix.eval.ptq --config promix/configs/llama-3.2-1b-resq.yaml
