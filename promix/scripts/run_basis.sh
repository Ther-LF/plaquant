#!/bin/bash
# Compute PCA basis for ResQ quantization
# Run from plaquant root directory on remote H20
cd $(dirname $0)/../..
source /vllm-workspace/plaquant/.venv/bin/activate 2>/dev/null || true

python -m promix.quantize.basis \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation \
    --nsamples 512
