#!/bin/bash
# GB10 densesweep-recipe export for one 3B r point (mirrors serve_point.sh 3b)
set -u
R=$1
export PATH=/home/luofan/.venv/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/home/luofan/plaquant
export TORCH_CUDA_ARCH_LIST=12.1a
export HF_HOME=/home/luofan/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROMIX_ACT_MSE_STEPS_UP=2
export PROMIX_WEIGHT_MSE_STEPS=7
export PYTHONUNBUFFERED=1
cd /home/luofan/plaquant
CKPT=/home/luofan/gb10_sweep/ckpt/promix-3b-r${R}
echo "EXPORT_START r=${R} $(date)"
python -m promix.vllm_integ.export --config promix/configs/llama-3.2-3b-fp-perhead.yaml \
  --out $CKPT --substrate ring --ring_size 512 --alloc measured --resolve-paths \
  --high_fraction $R --gptq_nsamples 512 --percdamp 0.025 --no-perproj-ppl
echo "EXPORT_RC $? r=${R} $(date)"
