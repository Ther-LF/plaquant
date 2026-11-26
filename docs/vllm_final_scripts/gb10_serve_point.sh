#!/bin/bash
# GB10 serving point: committed sweep protocol (docs/vllm_final_scripts/run_sweep_point.sh)
# klow_pad option (a): PROMIX_KLOW_PAD=0 (producers do not support the 128-pad).
set -u
R=$1
CK=/home/luofan/gb10_sweep/ckpt/promix-3b-r${R}
TAG=gb10_pm3b_r${R}
B=/home/luofan/gb10_sweep/bench
source /home/luofan/gb10_sweep/env.sh
cd /home/luofan/plaquant
echo "SERVE_START $TAG $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_main --dtype float16 \
  --batches 16,64,128,256 --cases 1024:128,128:1024 --prefill-len 2048 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_main.json > $B/${TAG}_main.log 2>&1
echo "MAIN_RC $? $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_qserve --dtype float16 \
  --batches 128,256 --cases 1024:512 --prefill-len 0 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_qserve.json > $B/${TAG}_qserve.log 2>&1
echo "QSERVE_RC $? $(date)"
