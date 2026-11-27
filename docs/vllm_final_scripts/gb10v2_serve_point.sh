#!/bin/bash
# GB10 v2 serving point: tuned kernel stack (KDA-3/4), decimal-grid protocol.
# batches 16,64,128 (b256 saturation verify on r0.5 only); qserve 1024:512 same batches.
set -u
R=$1
CK=/home/luofan/gb10_sweep/ckpt/promix-3b-r${R}
TAG=gb10v2_pm3b_r${R}
B=/home/luofan/gb10_sweep/bench
BATCHES=16,64,128
if [ "$R" = "0.5" ]; then BATCHES=16,64,128,256; fi
source /home/luofan/gb10_sweep/env.sh
cd /home/luofan/plaquant
echo "SERVE_START $TAG batches=$BATCHES $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_main --dtype float16 \
  --batches $BATCHES --cases 1024:128,128:1024 --prefill-len 2048 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_main.json > $B/${TAG}_main.log 2>&1
echo "MAIN_RC $? $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_qserve --dtype float16 \
  --batches $BATCHES --cases 1024:512 --prefill-len 0 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_qserve.json > $B/${TAG}_qserve.log 2>&1
echo "QSERVE_RC $? $(date)"
