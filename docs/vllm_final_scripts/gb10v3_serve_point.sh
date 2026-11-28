#!/bin/bash
# GB10 v3 serving point: P0-1 same-boot baseline comparison group.
# Same stack/protocol as gb10v2 (graphs ON, prefix caching OFF, greedy+ignore_eos,
# 1 warmup + median of 3), decimal-grid batches.
#   main  : cases 1024:128,128:1024  b {16,64,128} + prefill TTFT L2048 sweep
#   qserve: case  1024:512           b {16,64,128,256}
# Usage: gb10v3_serve_point.sh <ckpt> <tag> <dtype>
set -u
CK=$1
TAG=$2
DT=$3
B=/home/luofan/gb10_sweep/bench
source /home/luofan/gb10_sweep/env.sh
cd /home/luofan/plaquant
echo "SERVE_START $TAG dtype=$DT boot=$(cat /proc/sys/kernel/random/boot_id) $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_main --dtype $DT \
  --batches 16,64,128 --cases 1024:128,128:1024 --prefill-len 2048 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_main.json > $B/${TAG}_main.log 2>&1
echo "MAIN_RC $? $(date)"
python -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_qserve --dtype $DT \
  --batches 16,64,128,256 --cases 1024:512 --prefill-len 0 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_qserve.json > $B/${TAG}_qserve.log 2>&1
echo "QSERVE_RC $? $(date)"
