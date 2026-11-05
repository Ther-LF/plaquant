#!/bin/bash
# rd-densesweep EXTENDED protocol: long-context + big-batch fixed-length grid
# for ONE method checkpoint (synthetic fixed-length = the standard serving
# measurement; latency depends on shape, not content).
#   grid: ISL {128,1024,2048,4096} x OSL {128,512,1024} x batch {16,64,128,256,512}
#   engine: max-model-len 5120 (covers 4096+1024), fp16 KV, max_num_seqs 512,
#           gpu-mem-util 0.90 — capacity-limited cells show up as preemption/
#           throughput collapse or per-case errors (recorded in the JSON).
# Usage: ext_point.sh <size 3b|8b> <method>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
SIZE=$1; M=$2
source $HERE/methods.sh            # sets MODEL DTYPE QFLAG BENCH_PY per (SIZE,M)
method_map $SIZE $M || { echo "[ext_point] no mapping $SIZE $M"; exit 2; }
case $MODEL in /*) [ -e "$MODEL" ] || {
  echo "[ext_point] SKIP $SIZE $M — checkpoint missing: $MODEL"; exit 0; } ;; esac
TAG=ext_${M}${SIZE}
DATA=$REPO/docs/vllm_final_data
cd $REPO

echo "[ext_point] $TAG gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"
if [ ! -f $DATA/${TAG}.json ]; then
  env PYTHONPATH=$REPO VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    $VENV_PY $BENCH_PY --model $MODEL --tag $TAG --dtype $DTYPE \
    ${QFLAG:+--quantization $QFLAG} \
    --batches 16,64,128,256,512 \
    --cases 128:128,128:512,128:1024,1024:128,1024:512,1024:1024,2048:128,2048:512,2048:1024,4096:128,4096:512,4096:1024 \
    --prefill-len 0 --max-model-len 5120 --gpu-mem-util 0.90 \
    --extra-llm-kwargs "{\"max_num_seqs\":512}" \
    --out $BENCH/${TAG}.json > $BENCH/${TAG}.log 2>&1
  [ -f $BENCH/${TAG}.json ] || { echo "[ext_point] FAILED $TAG"; exit 1; }
  cp -f $BENCH/${TAG}.json $DATA/
fi
git_commit_point "rd-densesweep: extended long-ctx grid $TAG"
echo "[ext_point] DONE $TAG $(date)"
