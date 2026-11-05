#!/bin/bash
# rd-densesweep EXTENDED protocol: ShareGPT real-distribution serving point for
# ONE method (vLLM openai server + `vllm bench serve --dataset-name sharegpt`,
# the real conversational ISL/OSL mix — the "realistic mixed load" number;
# fixed-length cells elsewhere are the synthetic standard).
# Usage: sgpt_point.sh <size 3b|8b> <method>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
SIZE=$1; M=$2
source $HERE/methods.sh
method_map $SIZE $M || { echo "[sgpt_point] no mapping $SIZE $M"; exit 2; }
case $MODEL in /*) [ -e "$MODEL" ] || {
  echo "[sgpt_point] SKIP $SIZE $M — checkpoint missing: $MODEL"; exit 0; } ;; esac
[ "$M" = "sharq" ] && { echo "[sgpt_point] SKIP sharq — needs out-of-tree quant-config registration inside vllm serve (wrapper is bench_serve-only)"; exit 0; }
TAG=sgpt_${M}${SIZE}
DATA=$REPO/docs/vllm_final_data
PORT=$((8300 + ${CUDA_VISIBLE_DEVICES:-0}))
VLLM=/home/luofan/baselines/venv/bin/vllm
SGPT=$WORK/ShareGPT_V3_unfiltered_cleaned_split.json
cd $REPO

echo "[sgpt_point] $TAG gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"
[ -f $DATA/${TAG}.json ] && { git_commit_point "rd-densesweep: sharegpt $TAG"; exit 0; }

if [ ! -f $SGPT ]; then
  flock $SGPT.lock -c "wget -q -O $SGPT.part \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
    && mv $SGPT.part $SGPT" || { echo "[sgpt_point] sharegpt download failed"; exit 1; }
fi

env PYTHONPATH=$REPO VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HOME=/home/luofan/hf_cache \
  $VLLM serve $MODEL --dtype $DTYPE ${QFLAG:+--quantization $QFLAG} \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 512 \
  --port $PORT > $BENCH/${TAG}_server.log 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null; wait $SRV 2>/dev/null" EXIT
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "[sgpt_point] server died (see ${TAG}_server.log)"; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "[sgpt_point] server never healthy"; exit 1; }

env PYTHONPATH=$REPO $VLLM bench serve --backend vllm \
  --base-url http://127.0.0.1:$PORT --model $MODEL \
  --dataset-name sharegpt --dataset-path $SGPT --num-prompts 512 \
  --save-result --result-dir $BENCH --result-filename ${TAG}.json \
  > $BENCH/${TAG}.log 2>&1
RC=$?
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; trap - EXIT
[ $RC -eq 0 ] && [ -f $BENCH/${TAG}.json ] || { echo "[sgpt_point] bench failed"; exit 1; }
cp -f $BENCH/${TAG}.json $DATA/
git_commit_point "rd-densesweep: sharegpt real-distribution point $TAG"
echo "[sgpt_point] DONE $TAG $(date)"
