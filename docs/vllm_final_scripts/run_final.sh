#!/bin/bash
# FINAL five-way campaign: one config end-to-end on one GPU.
# Usage: run_final.sh <ckpt> <tag> <dtype> [gmu]
# Protocol (large-batch-centric): main = cases 1024:128 + 128:1024, b {1,16,32,64,128,256,512},
#   + prefill TTFT L2048 sweep; qserve = 1024:512 b {64,128,256} (continuous-batching, QServe ISL/OSL);
#   kv8 = long-ISL 2048/4096 with fp8 KV, b {16,64}.
set -x
B=/home/luofan/vllm_integ/bench_final
P=/home/luofan/baselines/venv/bin/python
CK=$1; TAG=$2; DT=$3; GMU=${4:-0.90}
export PYTHONPATH=/home/luofan/plaquant-vllm-integ PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HOME=/home/luofan/hf_cache
cd /home/luofan/plaquant-vllm-integ
$P -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_main --dtype $DT \
  --batches 1,16,32,64,128,256,512 --cases 1024:128,128:1024 --prefill-len 2048 \
  --max-model-len 4096 --gpu-mem-util $GMU \
  --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_main.json > $B/${TAG}_main.log 2>&1
$P -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_qserve --dtype $DT \
  --batches 64,128,256 --cases 1024:512 --prefill-len 0 \
  --max-model-len 4096 --gpu-mem-util $GMU \
  --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_qserve.json > $B/${TAG}_qserve.log 2>&1
$P -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_kv8 --dtype $DT \
  --batches 16,64 --cases 2048:128,4096:128 --prefill-len 0 \
  --max-model-len 8192 --gpu-mem-util $GMU \
  --extra-llm-kwargs "{\"kv_cache_dtype\":\"fp8\",\"max_num_seqs\":512}" \
  --out $B/${TAG}_kv8.json > $B/${TAG}_kv8.log 2>&1
touch $B/${TAG}.done
