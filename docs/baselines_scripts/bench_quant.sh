#!/bin/bash
# bench_quant.sh <gpu> <tag> <model_dir> [quant_method] -- waits for free mem, benches, logs
cd /home/luofan/baselines
source /home/luofan/baselines/venv/bin/activate
export HF_HOME=/home/luofan/hf_cache
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MML=${MML:-12288}
G=$1
/home/luofan/baselines/scripts/waiter.sh "$G" ${NEED:-40000} 30 20 || exit 1
QARG=""
if [ -n "${4:-}" ]; then QARG="--quantization $4"; fi
CUDA_VISIBLE_DEVICES=$G python scripts/bench_vllm.py --model "$3" --tag "$2" \
  --out /home/luofan/baselines/results/$2.json $QARG --max-model-len $MML \
  --gpu-mem-util ${GMU:-0.6} --prefill-lens 2048,4096,8192 --prefill-batches 1,8 --decode-lens 1,8 \
  > /home/luofan/baselines/logs/$2.log 2>&1
echo "==== EXIT $? $(date)" >> /home/luofan/baselines/logs/$2.log
