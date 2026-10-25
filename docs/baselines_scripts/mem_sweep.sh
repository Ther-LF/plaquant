#!/bin/bash
# mem_sweep.sh <gpu> -- sequentially probe peak memory for every config, one GPU
G=$1
cd /home/luofan/baselines
source /home/luofan/baselines/venv/bin/activate
export HF_HOME=/home/luofan/hf_cache CUDA_VISIBLE_DEVICES=$G
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
OUT=/home/luofan/baselines/results/mem
mkdir -p $OUT
M1=$(ls -d /home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/)
M3=$(ls -d /home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-3B/snapshots/*/)
M8=$(ls -d /home/luofan/hf_cache/hub/models--unsloth--llama-3-8b/snapshots/*/)
run() { # tag model [quant] [mml]
  local tag=$1 model=$2 q=${3:-} mml=${4:-12288}
  local qarg=""
  [ -n "$q" ] && qarg="--quantization $q"
  /home/luofan/baselines/scripts/waiter.sh $G 45000 30 20 >> /home/luofan/baselines/logs/mem_sweep.log 2>&1 || return
  echo "=== $tag $(date +%H:%M:%S)" >> /home/luofan/baselines/logs/mem_sweep.log
  python scripts/mem_probe.py --model "$model" --tag "$tag" --out $OUT/$tag.json \
     --max-model-len $mml --gpu-mem-util 0.45 $qarg >> /home/luofan/baselines/logs/mem_sweep.log 2>&1
}
run bf16__llama-3.2-1b "$M1"
run gptq_w4a16__llama-3.2-1b /home/luofan/baselines/quant/Llama-3.2-1B-Instruct-GPTQ-W4A16 compressed-tensors
run awq_w4a16__llama-3.2-1b /home/luofan/baselines/quant/Llama-3.2-1B-Instruct-AWQ-W4A16 compressed-tensors
run nvfp4__llama-3.2-1b /home/luofan/baselines/quant/Llama-3.2-1B-Instruct-NVFP4
run fp8__llama-3.2-1b /home/luofan/baselines/quant/unsloth--Llama-3.2-1B-Instruct-fp8
run bf16__llama-3.2-3b "$M3"
run gptq_w4a16__llama-3.2-3b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-gptq_w4a16 compressed-tensors
run awq_w4a16__llama-3.2-3b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-awq_w4a16 compressed-tensors
run nvfp4__llama-3.2-3b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-nvfp4
run bf16__llama-3-8b "$M8" "" 8192
run gptq_w4a16__llama-3-8b /home/luofan/baselines/quant/unsloth--llama-3-8b-gptq_w4a16 compressed-tensors 8192
run awq_w4a16__llama-3-8b /home/luofan/baselines/quant/unsloth--llama-3-8b-awq_w4a16 compressed-tensors 8192
run nvfp4__llama-3-8b /home/luofan/baselines/quant/unsloth--llama-3-8b-nvfp4 "" 8192
echo "MEM SWEEP DONE $(date)" >> /home/luofan/baselines/logs/mem_sweep.log
