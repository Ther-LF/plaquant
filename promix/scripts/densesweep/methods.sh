#!/bin/bash
# rd-densesweep: method -> (MODEL, DTYPE, QFLAG, BENCH_PY) mapping for the
# extended serving protocol. Paths are the SAME checkpoints the committed
# five-way campaign benched (taken from the committed JSON metas).
# BENCH_PY defaults to the repo bench_serve; sharq needs the peers wrapper
# (out-of-tree vllm quant-config registration).
method_map() {   # $1 = size, $2 = method
  local size=$1 m=$2
  BENCH_PY="-m promix.vllm_integ.bench_serve"
  QFLAG=""; DTYPE=bfloat16
  case ${m}_${size} in
    bf_3b)    MODEL=/home/luofan/models/Llama-3.2-3B ;;
    bf_8b)    MODEL=/home/luofan/models/llama-3-8b ;;
    fp8_3b)   MODEL=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-fp8 ;;
    fp8_8b)   MODEL=/home/luofan/baselines/quant/unsloth--llama-3-8b-fp8 ;;
    gptq_3b)  MODEL=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-gptq_w4a16 ;;
    gptq_8b)  MODEL=/home/luofan/baselines/quant/unsloth--llama-3-8b-gptq_w4a16 ;;
    awq_3b)   MODEL=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-awq_w4a16 ;;
    awq_8b)   MODEL=/home/luofan/baselines/quant/unsloth--llama-3-8b-awq_w4a16 ;;
    nv_3b)    MODEL=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-nvfp4 ;;
    nv_8b)    MODEL=/home/luofan/baselines/quant/unsloth--llama-3-8b-nvfp4 ;;
    pm_3b)    MODEL=/home/luofan/vllm_integ/promix-llama-3.2-3b; DTYPE=float16 ;;
    pm_8b)    MODEL=/home/luofan/vllm_integ/promix-llama-3-8b;   DTYPE=float16 ;;
    # "a couple ratios" of the ProMix knob for the extended grid — the
    # persistent vllm-final sweep checkpoints (no re-export needed)
    pmr0.0625_3b) MODEL=/home/luofan/vllm_integ/promix-3b-r0.0625; DTYPE=float16 ;;
    pmr0.25_3b)   MODEL=/home/luofan/vllm_integ/promix-3b-r0.25;   DTYPE=float16 ;;
    pmr0.0625_8b) MODEL=/home/luofan/vllm_integ/promix-8b-r0.0625; DTYPE=float16 ;;
    pmr0.25_8b)   MODEL=/home/luofan/vllm_integ/promix-8b-r0.25;   DTYPE=float16 ;;
    sharq_3b) MODEL=/home/luofan/vllm_integ/bench_final/peers/sharq/ckpt-llama32-3b
              QFLAG=sharq
              BENCH_PY=/home/luofan/vllm_integ/bench_final/peers/sharq/bench_serve_sharq.py ;;
    fpquant_3b) MODEL=/home/luofan/vllm_integ/bench_final/peers/qutlass/ckpt-llama32-3b-nvfp4-rtn-vllm ;;
    *) return 1 ;;
  esac
  return 0
}
