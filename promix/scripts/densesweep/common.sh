#!/bin/bash
# rd-densesweep: shared per-size knobs (the acctune shipping recipes,
# docs/rd_accuracy_tuning.md) + campaign paths. Sourced by serve_point.sh /
# acc_point.sh. All sizes: ring512 substrate + global measured-deltaD ring
# allocation + act-MSE d3u2 (PROMIX_ACT_MSE_STEPS_UP=2). Per size:
#   1b  : GPTQ nsamples 1024                      (no weight-MSE depth)
#   3b  : GPTQ 512 + percdamp 0.025 + weight-MSE depth 7
#   8b  : GPTQ 512 + weight-MSE depth 7
#   70b : GPTQ 512 (tuned E2 recipe "up2+g512"), CPU-resident build (offload)
REPO=${DENSESWEEP_REPO:-/home/luofan/plaquant-densesweep}
WORK=${DENSESWEEP_WORK:-/home/luofan/densesweep}
BENCH=$WORK/bench
VENV_PY=/home/luofan/baselines/venv/bin/python
mkdir -p $WORK/logs $WORK/ckpt $BENCH

export HF_HOME=/home/luofan/hf_cache PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=12.0a
export PROMIX_ACT_MSE_STEPS_UP=2

size_knobs() {       # $1 = size -> sets CFG_SERVE CFG_ACC GPTQ_N PD WMSE LOADDEV
  PD=""; WMSE=""; LOADDEV=""
  case $1 in
    1b)  CFG_SERVE=promix/configs/llama-3.2-1b-fp-perhead.yaml
         CFG_ACC=promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml
         GPTQ_N=1024 ;;
    3b)  CFG_SERVE=promix/configs/llama-3.2-3b-fp-perhead.yaml
         CFG_ACC=promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml
         GPTQ_N=512; PD=0.025; WMSE=7 ;;
    8b)  CFG_SERVE=promix/configs/llama-3-8b-fp-perhead.yaml
         CFG_ACC=promix/configs/llama-3-8b-mxfp8-nvfp4.yaml
         GPTQ_N=512; WMSE=7 ;;
    70b) CFG_SERVE=""   # 70B serving: plugin has no TP + disk-blocked (E2)
         CFG_ACC=promix/configs/llama-3-70b-instruct-mxfp8-nvfp4.yaml
         GPTQ_N=512; LOADDEV=cpu ;;
    *) echo "unknown size $1" >&2; return 2 ;;
  esac
}

git_commit_point() { # $1 = message; stages everything under docs/
  (cd $REPO && flock $WORK/git.lock -c \
    "git add -A docs/vllm_final_data docs/rd_acctune_data && git commit -q -m \"$1\" || true")
}
