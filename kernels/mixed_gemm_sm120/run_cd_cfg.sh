#!/bin/bash
cd /home/luofan/plaquant/kernels/mixed_gemm_sm120
export CUDA_VISIBLE_DEVICES=${1:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=12.0a
export MAX_JOBS=${2:-16}
echo CFG_START $(date) > logs/_cd_cfg_search.log
python -u cd_cfg_search.py >> logs/_cd_cfg_search.log 2>&1
echo "EXIT=$?" >> logs/_cd_cfg_search.log
echo CFG_END $(date) >> logs/_cd_cfg_search.log
