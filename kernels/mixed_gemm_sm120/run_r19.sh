#!/bin/bash
cd /home/luofan/plaquant/kernels/mixed_gemm_sm120
export CUDA_VISIBLE_DEVICES=${1:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=12.0a
echo R19_START $(date) > logs/_r19_lat.log
python -u r19_lat.py >> logs/_r19_lat.log 2>&1
echo "EXIT=$?" >> logs/_r19_lat.log
echo R19_END $(date) >> logs/_r19_lat.log
