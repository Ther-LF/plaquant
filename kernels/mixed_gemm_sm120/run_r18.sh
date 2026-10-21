#!/bin/bash
cd /home/luofan/plaquant/kernels/mixed_gemm_sm120
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=12.0a
echo R18_START $(date) > logs/_r18_lat.log
python -u r18_lat.py >> logs/_r18_lat.log 2>&1
echo "EXIT=$?" >> logs/_r18_lat.log
echo R18_END $(date) >> logs/_r18_lat.log
