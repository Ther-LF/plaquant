#!/bin/bash
# Round-20 STAGE-1 PoC: build the cross-phase OVERLAP dual-format fused GEMM across
# shallow stage configs, gate cosine, and A/B its GEMM-only latency vs the SERIAL
# codesign kernel at gate.8192 r=1/2 and down.8192 r=1/2. DETACHED. Harvest r20_overlap.csv.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CVD:-3}
K=kernels/mixed_gemm_sm120
python $K/codesign_overlap_check.py
echo "EXIT=$?"
