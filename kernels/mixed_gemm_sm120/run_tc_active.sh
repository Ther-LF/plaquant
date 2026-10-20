#!/bin/bash
# Round-8 TC ACTIVE-rollup pass: both TC rollups (elapsed + active) + duration,
# all 36 points. Lean metric set -> fast. Merges with the elapsed/stall pass.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcA
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum"
runp() { local label=$1 kr=$2; shift 2
  ncu --target-processes all -c 1 -s 5 -k "regex:$kr" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[tcA] done $label rc=$?"; }
for M in 2048 8192; do
  for KH in 0 256 1024 2048; do KL=$((2048-KH)); runp qkv_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 2048 $KL $KH 0 3 0 2 1 0; done
  for KH in 0 1024 4096 8192; do KL=$((8192-KH)); runp down_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 2048 $KL $KH 0 3 0 2 1 0; done
  for KH in 0 256 1024 2048; do KL=$((2048-KH)); runp gate_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 8192 $KL $KH 0 3 8 2 1 1; done
done
for M in 2048 8192; do
  runp qkv_M${M}_cutfp4 device_kernel fp4 $M 2048 2048 0
  runp qkv_M${M}_cutfp8 device_kernel fp8 $M 2048 0 2048
  runp down_M${M}_cutfp4 device_kernel fp4 $M 2048 8192 0
  runp down_M${M}_cutfp8 device_kernel fp8 $M 2048 0 8192
  runp gate_M${M}_cutfp4 device_kernel fp4 $M 8192 2048 0
  runp gate_M${M}_cutfp8 device_kernel fp8 $M 8192 0 2048
done
echo "[tcA] TC_ACTIVE_DONE"
