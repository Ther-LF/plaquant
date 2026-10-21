#!/bin/bash
# Round-16 trace v2: capture a LARGE kernel window (no blind skip) and filter to the
# co-design's own kernels (cd_fill_sf_kernel scatter + cutlass device_kernel GEMM),
# grouping the steady fused_gemm calls to split per-call DEVICE time into
# (2x activation-SF scatter) vs (fused GEMM). Also traces the shipping kernel to
# confirm it launches NO per-call scatter. Base-clock. DETACHED under ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
OUT=$K/logs/traceR16b
mkdir -p $OUT
MET="gpu__time_duration.sum"
cdtrace() { local label=$1; shift
  ncu --target-processes all -c 130 --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $K/prof_r15dual.py cd "$@" > "$OUT/$label.run" 2>&1
  echo "[tr2] done $label rc=$?"; }
shtrace() { local label=$1; shift
  ncu --target-processes all -c 60 --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $K/prof_tc.py dual "$@" > "$OUT/$label.run" 2>&1
  echo "[tr2] done $label rc=$?"; }
for M in 8192 2048; do
  cdtrace CDtr_qkv_r8_M${M}   $M 2048 1792 256
  cdtrace CDtr_down_r8_M${M}  $M 2048 7168 1024
  cdtrace CDtr_gate_r8_M${M}  $M 8192 1792 256
done
shtrace SHtr_qkv_r8_M8192   8192 2048 1792 256  0 3 0 2 1 0 1 1 0 0 0
shtrace SHtr_down_r8_M8192  8192 2048 7168 1024 0 3 0 2 1 0 1 1 0 0 0
shtrace SHtr_gate_r8_M8192  8192 8192 1792 256  0 3 8 2 1 1 1 1 0 0 0
echo "[tr2] TRACE_R16B_DONE"
