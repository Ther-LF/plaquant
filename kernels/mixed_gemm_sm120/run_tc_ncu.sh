#!/bin/bash
# Round-8 TC-duty baseline health-check (DETACHED under luofan-ncu).
# TC duty-cycle table: all shapes x M{2048,8192} x r{0,1/8,1/2,1} for our dual kernel
# (per-shape round-7 best cfg) + CUTLASS pure-fp4/fp8 TC% at the pure endpoints.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant
export TORCH_CUDA_ARCH_LIST=12.0a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tc
mkdir -p $OUT

echo "[tc] building exts..."
python $PY build 0 0 0 0 0 || { echo "[tc] BUILD_FAIL"; exit 1; }
echo "[tc] BUILD_OK"

MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"

runp() { # label kregex kind M N Kl Kh [cfg...]
  local label=$1 kr=$2; shift 2
  ncu --target-processes all -c 1 -s 5 -k "regex:$kr" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[tc] done $label rc=$?"
}

for M in 2048 8192; do
  # qkv N2048 Kt2048 : dual t0 st3 sw0 km2 mb1 p0
  for KH in 0 256 1024 2048; do KL=$((2048-KH)); runp qkv_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 2048 $KL $KH 0 3 0 2 1 0; done
  # down N2048 Kt8192 : dual t0 st3 sw0 km2 mb1 p0
  for KH in 0 1024 4096 8192; do KL=$((8192-KH)); runp down_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 2048 $KL $KH 0 3 0 2 1 0; done
  # gate N8192 Kt2048 : dual t0 st3 sw8 km2 mb1 p1 (round-7 best)
  for KH in 0 256 1024 2048; do KL=$((2048-KH)); runp gate_M${M}_Kh${KH} b1_tma_dual_kernel dual $M 8192 $KL $KH 0 3 8 2 1 1; done
done

# CUTLASS same-shape TC% at the pure endpoints (r=0 fp4, r=1 fp8)
for M in 2048 8192; do
  runp qkv_M${M}_cutfp4 device_kernel fp4 $M 2048 2048 0
  runp qkv_M${M}_cutfp8 device_kernel fp8 $M 2048 0 2048
  runp down_M${M}_cutfp4 device_kernel fp4 $M 2048 8192 0
  runp down_M${M}_cutfp8 device_kernel fp8 $M 2048 0 8192
  runp gate_M${M}_cutfp4 device_kernel fp4 $M 8192 2048 0
  runp gate_M${M}_cutfp8 device_kernel fp8 $M 8192 0 2048
done

echo "[tc] TC_BASELINE_DONE"
