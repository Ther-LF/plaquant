#!/bin/bash
# Round-10 PINGPONG before/after: pp in {0 (current single-wide 8-warp TiledMMA),
# 1 (CUTLASS-faithful phase-aligned cooperative M-split), 2 (free-asymmetric skew)}.
# Coordinator success criteria: wait stall DOWN + ACTIVE intensity UP + ELAPSED->ACTIVE.
# Also capture regs/thr + occupancy limiters + warp-active (does the split change the
# occupancy story?) + long/short scoreboard (don't trade one stall for another).
# 3 shapes x M{2048,8192} x r{0,mid,high} x pp{0,1,2}, sfa fixed per-r (r=0->0 else 1).
# DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcPP
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
gpu__time_duration.sum"
runp() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[tcPP] done $label rc=$?"; }
# dual: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong
# All pp measured at sfa=0 to isolate the pingpong variable and keep every point
# bit-exact (pp2 skew + sfa1 NaNs at r=0.125; pp2 is exploratory-only). pp1 also
# composes bit-exactly with sfa1 for the shippable dispatch (gated separately).
# qkv N2048 Kt2048 t0 st3 sw0 p0 ; down N2048 Kt8192 t0 st3 sw0 p0 ; gate N8192 Kt2048 t0 st3 sw8 p1
for M in 8192 2048; do
  for PP in 0 1; do
    for KH in 0 256 1024; do KL=$((2048-KH)); runp qkv_M${M}_Kh${KH}_pp${PP} dual $M 2048 $KL $KH 0 3 0 2 1 0 0 0 $PP; done
    for KH in 0 1024 4096; do KL=$((8192-KH)); runp down_M${M}_Kh${KH}_pp${PP} dual $M 2048 $KL $KH 0 3 0 2 1 0 0 0 $PP; done
    for KH in 0 256 1024; do KL=$((2048-KH)); runp gate_M${M}_Kh${KH}_pp${PP} dual $M 8192 $KL $KH 0 3 8 2 1 1 0 0 $PP; done
  done
done
echo "[tcPP] TC_PINGPONG_DONE"
