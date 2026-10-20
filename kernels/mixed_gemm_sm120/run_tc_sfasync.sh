#!/bin/bash
# Round-9 Lever-1 SFASYNC before/after TC measurement: sfa in {0,1} at the round-8
# best kpipe per r (r=0 -> kp0, r>0 -> kp1), both TC rollups + long_scoreboard + wait
# + duration. 3 shapes x M{2048,8192} x r{0,1/8,1/2,1}. DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcS
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
runp() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[tcS] done $label rc=$?"; }
# kp(r): r=0 -> 0, else 1
for M in 2048 8192; do
  for SFA in 0 1; do
    # qkv N2048 Kt2048 t0 st3 sw0 p0
    for KH in 0 256 1024; do KL=$((2048-KH)); KP=$([ $KH -eq 0 ] && echo 0 || echo 1); runp qkv_M${M}_Kh${KH}_sfa${SFA} dual $M 2048 $KL $KH 0 3 0 2 1 0 $KP $SFA; done
    # down N2048 Kt8192 t0 st3 sw0 p0
    for KH in 0 1024 4096; do KL=$((8192-KH)); KP=$([ $KH -eq 0 ] && echo 0 || echo 1); runp down_M${M}_Kh${KH}_sfa${SFA} dual $M 2048 $KL $KH 0 3 0 2 1 0 $KP $SFA; done
    # gate N8192 Kt2048 t0 st3 sw8 p1
    for KH in 0 256 1024; do KL=$((2048-KH)); KP=$([ $KH -eq 0 ] && echo 0 || echo 1); runp gate_M${M}_Kh${KH}_sfa${SFA} dual $M 8192 $KL $KH 0 3 8 2 1 1 $KP $SFA; done
  done
done
echo "[tcS] TC_SFASYNC_DONE"
