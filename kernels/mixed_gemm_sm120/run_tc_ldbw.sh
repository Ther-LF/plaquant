#!/bin/bash
# Round-11 LDBW before/after: operand-B smem->reg ldmatrix width
# ldbw0 (SM75_U32x2_LDSM_N, 64-bit, round-10 default) vs ldbw1 (SM75_U32x4_LDSM_N,
# 128-bit, CUTLASS sm120_rr_smem_copy_selector_B). Success criterion (coordinator):
# raise MMA issue density -> ACTIVE TC up; the B half of the LSU (ldmatrix) pipe
# instruction count should DROP (half the B ldmatrix ops) without trading into a
# new stall. Capture instruction mix per pipe (tensor/lsu/alu/fma) + issue-active +
# TC both rollups + regs + wait/long/short scoreboard + duration.
# All measured at sfa=0 to isolate the LDBW variable (every point bit-exact).
# DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcLDBW
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed.sum,\
sm__inst_executed_pipe_lsu.sum,\
sm__inst_executed_pipe_alu.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
gpu__time_duration.sum"
runp() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[tcLDBW] done $label rc=$?"; }
# dual: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong ldbw
# qkv/down non-persist sw0 ; gate persist sw8. kpipe follows per-shape dispatch
# (r=0 -> kp0, r>0 -> kp1). All sfa=0 to isolate LDBW.
for M in 8192 2048; do
  for LW in 0 1; do
    # qkv (N2048 Kt2048)
    runp qkv_M${M}_Kh0_ld${LW}    dual $M 2048 2048 0    0 3 0 2 1 0 0 0 0 $LW
    runp qkv_M${M}_Kh256_ld${LW}  dual $M 2048 1792 256  0 3 0 2 1 0 1 0 0 $LW
    runp qkv_M${M}_Kh1024_ld${LW} dual $M 2048 1024 1024 0 3 0 2 1 0 1 0 0 $LW
    # down (N2048 Kt8192)
    runp down_M${M}_Kh0_ld${LW}    dual $M 2048 8192 0    0 3 0 2 1 0 0 0 0 $LW
    runp down_M${M}_Kh1024_ld${LW} dual $M 2048 7168 1024 0 3 0 2 1 0 1 0 0 $LW
    runp down_M${M}_Kh4096_ld${LW} dual $M 2048 4096 4096 0 3 0 2 1 0 1 0 0 $LW
    # gate (N8192 Kt2048) persist sw8
    runp gate_M${M}_Kh0_ld${LW}    dual $M 8192 2048 0    0 3 8 2 1 1 0 0 0 $LW
    runp gate_M${M}_Kh256_ld${LW}  dual $M 8192 1792 256  0 3 8 2 1 1 1 0 0 $LW
    runp gate_M${M}_Kh1024_ld${LW} dual $M 8192 1024 1024 0 3 8 2 1 1 1 0 0 $LW
  done
done
echo "[tcLDBW] TC_LDBW_DONE"
