#!/bin/bash
# Round-11 instruction-mix diagnosis: per-pipe instruction counts to locate the
# non-TC instructions that occupy issue slots between MMAs (coordinator's "issue
# density" question), comparing OUR dual baseline (round-10 byte-equivalent, per-shape
# shipped dispatch: r=0 -> kp0 sfa1, r>0 -> kp1 sfa1) against the CUTLASS pure-NVFP4
# (fp4) and pure-MXFP8 (fp8) endpoints on the SAME shape. Shows tensor vs lsu vs alu
# vs fma pipe instruction counts + issue-active + TC both rollups + regs.
# DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcIMIX
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed.sum,\
sm__inst_executed_pipe_lsu.sum,\
sm__inst_executed_pipe_alu.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__inst_executed_pipe_xu.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
gpu__time_duration.sum"
runo() { local label=$1; local kre=$2; shift 2
  ncu --target-processes all -c 1 -s 5 -k "regex:$kre" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[imix] done $label rc=$?"; }
# OUR dual baseline (ldbw0). dual: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong ldbw
for M in 8192 2048; do
  # qkv (N2048 Kt2048) non-persist sw0
  runo qkv_M${M}_Kh0   b1_tma_dual_kernel dual $M 2048 2048 0    0 3 0 2 1 0 0 1 0 0
  runo qkv_M${M}_Kh256 b1_tma_dual_kernel dual $M 2048 1792 256  0 3 0 2 1 0 1 1 0 0
  # down (N2048 Kt8192)
  runo down_M${M}_Kh0    b1_tma_dual_kernel dual $M 2048 8192 0    0 3 0 2 1 0 0 1 0 0
  runo down_M${M}_Kh4096 b1_tma_dual_kernel dual $M 2048 4096 4096 0 3 0 2 1 0 1 1 0 0
  runo down_M${M}_Kh8192 b1_tma_dual_kernel dual $M 2048 0    8192 0 3 0 2 1 0 0 1 0 0
  # gate (N8192 Kt2048) persist sw8
  runo gate_M${M}_Kh0   b1_tma_dual_kernel dual $M 8192 2048 0    0 3 8 2 1 1 0 1 0 0
  runo gate_M${M}_Kh256 b1_tma_dual_kernel dual $M 8192 1792 256  0 3 8 2 1 1 1 1 0 0
  # CUTLASS endpoints (same shape): fp4 pure (Kt) and fp8 pure (Kt)
  runo CUTfp4_qkv_M${M}  device_kernel fp4 $M 2048 2048 0
  runo CUTfp8_qkv_M${M}  device_kernel fp8 $M 2048 0    2048
  runo CUTfp4_down_M${M} device_kernel fp4 $M 2048 8192 0
  runo CUTfp8_down_M${M} device_kernel fp8 $M 2048 0    8192
  runo CUTfp4_gate_M${M} device_kernel fp4 $M 8192 2048 0
  runo CUTfp8_gate_M${M} device_kernel fp8 $M 8192 0    2048
done
echo "[imix] TC_IMIX_DONE"
