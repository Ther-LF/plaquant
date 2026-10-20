#!/bin/bash
# Round-12 SFILV imix: isolate the interleaved-SF (fp4) effect on the ALU pipe.
# For each shape x M x r, compare SFILV=0 vs SFILV=1 at the SAME producer (sfa=0)
# and the SAME kpipe, so only the consumer SF smem layout + copy atom differ.
# Shows alu/tc dropping toward CUTLASS ~1 and TC both rollups rising. DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcSFILV
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
# dual: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong ldbw sfilv
runo() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[sfilv] done $label rc=$?"; }
for M in 8192 2048; do
  # qkv N2048 Kt2048 sw0 nonpersist ; r=0 (pure fp4) and r=0.125
  for kp in 0 1; do
    runo qkv_M${M}_r0_kp${kp}_i0   dual $M 2048 2048 0   0 3 0 2 1 0 $kp 0 0 0 0
    runo qkv_M${M}_r0_kp${kp}_i1   dual $M 2048 2048 0   0 3 0 2 1 0 $kp 0 0 0 1
    runo down_M${M}_r0_kp${kp}_i0  dual $M 2048 8192 0   0 3 0 2 1 0 $kp 0 0 0 0
    runo down_M${M}_r0_kp${kp}_i1  dual $M 2048 8192 0   0 3 0 2 1 0 $kp 0 0 0 1
    runo gate_M${M}_r0_kp${kp}_i0  dual $M 8192 2048 0   0 3 8 2 1 1 $kp 0 0 0 0
    runo gate_M${M}_r0_kp${kp}_i1  dual $M 8192 2048 0   0 3 8 2 1 1 $kp 0 0 0 1
  done
  # mixed r=0.125 (fp4-heavy) at kp1 (shipped mixed dispatch)
  runo qkv_M${M}_r0125_kp1_i0  dual $M 2048 1792 256  0 3 0 2 1 0 1 0 0 0 0
  runo qkv_M${M}_r0125_kp1_i1  dual $M 2048 1792 256  0 3 0 2 1 0 1 0 0 0 1
  runo down_M${M}_r0125_kp1_i0 dual $M 2048 7168 1024 0 3 0 2 1 0 1 0 0 0 0
  runo down_M${M}_r0125_kp1_i1 dual $M 2048 7168 1024 0 3 0 2 1 0 1 0 0 0 1
  runo gate_M${M}_r0125_kp1_i0 dual $M 8192 1792 256  0 3 8 2 1 1 1 0 0 0 0
  runo gate_M${M}_r0125_kp1_i1 dual $M 8192 1792 256  0 3 8 2 1 1 1 0 0 0 1
done
echo "[sfilv] TC_SFILV_DONE"
