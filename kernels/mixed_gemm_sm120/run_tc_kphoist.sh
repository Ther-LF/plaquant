#!/bin/bash
# Round-11 ALU-density probe: does the hoisted register-pipeline consumer (KPIPE=1,
# faithful CUTLASS mma() port — partition/fragment setup hoisted OUT of the k-loop,
# per-k_block copy+gemm interleave) cut the ALU instruction count vs the non-hoisted
# consume4/consume8 (KPIPE=0, which rebuilds the SF tensor + re-partitions every chunk)?
# Clean A/B at the SAME shape x r, kp in {0,1}, pure phases (r=0 fp4, r=1 fp8) to
# isolate. alu/tc is the headline. sfa=0 to isolate. DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/tcKPH
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed.sum,\
sm__inst_executed_pipe_alu.sum,\
sm__inst_executed_pipe_lsu.sum,\
sm__inst_executed_pipe_fma.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
gpu__time_duration.sum"
runp() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[kph] done $label rc=$?"; }
# dual: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong ldbw
M=8192
for KP in 0 1; do
  runp qkv_r0_kp${KP}  dual $M 2048 2048 0    0 3 0 2 1 0 $KP 0 0 0
  runp qkv_r1_kp${KP}  dual $M 2048 0    2048 0 3 0 2 1 0 $KP 0 0 0
  runp down_r0_kp${KP} dual $M 2048 8192 0    0 3 0 2 1 0 $KP 0 0 0
  runp down_r1_kp${KP} dual $M 2048 0    8192 0 3 0 2 1 0 $KP 0 0 0
  runp gate_r0_kp${KP} dual $M 8192 2048 0    0 3 8 2 1 1 $KP 0 0 0
  runp gate_r1_kp${KP} dual $M 8192 0    2048 0 3 8 2 1 1 $KP 0 0 0
done
echo "[kph] TC_KPH_DONE"
