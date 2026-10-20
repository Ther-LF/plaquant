#!/bin/bash
# Round-13 STAGE-1 TC proof: per-pipe instruction mix (round-11 imix methodology)
# for the codesign NVFP4 kernel (stock SM120 block-scaled collective, our data
# contract) vs the CUTLASS pure-NVFP4 endpoint, on the three b1 shapes x
# M{2048,8192}. Proves whether the co-design reaches CUTLASS alu/tc ~ 1 and
# ACTIVE 68-86%. DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_codesign.py
OUT=$K/logs/tcCODESIGN
mkdir -p $OUT
python $PY build || { echo "BUILD_FAIL"; exit 1; }
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
runo() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:device_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[cd] done $label rc=$?"; }
for M in 8192 2048; do
  runo CD_qkv_M${M}   cd  $M 2048 2048
  runo CUT_qkv_M${M}  cut $M 2048 2048
  runo CD_down_M${M}  cd  $M 2048 8192
  runo CUT_down_M${M} cut $M 2048 8192
  runo CD_gate_M${M}  cd  $M 8192 2048
  runo CUT_gate_M${M} cut $M 8192 2048
done
echo "[cd] TC_CODESIGN_DONE"
