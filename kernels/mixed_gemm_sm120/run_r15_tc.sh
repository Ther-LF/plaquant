#!/bin/bash
# Round-15 TC proof: ncu the DUAL-format shared-accumulator co-design fused kernel
# (round-11 imix metric set + BOTH TC rollups) on the 3 b1 shapes x M{2048,8192}
# x r{1/8,1/2}. Compared in the report to (a) the round-13 single-format PoC and
# (b) the CUTLASS endpoints (round-9/13). DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_r15dual.py
OUT=$K/logs/tcR15
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
  echo "[r15] done $label rc=$?"; }
# r=1/8 (production): qkv/gate Kl1792 Kh256 ; down Kl7168 Kh1024
# r=1/2            : qkv/gate Kl1024 Kh1024; down Kl4096 Kh4096
for M in 8192 2048; do
  runo CDd_qkv_r8_M${M}   cd  $M 2048 1792 256
  runo CDd_down_r8_M${M}  cd  $M 2048 7168 1024
  runo CDd_gate_r8_M${M}  cd  $M 8192 1792 256
  runo CDd_qkv_r2_M${M}   cd  $M 2048 1024 1024
  runo CDd_down_r2_M${M}  cd  $M 2048 4096 4096
  runo CDd_gate_r2_M${M}  cd  $M 8192 1024 1024
done
echo "[r15] TC_R15_DONE"
