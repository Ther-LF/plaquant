#!/bin/bash
# Round-20 ncu ROOT-CAUSE at MID-r: confirm/refute the serial dual-format co-design
# GEMM's "phase-seam + double-fill" hypothesis at gate.8192 r=1/2 and down.8192 r=1/2.
# For each cell: profile the DUAL fused GEMM mainloop vs the pure-NVFP4 / pure-MXFP8
# endpoints (at the SAME per-phase K). If the dual TC-elapsed is much lower than the
# endpoints AND barrier-stall is high -> seam is real (overlap helps). If dual TC is
# already high / barrier-stall low -> no seam; the mid-r bulge lives in pack, not GEMM.
# DETACHED. CSV harvested from $OUT/*.csv on disk.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CVD:-1}
K=kernels/mixed_gemm_sm120
PY=$K/prof_cdmid.py
OUT=$K/logs/cdmid
mkdir -p $OUT
python $PY build > $OUT/_build.log 2>&1 || { echo "BUILD_FAIL"; cat $OUT/_build.log; exit 1; }
echo "BUILD_OK"
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
sm__inst_executed_pipe_tensor.sum,\
launch__registers_per_thread,\
launch__occupancy_limit_shared_mem,\
gpu__time_duration.sum"
runo() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:device_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[cdmid] done $label rc=$?"; }
# down.8192 r=1/2 : M N Kl Kh  (N=2048, K=8192 -> Kl=Kh=4096)
runo DUAL_down   cddual 8192 2048 4096 4096
runo FP4_down    fp4    8192 2048 4096
runo FP8_down    fp8    8192 2048 4096
# gate.8192 r=1/2 : N=8192, K=2048 -> Kl=Kh=1024
runo DUAL_gate   cddual 8192 8192 1024 1024
runo FP4_gate    fp4    8192 8192 1024
runo FP8_gate    fp8    8192 8192 1024
echo "[cdmid] CDMID_NCU_DONE"
