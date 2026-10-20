#!/bin/bash
# Round-9 Lever-2 occupancy probe: does lowering smem (fewer stages) reach 2 CTA/SM
# and raise ACTIVE intensity? Capture occupancy + registers + max-active-blocks limiter
# + both TC rollups + wait, for representative fp8-heavy cells across st in {2,3,4}.
# DETACHED under luofan-ncu.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
PY=$K/prof_tc.py
OUT=$K/logs/occ
mkdir -p $OUT
python $PY build 0 0 0 0 0 || { echo "BUILD_FAIL"; exit 1; }
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
gpu__time_duration.sum"
runp() { local label=$1; shift
  ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[occ] done $label rc=$?"; }
# down M8192 r0.5 (fp8-heavy, non-persist sw0, kp0 sfa0): st in {2,3,4,5} spans smem.
for ST in 2 3 4 5; do
  runp down_M8192_r05_st${ST} dual 8192 2048 4096 4096 0 $ST 0 2 1 0 0 0
done
# gate M8192 r0.5 (persist sw8, kp0 sfa0): st in {3,4} (persist dispatch supports 3,4).
for ST in 3 4; do
  runp gate_M8192_r05_st${ST} dual 8192 8192 1024 1024 0 $ST 8 2 1 1 0 0
done
echo "[occ] OCC_DONE"
