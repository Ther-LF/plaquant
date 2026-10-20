set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120; PY=$K/prof_tc.py; OUT=$K/logs/occ4; mkdir -p $OUT
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__registers_per_thread,launch__shared_mem_per_block_static,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,gpu__time_duration.sum"
runp(){ local l=$1; shift; ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv --log-file "$OUT/$l.csv" python $PY "$@" > "$OUT/$l.run" 2>&1; echo "[occ4] $l rc=$?"; }
# tile4 = 128x64 (WM2 WN2), down M8192 r0.5, st in {3,4,5}, nonpersist sw0 kp0 sfa0
for ST in 3 4 5; do runp down_t4_st${ST} dual 8192 2048 4096 4096 4 $ST 0 2 1 0 0 0; done
# tile3 = 64x128 (WM1 WN4), same
for ST in 3 4 5; do runp down_t3_st${ST} dual 8192 2048 4096 4096 3 $ST 0 2 1 0 0 0; done
echo "[occ4] DONE"
