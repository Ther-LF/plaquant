set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120; PY=$K/prof_tc.py; OUT=$K/logs/bc; mkdir -p $OUT
MET="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
runp(){ local l=$1; shift; ncu --target-processes all -c 1 -s 5 -k "regex:b1_tma_dual_kernel" --metrics "$MET" --csv --log-file "$OUT/$l.csv" python $PY "$@" > "$OUT/$l.run" 2>&1; echo "[bc] $l rc=$?"; }
# Cells with largest lsb drop + a pure-fp4 + an fp8-heavy, sfa0 vs sfa1, kp per round-8.
for SFA in 0 1; do
  runp down_M8192_Kh1024_sfa${SFA} dual 8192 2048 7168 1024 0 3 0 2 1 0 1 $SFA
  runp qkv_M8192_Kh256_sfa${SFA}  dual 8192 2048 1792 256  0 3 0 2 1 0 1 $SFA
  runp down_M8192_Kh0_sfa${SFA}   dual 8192 2048 8192 0    0 3 0 2 1 0 0 $SFA
  runp gate_M8192_Kh1024_sfa${SFA} dual 8192 8192 1024 1024 0 3 8 2 1 1 1 $SFA
done
echo "[bc] DONE"
