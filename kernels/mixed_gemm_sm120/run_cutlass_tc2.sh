set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120; PY=$K/prof_tc.py; OUT=$K/logs/tcC2; mkdir -p $OUT
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"
runp(){ local l=$1; shift; ncu --target-processes all -c 1 -s 0 -k "regex:device_kernel" --metrics "$MET" --csv --log-file "$OUT/$l.csv" python $PY "$@" > "$OUT/$l.run" 2>&1; echo "[tcC2] $l rc=$?"; }
for M in 2048 8192; do
  runp qkv_M${M}_fp4 fp4 $M 2048 2048 0;  runp qkv_M${M}_fp8 fp8 $M 2048 0 2048
  runp down_M${M}_fp4 fp4 $M 2048 8192 0; runp down_M${M}_fp8 fp8 $M 2048 0 8192
  runp gate_M${M}_fp4 fp4 $M 8192 2048 0; runp gate_M${M}_fp8 fp8 $M 8192 0 2048
done
echo "[tcC2] DONE"
