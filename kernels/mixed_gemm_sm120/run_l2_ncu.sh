#!/bin/bash
# Round-7 L2/data-access ncu diagnostic (project-lead L2 line). Run DETACHED inside
# luofan-ncu on a free GPU. Measures L2/L1 hit rate, DRAM traffic + %peak, TC util for
# the gate class (N8192) across rasterization (swz) x persist x r, plus FP4(r=0)/FP8(r=1)
# pure-phase L2 footprint. Honest: judges how much of gate's residual is L2-reuse-limited.
#
# Launch:
#  docker exec -d -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant \
#    -e TORCH_CUDA_ARCH_LIST=12.0a -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#    -e CUDA_VISIBLE_DEVICES=2 luofan-ncu \
#    bash kernels/mixed_gemm_sm120/run_l2_ncu.sh > kernels/mixed_gemm_sm120/logs/l2_ncu.log 2>&1
set -u
cd /home/luofan/plaquant
D=kernels/mixed_gemm_sm120
OUT=$D/l2_ncu.csv
NCU=/usr/local/cuda/bin/ncu
MET="lts__t_sector_hit_rate.pct,l1tex__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum"

# Pre-build the extension outside ncu (first import compiles/caches the .so).
echo "=== prebuild ==="
python $D/prof_l2.py 8192 8192 1024 1024 3 0 0 0 || { echo PREBUILD_FAIL; exit 1; }

# config matrix: "M N Kl Kh st tile swz persist tag"
run() {
  local M=$1 N=$2 Kl=$3 Kh=$4 st=$5 tile=$6 swz=$7 pers=$8 tag=$9
  echo "##CFG $tag M=$M N=$N Kl=$Kl Kh=$Kh st=$st tile=$tile swz=$swz P=$pers"
  $NCU --target-processes all -k regex:"b1_tma_dual_kernel" -c 1 -s 2 --csv \
    --metrics "$MET" \
    python $D/prof_l2.py $M $N $Kl $Kh $st $tile $swz $pers 2>&1
}
{
# gate M8192 r=0.5 : rasterization x persist (the target)
run 8192 8192 1024 1024 3 0 0 0 gate_M8192_r50_sw0_P0
run 8192 8192 1024 1024 3 0 8 0 gate_M8192_r50_sw8_P0
run 8192 8192 1024 1024 3 0 8 1 gate_M8192_r50_sw8_P1
run 8192 8192 1024 1024 3 0 0 1 gate_M8192_r50_sw0_P1
# gate M8192 pure-phase L2 footprint: FP4 (r=0) vs FP8 (r=1, 2x bytes)
run 8192 8192 2048 0    3 0 0 0 gate_M8192_r0_fp4_sw0_P0
run 8192 8192 2048 0    3 0 8 0 gate_M8192_r0_fp4_sw8_P0
run 8192 8192 0    2048 3 0 0 0 gate_M8192_r1_fp8_sw0_P0
run 8192 8192 0    2048 3 0 8 0 gate_M8192_r1_fp8_sw8_P0
# down M8192 r=0.5 (long-K control: L2 reuse less critical)
run 8192 2048 4096 4096 3 0 0 0 down_M8192_r50_sw0_P0
run 8192 2048 4096 4096 3 0 8 0 down_M8192_r50_sw8_P0
} | tee $OUT
echo "=== L2_NCU_DONE ==="
