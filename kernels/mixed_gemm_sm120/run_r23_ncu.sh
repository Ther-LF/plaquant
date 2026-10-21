#!/bin/bash
# Round-23 ncu: (A) Part-A TC-duty ELAPSED for co-design / shipping / pure endpoints;
# (B) Part-B register + occupancy for the serial shared-accum (base) vs two-accumulator
# Option-C (optc) co-design kernels -> does 2x accumulator blow the 64K-reg budget?
# DETACHED. CSV harvested from $OUT/*.csv on disk.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CVD:-4}
K=kernels/mixed_gemm_sm120
PY=$K/prof_optc.py
OUT=$K/logs/r23ncu
mkdir -p $OUT
python $PY build > $OUT/_build.log 2>&1 || { echo "BUILD_FAIL"; tail -20 $OUT/_build.log; exit 1; }
echo "BUILD_OK"
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__waves_per_multiprocessor,\
gpu__time_duration.sum"
runo() { local label=$1; local kregex=$2; shift 2
  ncu --target-processes all -c 1 -s 10 -k "regex:$kregex" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python $PY "$@" > "$OUT/$label.run" 2>&1
  echo "[r23ncu] done $label rc=$?"; }
# down.8192 r=1/8 : N=2048 K=8192 -> Kl=7168 Kh=1024
runo base_down  device_kernel base 8192 2048 7168 1024
runo optc_down  device_kernel optc 8192 2048 7168 1024
SH_TILE=0 SH_SW=0 SH_PERS=0 runo ship_down  b1_tma ship 8192 2048 7168 1024
runo fp4_down   device_kernel fp4  8192 2048 7168 1024
runo fp8_down   device_kernel fp8  8192 2048 7168 1024
# gate.8192 r=1/2 : N=8192 K=2048 -> Kl=Kh=1024 (round-20 bulge cell)
runo base_gate  device_kernel base 8192 8192 1024 1024
runo optc_gate  device_kernel optc 8192 8192 1024 1024
SH_TILE=0 SH_SW=8 SH_PERS=1 runo ship_gate  b1_tma ship 8192 8192 1024 1024
runo fp4_gate   device_kernel fp4  8192 8192 1024 1024
runo fp8_gate   device_kernel fp8  8192 8192 1024 1024
# qkv.8192 r=1/8 : N=2048 K=2048 -> Kl=1792 Kh=256
runo base_qkv   device_kernel base 8192 2048 1792 256
runo optc_qkv   device_kernel optc 8192 2048 1792 256
SH_TILE=0 SH_SW=0 SH_PERS=0 runo ship_qkv  b1_tma ship 8192 2048 1792 256
echo "[r23ncu] R23_NCU_DONE"
