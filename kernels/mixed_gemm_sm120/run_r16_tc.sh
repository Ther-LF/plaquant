#!/bin/bash
# Round-16 PARADOX DIAGNOSTIC: profile BOTH kernels in ONE ncu session (matched
# base-clock lock) at the production cells, with BOTH TC rollups + ABSOLUTE cycles
# (tensor-active + elapsed) + DRAM%/L2/L1 roofline + SM% + dur. This gives the
# ELAPSED-vs-ACTIVE table and cycle-accounting for co-design (CDd, device_kernel)
# vs the shipping hand-rolled kernel (SHIP, b1_tma_dual_kernel, round-12 production
# config: SFILV=0 + SFASYNC(sfa=1) + KPIPE, per-shape tile/sw/pers). DETACHED under
# luofan-ncu. Line-buffered; harvest from disk.
set -u
cd /home/luofan/plaquant
export PYTHONPATH=/home/luofan/plaquant TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=kernels/mixed_gemm_sm120
OUT=$K/logs/tcR16
mkdir -p $OUT
# prebuild both exts OUTSIDE ncu (fast if cached)
python $K/prof_r15dual.py build || { echo "CD_BUILD_FAIL"; exit 1; }
python $K/prof_tc.py build       || { echo "SHIP_BUILD_FAIL"; exit 1; }
echo "R16_BUILD_OK"
MET="sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_tensor_cycles_active.sum,\
sm__cycles_elapsed.sum,\
sm__cycles_active.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed_pipe_alu.sum,\
launch__registers_per_thread,\
gpu__time_duration.sum"
runo() { local label=$1; local kre=$2; shift 2
  ncu --target-processes all -c 1 -s 5 -k "regex:$kre" --metrics "$MET" --csv \
      --log-file "$OUT/$label.csv" python "$@" > "$OUT/$label.run" 2>&1
  echo "[r16] done $label rc=$?"; }

# per-shape shipping production config via prof_tc.py dual:
#   args: M N Kl Kh tile st sw km mb pers kpipe sfasync pingpong ldbw sfilv
#   qkv  : sw=0 pers=0 ; down: sw=0 pers=0 ; gate: sw=8 pers=1 ; kpipe=1 sfasync=1
for M in 8192 2048; do
  # ---- r = 1/8 (production) : qkv/gate Kl1792 Kh256 ; down Kl7168 Kh1024 ----
  runo CDd_qkv_r8_M${M}   device_kernel        $K/prof_r15dual.py cd  $M 2048 1792 256
  runo CDd_down_r8_M${M}  device_kernel        $K/prof_r15dual.py cd  $M 2048 7168 1024
  runo CDd_gate_r8_M${M}  device_kernel        $K/prof_r15dual.py cd  $M 8192 1792 256
  runo SHIP_qkv_r8_M${M}  b1_tma_dual_kernel   $K/prof_tc.py dual  $M 2048 1792 256  0 3 0 2 1 0 1 1 0 0 0
  runo SHIP_down_r8_M${M} b1_tma_dual_kernel   $K/prof_tc.py dual  $M 2048 7168 1024 0 3 0 2 1 0 1 1 0 0 0
  runo SHIP_gate_r8_M${M} b1_tma_dual_kernel   $K/prof_tc.py dual  $M 8192 1792 256  0 3 8 2 1 1 1 1 0 0 0
  # ---- r = 1/2 : qkv/gate Kl1024 Kh1024 ; down Kl4096 Kh4096 ----
  runo CDd_qkv_r2_M${M}   device_kernel        $K/prof_r15dual.py cd  $M 2048 1024 1024
  runo CDd_down_r2_M${M}  device_kernel        $K/prof_r15dual.py cd  $M 2048 4096 4096
  runo CDd_gate_r2_M${M}  device_kernel        $K/prof_r15dual.py cd  $M 8192 1024 1024
  runo SHIP_qkv_r2_M${M}  b1_tma_dual_kernel   $K/prof_tc.py dual  $M 2048 1024 1024 0 3 0 2 1 0 1 1 0 0 0
  runo SHIP_down_r2_M${M} b1_tma_dual_kernel   $K/prof_tc.py dual  $M 2048 4096 4096 0 3 0 2 1 0 1 1 0 0 0
  runo SHIP_gate_r2_M${M} b1_tma_dual_kernel   $K/prof_tc.py dual  $M 8192 1024 1024 0 3 8 2 1 1 1 1 0 0 0
done
echo "[r16] TC_R16_DONE"
