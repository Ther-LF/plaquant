#!/bin/bash
# rd-densesweep: generate the campaign queue files (5% budget grid, 20 points
# r = 0.05 .. 1.00 per model).
#   queue.txt    — small-model lanes (1b/3b/8b, accuracy + serving)
#   queue70b.txt — dedicated 70B accuracy lane (layer-offload, slow)
# Serving points already committed by the vllm-final campaign that land ON a
# 5% target (r=0.25 for 3b/8b) are NOT re-run (reused per instruction; they
# are dense-uniform-substrate provenance — flagged in the harvest).
# r=1.00: the measured allocation is clamped to >=1 NVFP4 ring per Linear
# (kernel rejects K_low==0), so the realized point is "all-but-one-ring MXFP8";
# pure MXFP8 has no fused-kernel path (documented, not faked).
set -u
WORK=${DENSESWEEP_WORK:-/home/luofan/densesweep}
mkdir -p $WORK
GRID=$(seq 0.05 0.05 1.00 | LC_ALL=C xargs -n1 printf "%.2f\n")

Q=$WORK/queue.txt; : > $Q
for r in $GRID; do echo "acc 1b $r"   >> $Q; done
for r in $GRID; do echo "serve 1b $r" >> $Q; done
for r in $GRID; do [ "$r" = "0.25" ] || echo "serve 3b $r" >> $Q; done
for r in $GRID; do [ "$r" = "0.25" ] || echo "serve 8b $r" >> $Q; done
for r in $GRID; do echo "acc 3b $r"   >> $Q; done
for r in $GRID; do echo "acc 8b $r"   >> $Q; done

Q70=$WORK/queue70b.txt; : > $Q70
for r in $GRID; do echo "acc 70b $r" >> $Q70; done

# Extended serving protocol (user-approved additions): ShareGPT real-
# distribution point + long-context/big-batch fixed-length grid, per method,
# 3B + 8B (the five-way sizes). ProMix rides the shipped r=1/8 checkpoint +
# two persistent ratio checkpoints (r=1/16, r=1/4) — per-method comparison
# points, not a 20-pt re-sweep. sharq/fpquant are 3B-only peers (guarded:
# skipped with a note if their checkpoints/wrappers are absent).
QE=$WORK/queue_ext.txt; : > $QE
for s in 3b 8b; do
  for m in bf fp8 gptq awq nv pm pmr0.0625 pmr0.25; do echo "sgpt $s $m" >> $QE; done
done
echo "sgpt 3b fpquant" >> $QE
for s in 3b 8b; do
  for m in pm bf fp8 gptq awq nv pmr0.0625 pmr0.25; do echo "ext $s $m" >> $QE; done
done
echo "ext 3b sharq" >> $QE
echo "ext 3b fpquant" >> $QE

wc -l $Q $Q70 $QE
