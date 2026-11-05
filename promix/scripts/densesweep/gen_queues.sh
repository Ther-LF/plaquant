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

wc -l $Q $Q70
