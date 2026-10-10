#!/bin/bash
# Pareto sweep for adaptive high-fraction: run PPL for each energy_threshold.
# Usage (in container): bash promix/scripts/run_adaptive_sweep.sh
set -e
CFG=promix/configs/llama-3.2-1b-fp-adaptive.yaml
OUT=${OUT:-/tmp/adaptive_sweep.txt}
: > "$OUT"
for TAU in 0.50 0.60 0.70 0.80 0.85; do
  sed -i "s/energy_threshold: .*/energy_threshold: $TAU/" "$CFG"
  torchrun --nproc_per_node=1 --master_port=29531 promix/eval/ptq.py --config "$CFG" > "/tmp/adapt_$TAU.log" 2>&1
  echo "tau=$TAU  $(grep -i Perplexity /tmp/adapt_$TAU.log | tail -1)" >> "$OUT"
done
echo DONE >> "$OUT"; cat "$OUT"
