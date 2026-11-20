#!/bin/bash
# rd-metricsweep: ProMix accuracy ratio curve (块1 新 spec).
#   For each r in the 11-point uniform grid, quantize ONCE and eval
#   wiki+C4+0-shot+MMLU via rd_mmlu.py (--budget_frac r --c4 --mmlu --zeroshot).
#   Idempotent (skip complete points). Bad ratio (error/OOM/not-block-aligned)
#   -> SKIP + record to _skipped.md + continue (never block the whole curve).
#   Each landed point auto-commits.
# Usage: metricsweep.sh <tag> <config.yaml> <gpu> [kv16|kv4]
#   e.g. metricsweep.sh llama3.2-3b promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml 1 kv16
set -u
REPO=/home/luofan/plaquant-metricsweep
PY=/home/luofan/baselines/venv/bin/python   # has lm_eval 0.4.13 + TF5.x (evaluator.py tuple-fix on branch)
GRID="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
TAG=$1; CFG=$2; GPU=$3; KV=${4:-kv16}
OUTD=$REPO/docs/rd_metricsweep_data
SKIP=$OUTD/_skipped.md
mkdir -p $OUTD
[ -f "$SKIP" ] || echo "| tag | KV | r | metric | reason |" > "$SKIP"
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache PYTHONPATH=$REPO
# KV4 wiring TBD; for now only kv16 (KV cache bf16). kv4 branch reserved.
KVFLAG=""
[ "$KV" = "kv4" ] && { echo "kv4 not wired yet — aborting"; exit 2; }

complete() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if ('c4' in d and 'mmlu_5shot' in d and 'zeroshot' in d) else 1)" "$1" 2>/dev/null; }

for r in $GRID; do
  PT=${TAG}_${KV}_r${r}
  OUT=$OUTD/${PT}.json
  LOG=$OUTD/log_${PT}.log
  if [ -f "$OUT" ] && complete "$OUT"; then echo "[$TAG $KV] r=$r complete, skip"; continue; fi
  echo "[$TAG $KV] START r=$r $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PY promix/scripts/rd_mmlu.py --config $CFG --point $PT \
    --substrate ring --ring_size 512 --alloc measured --budget_frac $r \
    --c4 --mmlu --zeroshot $KVFLAG --out $OUT > $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q RD_MMLU_DONE "$LOG"; then
    reason=$(grep -oE "AssertionError[^$]*|RuntimeError[^$]*|CUDA out of memory|not divisible[^$]*" "$LOG" | tail -1 | head -c 140)
    echo "| $TAG | $KV | $r | all | rc=$rc ${reason:-see $LOG} |" >> "$SKIP"
    echo "[$TAG $KV] SKIP r=$r rc=$rc — ${reason:-unknown, see log}"
    rm -f "$OUT"
    continue
  fi
  git add "$OUT" "$SKIP" 2>/dev/null
  git commit -q -m "metricsweep: $PT (wiki+c4+mmlu+0shot)" 2>/dev/null
  echo "[$TAG $KV] DONE r=$r $(date)"
done
git add "$SKIP" 2>/dev/null; git commit -q -m "metricsweep: $TAG $KV skip-log" 2>/dev/null
echo "[$TAG $KV] GRID COMPLETE $(date)"
