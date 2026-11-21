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
TAG=$1; CFG=$2; GPU=$3; KV=${4:-kv16}; MODE=${5:-full}; LIM=${6:-}; BS=${7:-}
OUTD=$REPO/docs/rd_metricsweep_data
SKIP=$OUTD/_skipped.md
mkdir -p $OUTD
[ -f "$SKIP" ] || echo "| tag | KV | r | metric | reason |" > "$SKIP"
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache PYTHONPATH=$REPO
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + GPU))   # unique per-GPU port — parallel sweeps must not share torch.distributed 29500
# KV: kv16 = KV cache bf16 (default); kv4 = KV cache 4-bit (use the -kv4.yaml config, labels outputs _kv4)
# MODE: ppl = wiki+C4 only (fast core curve); full = +MMLU+0shot (slow downstream)
if [ "$MODE" = "ppl" ]; then EVALFLAGS="--c4"; REQ="('c4' in d)"; else EVALFLAGS="--c4 --mmlu --zeroshot${LIM:+ --limit $LIM}${BS:+ --batch_size $BS}"; REQ="('c4' in d and 'mmlu_5shot' in d and 'zeroshot' in d)"; fi

complete() { $PY -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if $REQ else 1)" "$1" 2>/dev/null; }

for r in $GRID; do
  PT=${TAG}_${KV}_r${r}
  OUT=$OUTD/${PT}.json
  LOG=$OUTD/log_${PT}.log
  if [ -f "$OUT" ] && complete "$OUT"; then echo "[$TAG $KV $MODE] r=$r complete, skip"; continue; fi
  echo "[$TAG $KV $MODE] START r=$r $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PY promix/scripts/rd_mmlu.py --config $CFG --point $PT \
    --substrate ring --ring_size 512 --alloc measured --budget_frac $r \
    $EVALFLAGS --out $OUT > $LOG 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q RD_MMLU_DONE "$LOG"; then
    reason=$(grep -oE "AssertionError[^$]*|RuntimeError[^$]*|CUDA out of memory|not divisible[^$]*" "$LOG" | tail -1 | head -c 140)
    echo "| $TAG | $KV | $r | $MODE | rc=$rc ${reason:-see $LOG} |" >> "$SKIP"
    echo "[$TAG $KV $MODE] SKIP r=$r rc=$rc — ${reason:-unknown, see log}"
    continue
  fi
  git add "$OUT" "$SKIP" 2>/dev/null
  git commit -q -m "metricsweep: $PT $MODE (wiki+c4$([ "$MODE" = full ] && echo +mmlu+0shot))" 2>/dev/null
  echo "[$TAG $KV $MODE] DONE r=$r $(date)"
done
git add "$SKIP" 2>/dev/null; git commit -q -m "metricsweep: $TAG $KV $MODE skip-log" 2>/dev/null
echo "[$TAG $KV $MODE] GRID COMPLETE $(date)"
