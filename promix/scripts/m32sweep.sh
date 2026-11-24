#!/bin/bash
# m32sweep.sh — Motivation §3.2 Rotated-condition lane launcher.
# Usage: m32sweep.sh <gpu> <lane_id> <model:3b|8b> <rotation_seed> [ratios...]
#   default ratios: 0.1..0.9 (seed0 main matrix); seeds 1/2 run 0.1 0.5 0.9.
set -u
REPO=/home/luofan/plaquant-metricsweep
PY=/home/luofan/baselines/venv/bin/python
GPU=$1; LANE=$2; MODEL=$3; RSEED=$4; shift 4
RATIOS=${@:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"}
OUTD=$REPO/docs/motivation_3_2
DATA=$OUTD/data
FAIL=$OUTD/failures.csv
LOGD=$OUTD/logs
mkdir -p "$DATA" "$LOGD"
[ -f "$FAIL" ] || echo "run_id,rc,reason,retryable" > "$FAIL"
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache PYTHONPATH=$REPO
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29560 + LANE))

commit_locked() {
  (
    flock -w 300 9 || exit 1
    git add "$@" 2>/dev/null
    git commit -q -m "m32: $COMMIT_MSG" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null
  ) 9>/tmp/m31_git.lock
}

for r in $RATIOS; do
  RID=${MODEL}_rotated_rot${RSEED}_r${r}
  OUT=$DATA/${RID}.json
  LOG=$LOGD/${RID}.log
  if [ -f "$OUT" ] && $PY -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('status')=='ok' else 1)" "$OUT" 2>/dev/null; then
    echo "[lane$LANE m32 $MODEL rot$RSEED] r=$r complete, skip"; continue
  fi
  echo "[lane$LANE m32 $MODEL rot$RSEED] START r=$r $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $PY promix/scripts/rd_m32.py run32 \
    --model $MODEL --ratio $r --rotation_seed $RSEED > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q M32_DONE "$LOG"; then
    reason=$(grep -oE "AssertionError[^\"]*|RuntimeError[^\"]*|CUDA out of memory|ValueError[^\"]*" "$LOG" | tail -1 | head -c 160 | tr ',' ';')
    echo "$RID,$rc,${reason:-see $LOG},maybe" >> "$FAIL"
    COMMIT_MSG="$RID FAILED rc=$rc" commit_locked "$FAIL"
    echo "[lane$LANE m32] FAIL r=$r rc=$rc — ${reason:-see log}"
    continue
  fi
  COMMIT_MSG="$RID $(grep -o 'M32_RESULT.*' "$LOG" | head -c 120)" commit_locked "$OUT" "$FAIL"
  echo "[lane$LANE m32 $MODEL rot$RSEED] DONE r=$r $(date)"
done
echo "[lane$LANE m32 $MODEL rot$RSEED] LANE COMPLETE $(date)"
