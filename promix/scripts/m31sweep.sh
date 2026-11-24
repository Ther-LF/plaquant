#!/bin/bash
# m31sweep.sh — Motivation §3.1 lane launcher (see docs/motivation_3_1/).
#   One lane = one GPU slice + a list of strategy:scope[:seed] specs; for each
#   spec it walks the 11-point r grid IN ORDER. Idempotent (rd_m31.py run skips
#   complete points); failures append to failures.csv and never block the lane;
#   every landed point commits (flock-serialized across lanes).
# Usage: m31sweep.sh <gpu> <lane_id> <model:3b|8b> <spec> [<spec> ...]
#   spec = strategy:scope[:seed], e.g. ss:within  ma:across  rnd:within:1
#   e.g.  m31sweep.sh 4 0 3b ss:within ma:within
set -u
REPO=/home/luofan/plaquant-metricsweep
PY=/home/luofan/baselines/venv/bin/python
GPU=$1; LANE=$2; MODEL=$3; shift 3
GRID="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1"   # must match python %g formatting of run ids
OUTD=$REPO/docs/motivation_3_1
DATA=$OUTD/data
FAIL=$OUTD/failures.csv
LOGD=$OUTD/logs
mkdir -p "$DATA" "$LOGD"
[ -f "$FAIL" ] || echo "run_id,rc,reason,retryable" > "$FAIL"
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache PYTHONPATH=$REPO
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29560 + LANE))

commit_locked() {  # serialize git across lanes
  (
    flock -w 300 9 || exit 1
    git add "$@" 2>/dev/null
    git commit -q -m "m31: $COMMIT_MSG" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null
  ) 9>/tmp/m31_git.lock
}

for SPEC in "$@"; do
  STRAT=$(echo "$SPEC" | cut -d: -f1)
  SCOPE=$(echo "$SPEC" | cut -d: -f2)
  SEED=$(echo "$SPEC" | cut -d: -f3 -s); SEED=${SEED:-0}
  for r in $GRID; do
    RID=${MODEL}_${STRAT}_${SCOPE}_s${SEED}_r${r}
    OUT=$DATA/${RID}.json
    LOG=$LOGD/${RID}.log
    if [ -f "$OUT" ] && $PY -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if str(d.get('status','')).startswith(('ok','reused')) else 1)" "$OUT" 2>/dev/null; then
      echo "[lane$LANE $MODEL $SPEC] r=$r complete, skip"; continue
    fi
    echo "[lane$LANE $MODEL $SPEC] START r=$r $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $PY promix/scripts/rd_m31.py run \
      --model $MODEL --strategy $STRAT --scope $SCOPE --ratio $r \
      --assignment_seed $SEED > "$LOG" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || ! grep -q M31_DONE "$LOG"; then
      reason=$(grep -oE "AssertionError[^\"]*|RuntimeError[^\"]*|CUDA out of memory|ValueError[^\"]*" "$LOG" | tail -1 | head -c 160 | tr ',' ';')
      echo "$RID,$rc,${reason:-see $LOG},maybe" >> "$FAIL"
      COMMIT_MSG="$RID FAILED rc=$rc" commit_locked "$FAIL"
      echo "[lane$LANE] FAIL r=$r rc=$rc — ${reason:-see log}"
      continue
    fi
    COMMIT_MSG="$RID $(grep -o 'M31_RESULT.*' "$LOG" | head -c 120)" commit_locked "$OUT" "$FAIL"
    echo "[lane$LANE $MODEL $SPEC] DONE r=$r $(date)"
  done
done
echo "[lane$LANE $MODEL] LANE COMPLETE $(date)"
