#!/bin/bash
# rd-densesweep: per-GPU lane worker. Drains a shared queue file (one job per
# line, format: "<serve|acc> <size> <r>") with flock-guarded pops so any number
# of lanes can share one queue. Runs each job via point.sh with this lane's
# pinned GPU. Failures are recorded in $WORK/failed.txt and the lane moves on.
# Usage (inside the luofan-plaquant container, detached per GPU):
#   CUDA_VISIBLE_DEVICES=5 bash densesweep_runner.sh <queue1> [queue2 ...]
# Multiple queues are drained IN ORDER (lane moves to queue2 when queue1 empty).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
GPU=${CUDA_VISIBLE_DEVICES:-x}
LOG=$WORK/logs/runner_gpu${GPU}.log
echo "[runner gpu$GPU] start $(date) queues=$*" >> $LOG

FAILS=0
for Q in "$@"; do
while true; do
  JOB=$(flock $Q.lock bash -c "head -n1 '$Q'; sed -i '1d' '$Q'")
  [ -z "$JOB" ] && break
  echo "[runner gpu$GPU] START $JOB $(date)" >> $LOG
  if bash $HERE/point.sh $JOB >> $LOG 2>&1; then
    echo "[runner gpu$GPU] OK    $JOB $(date)" >> $LOG
    FAILS=0
  else
    echo "[runner gpu$GPU] FAIL  $JOB $(date)" >> $LOG
    echo "gpu$GPU $JOB $(date)" >> $WORK/failed.txt
    FAILS=$((FAILS + 1))
    # systematic-failure brake: don't burn the shared queue on a broken lane
    if [ $FAILS -ge 3 ]; then
      echo "[runner gpu$GPU] 3 consecutive failures — lane ABORT $(date)" >> $LOG
      exit 1
    fi
    sleep 30
  fi
done
done
echo "[runner gpu$GPU] queue drained $(date)" >> $LOG
