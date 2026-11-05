#!/bin/bash
# rd-densesweep: per-GPU lane worker. Drains a shared queue file (one job per
# line, format: "<serve|acc> <size> <r>") with flock-guarded pops so any number
# of lanes can share one queue. Runs each job via point.sh with this lane's
# pinned GPU. Failures are recorded in $WORK/failed.txt and the lane moves on.
# Usage (inside the luofan-plaquant container, detached per GPU):
#   CUDA_VISIBLE_DEVICES=5 bash densesweep_runner.sh /home/luofan/densesweep/queue.txt
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
Q=$1
GPU=${CUDA_VISIBLE_DEVICES:-x}
LOG=$WORK/logs/runner_gpu${GPU}.log
echo "[runner gpu$GPU] start $(date) queue=$Q" >> $LOG

while true; do
  JOB=$(flock $Q.lock bash -c "head -n1 '$Q'; sed -i '1d' '$Q'")
  [ -z "$JOB" ] && break
  echo "[runner gpu$GPU] START $JOB $(date)" >> $LOG
  if bash $HERE/point.sh $JOB >> $LOG 2>&1; then
    echo "[runner gpu$GPU] OK    $JOB $(date)" >> $LOG
  else
    echo "[runner gpu$GPU] FAIL  $JOB $(date)" >> $LOG
    echo "gpu$GPU $JOB $(date)" >> $WORK/failed.txt
  fi
done
echo "[runner gpu$GPU] queue drained $(date)" >> $LOG
