#!/bin/bash
# Run a sequence of rd_accmatrix stages on ONE GPU lane, logging progress.
#   ... -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
#       bash promix/scripts/rd_accmatrix_lane.sh laneA zs_arc_3b zs_sharq_3b ...
LANE=$1; shift
OUT=/home/luofan/accmatrix
LOG=$OUT/lane_${LANE}.log
mkdir -p $OUT
echo "LANE $LANE START $(date) gpu=$CUDA_VISIBLE_DEVICES stages: $*" >> $LOG
for st in "$@"; do
  echo "=== STAGE $st BEGIN $(date) ===" >> $LOG
  bash /home/luofan/plaquant-accmatrix/promix/scripts/rd_accmatrix_run.sh $st >> $OUT/run_${st}.log 2>&1
  rc=$?
  echo "=== STAGE $st END $(date) rc=$rc ===" >> $LOG
done
echo "LANE $LANE ALL_DONE $(date)" >> $LOG
