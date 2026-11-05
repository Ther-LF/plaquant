#!/bin/bash
# rd-densesweep: ONE accuracy-curve point — fake-quant wiki+C4 PPL at a given
# FP8 ring-budget fraction (ring512+measured, acctune shipping recipe).
# 1b/3b/8b -> rd_acctune.py (GPU-resident); 70b -> rd_scale.py (layer-offload).
# Usage: acc_point.sh <size 1b|3b|8b|70b> <r e.g. 0.05>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
SIZE=$1; R=$2
size_knobs $SIZE || exit 2
TAG=${SIZE}_r${R}
OUTD=$REPO/docs/rd_acctune_data/densesweep
OUT=$OUTD/${TAG}.json
mkdir -p $OUTD
cd $REPO

echo "[acc_point] $TAG gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"
if [ ! -f $OUT ]; then
  if [ "$SIZE" = "70b" ]; then
    DRIVER=promix/scripts/rd_scale.py; DONEMARK=RDSCALE_DONE
  else
    DRIVER=promix/scripts/rd_acctune.py; DONEMARK=ACCTUNE_DONE
  fi
  env PYTHONPATH=$REPO ${WMSE:+PROMIX_WEIGHT_MSE_STEPS=$WMSE} \
      ${LOADDEV:+PROMIX_LOAD_DEVICE=$LOADDEV} \
    python $DRIVER --config $CFG_ACC --point $TAG \
      --substrate ring --ring_size 512 --alloc measured --budget_frac $R \
      --gptq_nsamples $GPTQ_N ${PD:+--percdamp $PD} --c4 \
      --stats_cache $WORK/stats_${SIZE}_ring512.pt \
      --out $OUT > $WORK/logs/acc_${TAG}.log 2>&1
  grep -q $DONEMARK $WORK/logs/acc_${TAG}.log || {
    echo "[acc_point] FAILED $TAG (see logs/acc_${TAG}.log)"; rm -f $OUT; exit 1; }
fi
git_commit_point "rd-densesweep: accuracy point $TAG (wiki+C4)"
echo "[acc_point] DONE $TAG $(date)"
