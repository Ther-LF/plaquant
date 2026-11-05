#!/bin/bash
# rd-densesweep: ONE serving-curve point.
#   export (ring512 + measured, acctune recipe, --high_fraction r)
#   -> T2 gate (|dPPL| <= 0.1 vs own exporter reference)
#   -> serve-bench (committed sweep protocol, run_sweep_point.sh)
#   -> JSONs + export meta into docs/vllm_final_data/ -> rm checkpoint -> commit
# Usage: serve_point.sh <size 1b|3b|8b> <r e.g. 0.30>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source $HERE/common.sh
SIZE=$1; R=$2
size_knobs $SIZE || exit 2
[ -z "$CFG_SERVE" ] && { echo "no serve config for $SIZE"; exit 2; }
TAG=pm${SIZE}_r${R}
CKPT=$WORK/ckpt/promix-${SIZE}-r${R}
DATA=$REPO/docs/vllm_final_data
cd $REPO

echo "[serve_point] $TAG gpu=${CUDA_VISIBLE_DEVICES:-?} $(date)"
if [ ! -f $DATA/${TAG}_qserve.json ]; then
  # ---- export -------------------------------------------------------------
  if [ ! -f $CKPT/promix_export_meta.json ]; then
    env PYTHONPATH=$REPO ${WMSE:+PROMIX_WEIGHT_MSE_STEPS=$WMSE} \
      python -m promix.vllm_integ.export --config $CFG_SERVE --out $CKPT \
        --substrate ring --ring_size 512 --alloc measured --resolve-paths \
        --high_fraction $R --gptq_nsamples $GPTQ_N ${PD:+--percdamp $PD} \
        --no-perproj-ppl > $WORK/logs/export_${TAG}.log 2>&1
    grep -q EXPORT_DONE $WORK/logs/export_${TAG}.log || {
      echo "[serve_point] EXPORT FAILED $TAG (see logs/export_${TAG}.log)"
      exit 1; }
  fi
  # ---- BF16 lane anchor (once per gpu x size; decode-heavy cell only) ------
  A=bf16anchor_${SIZE}_gpu${CUDA_VISIBLE_DEVICES:-x}
  if [ ! -f $DATA/${A}.json ]; then
    case $SIZE in
      1b) BM=unsloth/Llama-3.2-1B-Instruct ;;
      3b) BM=/home/luofan/models/Llama-3.2-3B ;;
      8b) BM=/home/luofan/models/llama-3-8b ;;
    esac
    env PYTHONPATH=$REPO VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      $VENV_PY -m promix.vllm_integ.bench_serve --model $BM --tag $A \
      --dtype bfloat16 --batches 128,256 --cases 128:1024 --prefill-len 0 \
      --max-model-len 4096 --gpu-mem-util 0.90 \
      --extra-llm-kwargs "{\"max_num_seqs\":512}" \
      --out $BENCH/${A}.json > $BENCH/${A}.log 2>&1 \
      && cp $BENCH/${A}.json $DATA/
  fi
  # ---- gate + serve (committed protocol) -----------------------------------
  bash $REPO/docs/vllm_final_scripts/run_sweep_point.sh $CKPT $TAG $BENCH $REPO
  RC=$?
  cp -f $BENCH/gate_t2_${TAG}.json $DATA/ 2>/dev/null
  cp -f $CKPT/promix_export_meta.json $DATA/${TAG}_export_meta.json 2>/dev/null
  if [ $RC -eq 3 ]; then
    echo "[serve_point] T2 FAIL $TAG — point NOT served (gate json committed)"
    git_commit_point "rd-densesweep: $TAG T2 FAIL (|dPPL|>0.1) — gate json only"
    rm -rf $CKPT
    exit 1
  fi
  cp -f $BENCH/${TAG}_main.json $BENCH/${TAG}_qserve.json $DATA/
fi
rm -rf $CKPT
git_commit_point "rd-densesweep: serving point $TAG (gate+main+qserve+meta)"
echo "[serve_point] DONE $TAG $(date)"
