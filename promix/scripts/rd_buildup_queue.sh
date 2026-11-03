#!/bin/bash
# rd-buildup E5a/E6 ACCURACY queue — GPU lane 5, runs INSIDE container
# luofan-plaquant, launched detached:
#   docker exec -d -w /home/luofan/plaquant-buildup -e CUDA_VISIBLE_DEVICES=5 \
#     luofan-plaquant bash promix/scripts/rd_buildup_queue.sh <phase>
# phases: e5a_3b | e5a_final | e6_seeds | e6_gate | all
set -x
cd /home/luofan/plaquant-buildup
export PYTHONPATH=/home/luofan/plaquant-buildup PYTHONUNBUFFERED=1
export HF_HOME=/home/luofan/hf_cache
export MASTER_PORT=${MASTER_PORT:-29647}
OUT=/home/luofan/buildup
mkdir -p $OUT

CFG1=promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml
CFG3=promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml
CFG8=promix/configs/llama-3-8b-mxfp8-nvfp4.yaml

# env classes (process-level; rd_buildup verifies these at start)
OFF="PROMIX_ACT_MSE=0 PROMIX_WEIGHT_MSE=0 PROMIX_WEIGHT_2LEVEL=0"
W_3B="PROMIX_ACT_MSE=0 PROMIX_WEIGHT_MSE=1 PROMIX_WEIGHT_2LEVEL=1 PROMIX_WEIGHT_MSE_STEPS=7"
FULL_3B="PROMIX_ACT_MSE=1 PROMIX_ACT_MSE_STEPS=3 PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE=1 PROMIX_WEIGHT_2LEVEL=1 PROMIX_WEIGHT_MSE_STEPS=7"
FULL_1B="PROMIX_ACT_MSE=1 PROMIX_ACT_MSE_STEPS=3 PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE=1 PROMIX_WEIGHT_2LEVEL=1 PROMIX_WEIGHT_MSE_STEPS=3"
FULL_8B="$FULL_3B"

row() {  # row <cfg> <tag> <row> <envstr> [extra args...]
  local cfg=$1 tag=$2 r=$3 envstr=$4; shift 4
  if [ -f $OUT/$tag.json ]; then echo "SKIP $tag (exists)"; return; fi
  env $envstr python promix/scripts/rd_buildup.py --config $cfg \
    --row $r --point $tag --c4 --out $OUT/$tag.json "$@" \
    > $OUT/$tag.log 2>&1
}

e5a_3b() {
  row $CFG3 3b_r1_norot     norot     "$OFF"
  row $CFG3 3b_r2_pca_u     pca_u     "$OFF"
  row $CFG3 3b_r3_ring512   ring512   "$OFF"
  row $CFG3 3b_r4_uniform18 uniform18 "$OFF"
  row $CFG3 3b_r5_gptq      gptq      "$OFF"
  row $CFG3 3b_r6_wscale    wscale    "$W_3B"
  row $CFG3 3b_r7_actmse    actmse    "$FULL_3B"
  # row 8 (+ row 9 decode-W4A16 in the same build)
  row $CFG3 3b_r8_measured  measured  "$FULL_3B" --also_w4a16
}

e5a_final() {
  # replicate the final row (full recipe) on 1B/8B (+ their w4a16 delta)
  row $CFG1 1b_r8_measured measured "$FULL_1B" --also_w4a16
  row $CFG8 8b_r8_measured measured "$FULL_8B" --also_w4a16
}

e6_seeds() {
  # headline recipes x calib seeds 1,2 (seed 0 = the campaign/e5a_final runs)
  for s in 1 2; do
    if [ ! -f $OUT/3b_final_s$s.json ]; then
      env $FULL_3B python promix/scripts/rd_acctune.py --config $CFG3 \
        --point 3b_final_s$s --substrate ring --ring_size 512 \
        --alloc measured --gptq_nsamples 512 --percdamp 0.025 \
        --calib_seed $s --c4 --out $OUT/3b_final_s$s.json \
        > $OUT/3b_final_s$s.log 2>&1
    fi
  done
  for s in 1 2; do
    if [ ! -f $OUT/8b_final_s$s.json ]; then
      env $FULL_8B python promix/scripts/rd_acctune.py --config $CFG8 \
        --point 8b_final_s$s --substrate ring --ring_size 512 \
        --alloc measured --gptq_nsamples 512 \
        --calib_seed $s --c4 --out $OUT/8b_final_s$s.json \
        > $OUT/8b_final_s$s.log 2>&1
    fi
  done
}

e6_gate() {
  # 3B three-way allocation gate (rd_validity, R1-off regime) x 3 seeds,
  # same-session protocol (seed 0 re-run for comparability)
  for s in 0 1 2; do
    if [ ! -f $OUT/validity_s$s.json ]; then
      python promix/scripts/rd_validity.py --config $CFG3 --r1_off \
        --calib_seed $s --out $OUT/validity_s$s.json \
        > $OUT/validity_s$s.log 2>&1
    fi
  done
}

case ${1:-all} in
  e5a_3b)    e5a_3b ;;
  e5a_final) e5a_final ;;
  e6_seeds)  e6_seeds ;;
  e6_gate)   e6_gate ;;
  all)       e5a_3b; e5a_final; e6_seeds; e6_gate ;;
  *) echo "unknown phase $1"; exit 1 ;;
esac
touch $OUT/queue_${1:-all}.done
