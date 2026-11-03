#!/bin/bash
# rd-buildup E5b PERFORMANCE build-up — GPU lane 7, runs INSIDE container
# luofan-plaquant, launched detached:
#   docker exec -d -e CUDA_VISIBLE_DEVICES=7 luofan-plaquant \
#     bash /home/luofan/plaquant-buildup/promix/vllm_integ/run_buildup_perf.sh <3b|8b|naive|all>
#
# Cumulative kill-switch ladder (same ckpt, same session, same GPU, one vLLM
# engine per step; env is read by the plugin at runtime so each engine
# process realizes exactly its step's routing):
#   s2_fused_r0   single-launch fused GEMM only (all routing/fusion levers OFF)
#   s3_dispatch   + per-shape dispatch / small-M configs   (PROMIX_SMALLM)
#   s4_splitk     + split-K                                 (PROMIX_SPLITK)
#   s5_gemv       + GEMV/marlin decode routes + hi companion
#                   (PROMIX_DECODE_GEMV, PROMIX_MARLIN, PROMIX_MARLIN_HI)
#   s6_fusions    + pack/gather/cast fusions == shipped default
#                   (PROMIX_QPFM, PROMIX_GATHER_PACK, PROMIX_CAST_FOLD)
# Step 1 (naive 2-launch) is GEMM-level (torch-native torchmix2 from the
# gemm_h2h harness, --ms 128,8192) + analytic serving estimate — see
# docs/rd_buildup_ablation.md for the method.
#
# Protocol: decode TPOT = case 128:1024 b128; prefill TTFT = L2048 b128.
set -x
cd /home/luofan/plaquant-buildup
export PYTHONPATH=/home/luofan/plaquant-buildup PYTHONUNBUFFERED=1
export HF_HOME=/home/luofan/hf_cache
export VLLM_ENABLE_V1_MULTIPROCESSING=0
P=/home/luofan/baselines/venv/bin/python
OUT=/home/luofan/buildup
mkdir -p $OUT

CK3=/home/luofan/vllm_ring512/promix-3b-ring512
CK8=/home/luofan/vllm_ring512/promix-8b-ring512

S2="PROMIX_SMALLM=0 PROMIX_SPLITK=0 PROMIX_DECODE_GEMV=0 PROMIX_MARLIN=0 PROMIX_MARLIN_HI=0 PROMIX_QPFM=0 PROMIX_GATHER_PACK=0 PROMIX_CAST_FOLD=0"
S3="PROMIX_SMALLM=1 PROMIX_SPLITK=0 PROMIX_DECODE_GEMV=0 PROMIX_MARLIN=0 PROMIX_MARLIN_HI=0 PROMIX_QPFM=0 PROMIX_GATHER_PACK=0 PROMIX_CAST_FOLD=0"
S4="PROMIX_SMALLM=1 PROMIX_SPLITK=1 PROMIX_DECODE_GEMV=0 PROMIX_MARLIN=0 PROMIX_MARLIN_HI=0 PROMIX_QPFM=0 PROMIX_GATHER_PACK=0 PROMIX_CAST_FOLD=0"
S5="PROMIX_SMALLM=1 PROMIX_SPLITK=1 PROMIX_DECODE_GEMV=1 PROMIX_MARLIN=1 PROMIX_MARLIN_HI=1 PROMIX_QPFM=0 PROMIX_GATHER_PACK=0 PROMIX_CAST_FOLD=0"
S6="PROMIX_SMALLM=1 PROMIX_SPLITK=1 PROMIX_DECODE_GEMV=1 PROMIX_MARLIN=1 PROMIX_MARLIN_HI=1 PROMIX_QPFM=1 PROMIX_GATHER_PACK=1 PROMIX_CAST_FOLD=1"

serve_step() {  # serve_step <ckpt> <tag> <envstr>
  local ck=$1 tag=$2 envstr=$3
  if [ -f $OUT/$tag.json ]; then echo "SKIP $tag"; return; fi
  env $envstr $P -m promix.vllm_integ.bench_serve --model $ck \
    --tag $tag --dtype float16 \
    --batches 128 --cases 128:1024 --prefill-len 2048 \
    --max-model-len 4096 --gpu-mem-util 0.90 \
    --extra-llm-kwargs '{"max_num_seqs":512}' \
    --out $OUT/$tag.json > $OUT/$tag.log 2>&1
}

model_ladder() {  # model_ladder <3b|8b> <ckpt>
  local m=$1 ck=$2
  serve_step $ck pb_${m}_s2_fused_r0 "$S2"
  serve_step $ck pb_${m}_s3_dispatch "$S3"
  serve_step $ck pb_${m}_s4_splitk   "$S4"
  serve_step $ck pb_${m}_s5_gemv     "$S5"
  serve_step $ck pb_${m}_s6_fusions  "$S6"
}

serve_step16() {  # small-batch pass: decode b16 (where the GEMV/marlin
  local ck=$1 tag=$2 envstr=$3   # m<=16 windows actually engage)
  if [ -f $OUT/$tag.json ]; then echo "SKIP $tag"; return; fi
  env $envstr $P -m promix.vllm_integ.bench_serve --model $ck \
    --tag $tag --dtype float16 \
    --batches 16 --cases 128:1024 --prefill-len 2048 \
    --max-model-len 4096 --gpu-mem-util 0.90 \
    --extra-llm-kwargs '{"max_num_seqs":512}' \
    --out $OUT/$tag.json > $OUT/$tag.log 2>&1
}

model_ladder_b16() {
  local m=$1 ck=$2
  serve_step16 $ck pb_${m}_b16_s2_fused_r0 "$S2"
  serve_step16 $ck pb_${m}_b16_s3_dispatch "$S3"
  serve_step16 $ck pb_${m}_b16_s4_splitk   "$S4"
  serve_step16 $ck pb_${m}_b16_s5_gemv     "$S5"
  serve_step16 $ck pb_${m}_b16_s6_fusions  "$S6"
}

naive_gemm() {
  # step-1 material: torch-native 2-launch (torchmix2) vs ours, GEMM-only,
  # decode-M (128) and prefill-chunk-M (8192); container base env.
  if [ ! -f $OUT/h2h_naive.json ]; then
    python -m promix.eval.gemm_h2h_torchfp4 --models 3B,8B --ms 128,8192 \
      --out $OUT/h2h_naive.json > $OUT/h2h_naive.log 2>&1
  fi
  if [ ! -f $OUT/h2h_naive_m16.json ]; then
    python -m promix.eval.gemm_h2h_torchfp4 --models 3B,8B --ms 16 \
      --out $OUT/h2h_naive_m16.json > $OUT/h2h_naive_m16.log 2>&1
  fi
  # prefill chunk M: engines run max_num_batched_tokens=16384 (log-verified)
  if [ ! -f $OUT/h2h_naive_m16384.json ]; then
    python -m promix.eval.gemm_h2h_torchfp4 --models 3B,8B --ms 16384 \
      --out $OUT/h2h_naive_m16384.json > $OUT/h2h_naive_m16384.log 2>&1
  fi
}

case ${1:-all} in
  3b)    model_ladder 3b $CK3 ;;
  8b)    model_ladder 8b $CK8 ;;
  naive) naive_gemm ;;
  b16)   naive_gemm; model_ladder_b16 3b $CK3; model_ladder_b16 8b $CK8 ;;
  all)   naive_gemm; model_ladder 3b $CK3; model_ladder 8b $CK8 ;;
  *) echo "unknown phase $1"; exit 1 ;;
esac
touch $OUT/perf_${1:-all}.done
