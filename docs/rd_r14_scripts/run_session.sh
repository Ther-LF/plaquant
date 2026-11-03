#!/bin/bash
# rd-r14: 3B r=1/4 decode-anomaly re-measure — one self-anchored session on ONE idle GPU.
# Three engines sequentially on the SAME GPU: BF16 anchor, ProMix r=1/8 control,
# ProMix r=1/4 (the anomalous point). Cases cover the anomalous cells
# (decode-heavy 128:1024 b128/b256, 1024:128 b256, qserve c256, prefill L2048).
# Usage: run_session.sh <gpu> <sess> [order]   order=rev runs r25 before r125.
set -x
R=/home/luofan/plaquant-r14probe
OUT=$R/docs/rd_r14_data
P=/home/luofan/baselines/venv/bin/python
GPU=$1; S=$2; ORDER=${3:-fwd}
BF=/home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331
R18=/home/luofan/vllm_integ/promix-llama-3.2-3b
R25=/home/luofan/vllm_integ/promix-3b-r0.25
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$R PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HOME=/home/luofan/hf_cache
export TORCH_EXTENSIONS_DIR=/home/luofan/.cache/torch_ext_r14probe
mkdir -p $OUT
cd $R
( while true; do nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used --format=csv,noheader -i $GPU >> $OUT/r14_${S}_gpumon.csv; sleep 20; done ) &
MON=$!
run_cfg () {  # ckpt tag dtype
  $P -m promix.vllm_integ.bench_serve --model $1 --tag r14_${S}_$2_main --dtype $3 \
    --batches 16,128,256 --cases 1024:128,128:1024 --prefill-len 2048 \
    --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
    --out $OUT/r14_${S}_$2_main.json > $OUT/r14_${S}_$2_main.log 2>&1
  $P -m promix.vllm_integ.bench_serve --model $1 --tag r14_${S}_$2_qserve --dtype $3 \
    --batches 256 --cases 1024:512 --prefill-len 0 \
    --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
    --out $OUT/r14_${S}_$2_qserve.json > $OUT/r14_${S}_$2_qserve.log 2>&1
}
run_cfg $BF bf16 bfloat16
if [ "$ORDER" = "rev" ]; then
  run_cfg $R25 r25 float16
  run_cfg $R18 r125 float16
else
  run_cfg $R18 r125 float16
  run_cfg $R25 r25 float16
fi
kill $MON
touch $OUT/r14_${S}.done
