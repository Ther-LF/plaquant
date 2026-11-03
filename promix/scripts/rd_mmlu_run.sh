#!/bin/bash
# rd-mmlu E1 campaign runner — ONE stage per invocation, ALL stages pinned to a
# single GPU lane by the caller (docker exec -e CUDA_VISIBLE_DEVICES=4).
#
#   docker exec -d -w /home/luofan/plaquant-mmlu \
#     -e CUDA_VISIBLE_DEVICES=4 -e MASTER_PORT=296xx \
#     -e HF_HOME=/home/luofan/hf_cache -e PYTHONPATH=/home/luofan/plaquant-mmlu \
#     -e PYTHONUNBUFFERED=1 luofan-plaquant \
#     bash promix/scripts/rd_mmlu_run.sh <stage>
#
# Stages: pm_1b pm_3b pm_8b  (ProMix NEW recipe: wiki gate + MMLU + 0-shot;
#                             1B = 0-shot only, MMLU scope is 3B/8B)
#         bf16_3b bf16_8b    (bf16 MMLU anchor, lm-eval 0.4.5 HF)
#         arc_3b arc_8b sharq_3b sharq_8b   (peers as-shipped, venv_acl44)
#         nv_3b nv_8b nvb_3b nvb_8b (vLLM-NVFP4 ckpt / bf16 bridge, lm-eval vllm)
set -x
STAGE=$1
OUT=/home/luofan/mmlu
PM=/home/luofan/plaquant-mmlu
ACL=/home/luofan/baselines/venv_acl44/bin/python
VLLMPY=/home/luofan/baselines/venv/bin/python
PEER_SCRIPT=$PM/promix/scripts/rd_newacc_peer.py
mkdir -p $OUT
cd $PM

case $STAGE in
# ---- ProMix NEW recipe (rd_accuracy_tuning.md shipping recipes) -------------
pm_1b)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PROMIX_ACT_MSE_STEPS_UP=2 python promix/scripts/rd_mmlu.py \
    --config promix/configs/llama-3.2-1b-mxfp8-nvfp4.yaml --point pm_1b \
    --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 1024 \
    --expect_wiki 14.1694 --zeroshot --batch_size 16 \
    --out $OUT/pm_1b.json ;;
pm_3b)
  # quantized-model lm-eval OOMs at bs32 (act-quant intermediates on ~2k-tok
  # 5-shot contexts); bs8 + expandable_segments fits.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 python promix/scripts/rd_mmlu.py \
    --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --point pm_3b \
    --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 512 \
    --percdamp 0.025 --expect_wiki 8.2180 --mmlu --zeroshot --batch_size 8 \
    --out $OUT/pm_3b.json ;;
pm_8b)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 python promix/scripts/rd_mmlu.py \
    --config promix/configs/llama-3-8b-mxfp8-nvfp4.yaml --point pm_8b \
    --substrate ring --ring_size 512 --alloc measured --gptq_nsamples 512 \
    --expect_wiki 6.5642 --mmlu --zeroshot --batch_size 4 \
    --out $OUT/pm_8b.json ;;
# ---- bf16 MMLU anchors (same lm-eval 0.4.5 HF harness) ----------------------
bf16_3b)
  python promix/scripts/rd_mmlu.py \
    --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --point bf16_3b \
    --alloc none --mmlu --batch_size 32 --out $OUT/bf16_3b.json ;;
bf16_8b)
  python promix/scripts/rd_mmlu.py \
    --config promix/configs/llama-3-8b-mxfp8-nvfp4.yaml --point bf16_8b \
    --alloc none --mmlu --batch_size 16 --out $OUT/bf16_8b.json ;;
# ---- peers as-shipped (quantize exactly as the committed PPL runs) ----------
arc_3b)
  cd /home/luofan/baselines/peers/arcquant/ARCQuant && \
  $ACL $PEER_SCRIPT --peer arcquant --repo . \
    --model /home/luofan/models/Llama-3.2-3B --quant NVFP4 --datasets none \
    --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 32 \
    --out $OUT/arc_3b_mmlu.json ;;
arc_8b)
  cd /home/luofan/baselines/peers/arcquant/ARCQuant && \
  $ACL $PEER_SCRIPT --peer arcquant --repo . \
    --model /home/luofan/models/llama-3-8b --quant NVFP4 --datasets none \
    --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 16 \
    --out $OUT/arc_8b_mmlu.json ;;
sharq_3b)
  cd /home/luofan/baselines/peers/sharq/SharQ && \
  $ACL $PEER_SCRIPT --peer sharq --repo . \
    --model /home/luofan/models/Llama-3.2-3B --quant SHARQ --datasets none \
    --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 32 \
    --out $OUT/sharq_3b_mmlu.json ;;
sharq_8b)
  cd /home/luofan/baselines/peers/sharq/SharQ && \
  $ACL $PEER_SCRIPT --peer sharq --repo . \
    --model /home/luofan/models/llama-3-8b --quant SHARQ --datasets none \
    --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 16 \
    --out $OUT/sharq_8b_mmlu.json ;;
# ---- vLLM-NVFP4 (llm-compressor ckpt through lm-eval vllm backend) ----------
nv_3b)
  $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-nvfp4,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path $OUT/nv_3b_mmlu ;;
nv_8b)
  $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/baselines/quant/unsloth--llama-3-8b-nvfp4,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path $OUT/nv_8b_mmlu ;;
# bf16 through the SAME vllm harness = cross-harness bridge for the nv rows
nvb_3b)
  $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/models/Llama-3.2-3B,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path $OUT/nvb_3b_mmlu ;;
nvb_8b)
  $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/models/llama-3-8b,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks mmlu --num_fewshot 5 --batch_size auto \
    --output_path $OUT/nvb_8b_mmlu ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "RD_MMLU_STAGE_DONE $STAGE rc=$?"
