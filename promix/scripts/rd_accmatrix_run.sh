#!/bin/bash
# rd-accmatrix: fill MEASURED downstream-metric cells (0-shot 6-task + MMLU) for
# the W4A4 peers that the newpeers/mmlu campaigns left PPL-only.
# ONE cell per invocation; GPU lane pinned by caller (-e CUDA_VISIBLE_DEVICES=N).
#   docker exec -d -w /home/luofan/plaquant-accmatrix \
#     -e CUDA_VISIBLE_DEVICES=7 -e HF_HOME=/home/luofan/hf_cache \
#     -e PYTHONPATH=/home/luofan/plaquant-accmatrix -e PYTHONUNBUFFERED=1 \
#     luofan-plaquant bash promix/scripts/rd_accmatrix_run.sh <stage>
set -x
STAGE=$1
OUT=/home/luofan/accmatrix
PM=/home/luofan/plaquant-accmatrix
PEER=$PM/promix/scripts/rd_newacc_peer.py
ACL=/home/luofan/baselines/venv_acl44/bin/python
MMPY=/home/luofan/baselines/venv_mm/bin/python
VLLMPY=/home/luofan/baselines/venv/bin/python
ZS="piqa,arc_easy,arc_challenge,hellaswag,winogrande,lambada_openai"
M3B=/home/luofan/models/Llama-3.2-3B
M8B=/home/luofan/models/llama-3-8b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT
case $STAGE in
# ---- 0-shot 6-task (num_fewshot 0), rd_metrics convention --------------------
zs_arc_3b) cd /home/luofan/baselines/peers/arcquant/ARCQuant && $ACL $PEER --peer arcquant --repo . \
    --model $M3B --quant NVFP4 --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 8 --out $OUT/arc_3b_zs.json ;;
zs_arc_8b) cd /home/luofan/baselines/peers/arcquant/ARCQuant && $ACL $PEER --peer arcquant --repo . \
    --model $M8B --quant NVFP4 --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 4 --out $OUT/arc_8b_zs.json ;;
zs_sharq_3b) cd /home/luofan/baselines/peers/sharq/SharQ && $ACL $PEER --peer sharq --repo . \
    --model $M3B --quant SHARQ --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 8 --out $OUT/sharq_3b_zs.json ;;
zs_sharq_8b) cd /home/luofan/baselines/peers/sharq/SharQ && $ACL $PEER --peer sharq --repo . \
    --model $M8B --quant SHARQ --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 4 --out $OUT/sharq_8b_zs.json ;;
zs_mm_3b) cd /home/luofan/baselines/peers/micromix/MicroMix && $MMPY $PEER --peer micromix --repo . \
    --model $M3B --quant MICROMIX --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 8 --out $OUT/mm_3b_zs.json ;;
zs_mm_8b) cd /home/luofan/baselines/peers/micromix/MicroMix && $MMPY $PEER --peer micromix --repo . \
    --model $M8B --quant MICROMIX --datasets none --lmeval_tasks $ZS --num_fewshot 0 --lmeval_batch 4 --out $OUT/mm_8b_zs.json ;;
# ---- MMLU 5-shot for MicroMix (CoQuant handled separately) -------------------
mmlu_mm_3b) cd /home/luofan/baselines/peers/micromix/MicroMix && $MMPY $PEER --peer micromix --repo . \
    --model $M3B --quant MICROMIX --datasets none --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 8 --out $OUT/mm_3b_mmlu.json ;;
mmlu_mm_8b) cd /home/luofan/baselines/peers/micromix/MicroMix && $MMPY $PEER --peer micromix --repo . \
    --model $M8B --quant MICROMIX --datasets none --lmeval_tasks mmlu --num_fewshot 5 --lmeval_batch 4 --out $OUT/mm_8b_mmlu.json ;;
# ---- NVFP4 0-shot via vLLM backend (same stack as committed NVFP4 rows) ------
zs_nv_3b) $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/baselines/quant/unsloth--Llama-3.2-3B-nvfp4,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks $ZS --num_fewshot 0 --batch_size auto --output_path $OUT/nv_3b_zs ;;
zs_nv_8b) $VLLMPY -m lm_eval --model vllm \
    --model_args pretrained=/home/luofan/baselines/quant/unsloth--llama-3-8b-nvfp4,gpu_memory_utilization=0.85,max_model_len=4096 \
    --tasks $ZS --num_fewshot 0 --batch_size auto --output_path $OUT/nv_8b_zs ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "RD_ACCMATRIX_STAGE_DONE $STAGE rc=$?"
