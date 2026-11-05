#!/bin/bash
# rd-accmatrix: CoQuant 0-shot / MMLU via its own shipped ptq.py (a_cov control
# variant = the committed representative row). Reproduces the committed wiki PPL
# in the SAME invocation (printed before lm-eval) as the trust gate.
# ONE cell per call; GPU lane pinned by caller. Runs on container python (CoQuant
# pins transformers 4.48.1 = container default). patch5 = num_fewshot + results dump.
set -x
STAGE=$1
OUT=/home/luofan/accmatrix
CQ=/home/luofan/baselines/peers/coquant/CoQuant
ZS="piqa,arc_easy,arc_challenge,hellaswag,winogrande,lambada_openai"
MP=${MASTER_PORT:-29561}
mkdir -p $OUT
cd $CQ
common() { # $1=model $2=outdir $3=modeltag $4=tasks $5=fewshot $6=lmbatch $7=jsontag
  RD_LMEVAL_OUT=$OUT/coq_${7}.json \
  torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MP fake_quant/ptq.py \
    --input_model $1 --per_device_eval_batch_size 4 --model_max_length 2048 \
    --fp16 False --bf16 True \
    --w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 --high_bits 8 --low_bits 2 \
    --w_clip --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
    --high_fraction 0.125 --low_fraction 0.0 --rotate_mode "mix" \
    --optimized_rotation_path $2/R-high-0.125-low-0.0-sparse-0.0-${3}.bin \
    --optimized_basis_path $2/U-a_cov-wikitext-512-${3}.bin \
    --rotation_granularity per_layer --rotate \
    --tasks "$4" --num_fewshot $5 --lm_eval_batch_size $6
}
case $STAGE in
coq_zs_3b)   common /home/luofan/models/Llama-3.2-3B $CQ/output-3b-acov models-Llama-3.2-3B "$ZS" 0 8 3b_zs ;;
coq_zs_8b)   common /home/luofan/models/llama-3-8b   $CQ/output-8b-acov models-llama-3-8b   "$ZS" 0 4 8b_zs ;;
coq_mmlu_3b) common /home/luofan/models/Llama-3.2-3B $CQ/output-3b-acov models-Llama-3.2-3B mmlu 5 8 3b_mmlu ;;
coq_mmlu_8b) common /home/luofan/models/llama-3-8b   $CQ/output-8b-acov models-llama-3-8b   mmlu 5 4 8b_mmlu ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "RD_ACCMATRIX_COQ_DONE $STAGE rc=$?"
