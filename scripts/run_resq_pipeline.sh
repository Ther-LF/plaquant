#!/bin/bash
# Run Step 1 (rotation opt) + Step 2 (PTQ eval) for a single config.
# Step 2 currently fails on lm_eval (P6 known issue); the wikitext PPL is
# printed before the failure, so we keep going regardless.
#
# Usage:
#   bash scripts/run_resq_pipeline.sh <config> <gpu> <port_base>
# e.g.:
#   bash scripts/run_resq_pipeline.sh promix/configs/llama-3.2-3b-w4a4.yaml 1 29520
set +e
cfg="$1"
gpu="$2"
port="${3:-29520}"
tag=$(basename "$cfg" .yaml)

cd "$(dirname "$0")/.."
mkdir -p logs

source .venv/bin/activate 2>/dev/null

echo "[$(date '+%H:%M:%S')] === ${tag} step 1 (rotation opt) on GPU ${gpu} ==="
CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 --master_port="$port" \
    -m promix.quantize.optimize_rotation \
    --config "$cfg" --output_dir ./rotation \
    --max_steps 100 --learning_rate 1.5

echo "[$(date '+%H:%M:%S')] === ${tag} step 2 (ptq eval) on GPU ${gpu} ==="
CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/torchrun \
    --nnodes=1 --nproc_per_node=1 --master_port=$((port+1)) \
    -m promix.eval.ptq --config "$cfg"

echo "[$(date '+%H:%M:%S')] === ${tag} done ==="
