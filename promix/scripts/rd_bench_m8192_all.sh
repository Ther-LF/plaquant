#!/bin/bash
# rd-bench: serialized M=8192 head-to-head suite. Run INSIDE container
# luofan-plaquant from /home/luofan/plaquant-bench. All methods back-to-back on
# ONE idle GPU (same clock/power regime); cudaEvent median-of-50 inside each
# script. Usage: bash promix/scripts/rd_bench_m8192_all.sh <GPU>
set -x
GPU=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
D=docs/rd_baselines_m8192_data
mkdir -p $D
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv

# 1) ProMix fused (+ our pure-NVFP4 + fp16 reference)
PYTHONPATH=/home/luofan/plaquant-bench TORCH_EXTENSIONS_DIR=/home/luofan/plaquant-bench/.torchext \
  python3 promix/eval/rd_bench_m8192_ours.py --m 8192 --out $D/ours.json

# 2) vLLM NVFP4 (baselines venv python)
/home/luofan/baselines/venv/bin/python promix/eval/rd_bench_m8192_vllm.py --m 8192 --out $D/vllm_nvfp4.json

# 3) ARCQuant
python3 promix/eval/rd_bench_m8192_arcquant.py --m 8192 --out $D/arcquant.json

# 4) MicroMix
python3 promix/eval/rd_bench_m8192_micromix.py --m 8192 --out $D/micromix.json

# 5) SharQ
python3 promix/eval/rd_bench_m8192_sharq.py --m 8192 --out $D/sharq.json

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv
echo SUITE_DONE
