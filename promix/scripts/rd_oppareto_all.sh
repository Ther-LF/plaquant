#!/bin/bash
# rd-oppareto (E4): operator-level Pareto suite. Run INSIDE container
# luofan-plaquant from /home/luofan/plaquant-oppareto on ONE idle GPU
# (default 6) — all methods back-to-back in one session so they share the same
# clock/power regime (rd-bench protocol). cudaEvent median-of-50 per script.
#
# Usage: bash promix/scripts/rd_oppareto_all.sh <GPU>
set -x
GPU=${1:-6}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
D=docs/rd_oppareto_data
mkdir -p $D
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv

# 1) OUR full latency-vs-r curve (shipping dispatch), M=2048 + 8192 —
#    one invocation per shape so each shape's JSON is a harvestable checkpoint
for S in 3B:qkvo 3B:gate_up 3B:down 8B:qkvo 8B:gate_up 8B:down; do
  TAG=$(echo $S | tr ':' '_')
  PYTHONPATH=$ROOT TORCH_EXTENSIONS_DIR=$ROOT/.torchext \
    python3 promix/eval/rd_oppareto_ours.py --ms 2048,8192 --shapes $S \
    --out $D/ours_curve_$TAG.json
done

# 2) torch-native FP4 / FP8 / bf16 points
PYTHONPATH=$ROOT TORCH_EXTENSIONS_DIR=$ROOT/.torchext \
  python3 promix/eval/rd_oppareto_torch.py --m 2048 --out $D/torch_m2048.json
PYTHONPATH=$ROOT TORCH_EXTENSIONS_DIR=$ROOT/.torchext \
  python3 promix/eval/rd_oppareto_torch.py --m 8192 --out $D/torch_m8192.json

# 3) vLLM NVFP4 (baselines venv python)
/home/luofan/baselines/venv/bin/python promix/eval/rd_bench_m8192_vllm.py --m 2048 --out $D/vllm_m2048.json
/home/luofan/baselines/venv/bin/python promix/eval/rd_bench_m8192_vllm.py --m 8192 --out $D/vllm_m8192.json

# 4) ARCQuant (K-extended GEMM, S in {0,256,512}; shipped S<=512)
python3 promix/eval/rd_bench_m8192_arcquant.py --m 2048 --out $D/arcquant_m2048.json
python3 promix/eval/rd_bench_m8192_arcquant.py --m 8192 --out $D/arcquant_m8192.json

# 5) MicroMix (segmented MXFP4/6/8 GEMM; CUDA-13 correctness unvalidated as shipped)
python3 promix/eval/rd_bench_m8192_micromix.py --m 2048 --out $D/micromix_m2048.json
python3 promix/eval/rd_bench_m8192_micromix.py --m 8192 --out $D/micromix_m8192.json

# 6) SharQ (dual-GEMM: 4:8 sparse backbone + dense residual, own CUDA harness)
python3 promix/eval/rd_bench_m8192_sharq.py --m 2048 --out $D/sharq_m2048.json
python3 promix/eval/rd_bench_m8192_sharq.py --m 8192 --out $D/sharq_m8192.json

nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv
echo OPPARETO_SUITE_DONE
