#!/bin/bash
# T2 gate + reduced serving protocol for one ProMix r-sweep checkpoint.
# (Repo copy of the campaign's bench_final/run_sweep_point.sh, parameterized
# for the rd-densesweep repo location + bench dir, with the T2 clause checked
# BETWEEN gate and serve: a point that fails |dPPL|<=0.1 vs its own exporter
# reference is NOT timed — the serve numbers would advertise a broken point.)
#
# Protocol (identical to the committed sweep points in docs/vllm_final_data):
#   gate   : gate_t2 --graphs  (CUDA-graph engine, the serving configuration)
#   main   : ISL/OSL 1024:128 (prefill-heavy) + 128:1024 (decode-heavy),
#            batch {16,64,128,256}, + prefill TTFT L2048 sweep, max-model-len
#            4096, gpu-mem-util 0.90, max_num_seqs 512
#   qserve : ISL/OSL 1024:512 continuous batching (QServe protocol), b {128,256}
#
# Usage: run_sweep_point.sh <ckpt> <tag> [bench_dir] [repo_dir]
set -x
CK=$1; TAG=$2
B=${3:-/home/luofan/vllm_integ/bench_final}
R=${4:-/home/luofan/plaquant-vllm-integ}
P=/home/luofan/baselines/venv/bin/python
export PYTHONPATH=$R PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HOME=/home/luofan/hf_cache
cd $R
$P -m promix.vllm_integ.gates.gate_t2 --ckpt $CK --graphs \
  --out $B/gate_t2_${TAG}.json > $B/gate_${TAG}.log 2>&1
$P - "$B/gate_t2_${TAG}.json" <<'EOF' || { touch $B/${TAG}.t2fail; exit 3; }
import json, sys
g = json.load(open(sys.argv[1]))
d = abs(g["dppl"])
print(f"T2 {'PASS' if d <= 0.1 else 'FAIL'} |dPPL|={d:.4f}")
sys.exit(0 if d <= 0.1 else 1)
EOF
$P -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_main --dtype float16 \
  --batches 16,64,128,256 --cases 1024:128,128:1024 --prefill-len 2048 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_main.json > $B/${TAG}_main.log 2>&1
$P -m promix.vllm_integ.bench_serve --model $CK --tag ${TAG}_qserve --dtype float16 \
  --batches 128,256 --cases 1024:512 --prefill-len 0 \
  --max-model-len 4096 --gpu-mem-util 0.90 --extra-llm-kwargs "{\"max_num_seqs\":512}" \
  --out $B/${TAG}_qserve.json > $B/${TAG}_qserve.log 2>&1
touch $B/${TAG}.done
