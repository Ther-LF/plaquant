#!/bin/bash
# quant_gen.sh <scheme> <model_glob> [extra quantize.py args...]
cd /home/luofan/baselines
export HF_HOME=/home/luofan/hf_cache
SCHEME=$1
M=$(ls -d "$2")
shift 2
BASE=$(basename "$(dirname "$(dirname "$M")")")
TAG=${BASE#models--}
exec ./venv/bin/python scripts/quantize.py --model "$M" --output /home/luofan/baselines/quant/${TAG}-${SCHEME} --scheme "$SCHEME" "$@"
