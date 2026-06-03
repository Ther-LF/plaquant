#!/bin/bash
# Run PTQ evaluation using ProMix pipeline
# This uses our promix/eval/ptq.py with config-driven approach
cd $(dirname $0)/../..
python -m promix.eval.ptq --config promix/configs/llama-3.2-1b-resq.yaml
