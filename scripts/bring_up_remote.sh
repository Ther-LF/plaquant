#!/bin/bash
# Run on the remote worker (B20Z) after a container restart.
# Brings everything back up in under a minute:
#   - persistent venv at /root/plaquant/.venv (must already exist)
#   - HF_TOKEN exported (read from ~/.bashrc if already persisted)
#   - keepalive on GPUs 3-7 (5 cards)
#   - basis runs for 1B/3B/8B on GPUs 0/1/2 (resume from HF cache)
#
# Usage:
#   bash scripts/bring_up_remote.sh                # all phases
#   bash scripts/bring_up_remote.sh keep_only      # only keepalive
#   bash scripts/bring_up_remote.sh basis_only     # only basis
set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p logs rotation

# Pull HF token from ~/.bashrc if not already set
if [ -z "$HF_TOKEN" ] && grep -q '^export HF_TOKEN=' "$HOME/.bashrc" 2>/dev/null; then
    eval "$(grep '^export HF_TOKEN=' "$HOME/.bashrc")"
fi
if [ -z "$HF_TOKEN" ]; then
    echo "WARN: HF_TOKEN not set; downloads will be rate-limited"
fi
export HF_TOKEN

mode="${1:-all}"

start_keep() {
    for g in 3 4 5 6 7; do
        if pgrep -f "keepalive_matmul.py.*--size 4096.*$g\b" >/dev/null 2>&1; then
            echo "keepalive g$g already running"; continue
        fi
        CUDA_VISIBLE_DEVICES=$g nohup python scripts/keepalive_matmul.py \
            --sleep 0.0 --size 4096 > "logs/keep_g${g}.log" 2>&1 &
        echo "keepalive g$g started PID=$!"
    done
}

start_basis() {
    declare -A CFG=(
        [1B]="promix/configs/llama-3.2-1b-resq.yaml"
        [3B]="promix/configs/llama-3.2-3b.yaml"
        [8B]="promix/configs/llama-3-8b.yaml"
    )
    declare -A GPU=([1B]=0 [3B]=1 [8B]=2)
    for tag in 1B 3B 8B; do
        if pgrep -f "promix.quantize.basis.*${CFG[$tag]}" >/dev/null 2>&1; then
            echo "basis $tag already running"; continue
        fi
        # Skip if basis output already exists
        out_glob=$(ls rotation/U-*"$(basename "${CFG[$tag]}" .yaml | sed -E 's/llama-?3?-?\.?2?-?//' )"* 2>/dev/null | head -1 || true)
        # (heuristic above isn't perfect — just relaunch to be safe; basis.py
        # internally checks "Basis already exists" and skips computation.)
        CUDA_VISIBLE_DEVICES=${GPU[$tag]} nohup python -m promix.quantize.basis \
            --config "${CFG[$tag]}" --output_dir ./rotation --nsamples 512 \
            > "logs/basis_${tag,,}.log" 2>&1 &
        echo "basis $tag (gpu=${GPU[$tag]}) started PID=$!"
    done
}

case "$mode" in
    keep_only) start_keep ;;
    basis_only) start_basis ;;
    all) start_keep; start_basis ;;
    *) echo "unknown mode: $mode"; exit 1 ;;
esac

sleep 3
echo
echo "=== state ==="
ps -o pid,etime,pcpu,cmd -p $(pgrep -f 'keepalive_matmul|promix.quantize.basis' | tr '\n' ',' | sed 's/,$//') 2>/dev/null
echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv | head -10
