#!/bin/bash
# Launch a wave of ablation jobs. Each line in the jobs file:
#   size|mode|extra_args|tag
# GPUs and ports assigned round-robin. Runs inside container, detached.
set -u
JOBS_FILE="$1"
GPUS=(0 2 3 6 7 5 1 4)
cd /home/luofan/plaquant
mkdir -p /home/luofan/ablation_results/logs
i=0
while IFS='|' read -r size mode extra tag; do
  [ -z "$size" ] && continue
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  port=$((29600 + i))
  log=/home/luofan/ablation_results/logs/${size}_${mode}_${tag}.log
  CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port nohup python promix/scripts/ablation_acc_per_bit.py \
    --size "$size" --mode "$mode" $extra --tag "$tag" --out /home/luofan/ablation_results \
    > "$log" 2>&1 &
  echo "launched gpu=$gpu port=$port size=$size mode=$mode extra='$extra' tag=$tag pid=$!"
  i=$((i+1))
  sleep 1
done < "$JOBS_FILE"
echo "WAVE_LAUNCHED $i jobs"
