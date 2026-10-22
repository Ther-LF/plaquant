#!/bin/bash
# Idle-GPU waiter + launch the decode-throughput harness pinned to a genuinely
# quiet GPU (util<=3%, mem<=2000MiB for 3 consecutive samples). Run INSIDE the
# luofan-plaquant container. Detached: nohup + JSON/log to disk for harvest.
set -u
cd /home/luofan/plaquant
PHASES="${PHASES:-correct,decode,diagnose}"
PROMPT="${PROMPT:-32}"
GEN="${GEN:-128}"
OUT="${OUT:-/home/luofan/decode_throughput.json}"

pick_idle() {
  for g in 0 1 2 3 4 5 6 7; do
    ok=1
    for s in 1 2 3; do
      line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null)
      mem=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1);print $1}')
      util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2);print $2}')
      if [ -z "$mem" ] || [ "$mem" -gt 2000 ] || [ "$util" -gt 3 ]; then ok=0; break; fi
      sleep 2
    done
    if [ "$ok" = "1" ]; then echo $g; return 0; fi
  done
  echo -1
}

echo "waiting for an idle GPU..."
GPU=-1
for try in $(seq 1 600); do
  GPU=$(pick_idle)
  if [ "$GPU" != "-1" ]; then break; fi
  sleep 5
done
if [ "$GPU" = "-1" ]; then echo "NO_IDLE_GPU_FOUND"; exit 1; fi
echo "IDLE GPU=$GPU -> launching decode harness"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 TORCH_CUDA_ARCH_LIST=12.0a \
  stdbuf -oL -eL python3 -m promix.eval.decode_throughput \
    --phases "$PHASES" --prompt "$PROMPT" --gen "$GEN" --out "$OUT"
echo "HARNESS_EXIT=$?"
