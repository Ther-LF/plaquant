#!/bin/bash
# waiter.sh <gpu> [need_mib] [max_tries] [sleep_s] -- block until GPU has >= need_mib free
G=$1; NEED=${2:-30000}; N=${3:-40}; IV=${4:-30}
for i in $(seq 1 $N); do
  usem=$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  u=$(nvidia-smi --id=$G --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d " ")
  free=$(( 97887 - usem ))
  echo "$(date +%H:%M:%S) gpu$G used=${usem}MiB free=${free}MiB util=${u}%"
  if [ "$free" -ge "$NEED" ] && [ "$u" -le 25 ]; then exit 0; fi
  sleep $IV
done
echo "TIMEOUT waiting for gpu $G"; exit 1
