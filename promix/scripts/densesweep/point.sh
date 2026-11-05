#!/bin/bash
# rd-densesweep: shared job dispatcher — one line of a queue file.
#   point.sh serve <size> <r>       serving-curve point (export -> T2 -> serve)
#   point.sh acc   <size> <r>       accuracy-curve point (fake-quant wiki+C4)
#   point.sh ext   <size> <method>  extended long-ctx/big-batch grid (one method)
#   point.sh sgpt  <size> <method>  ShareGPT real-distribution point (one method)
# size in {1b,3b,8b,70b}; r = FP8 ring-budget fraction (e.g. 0.05 .. 1.00).
# Runs INSIDE the luofan-plaquant container with CUDA_VISIBLE_DEVICES pinned
# by the queue runner. See densesweep_runner.sh for the lane loop.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
KIND=$1; SIZE=$2; ARG=$3
case $KIND in
  serve) exec bash $HERE/serve_point.sh $SIZE $ARG ;;
  acc)   exec bash $HERE/acc_point.sh   $SIZE $ARG ;;
  ext)   exec bash $HERE/ext_point.sh   $SIZE $ARG ;;
  sgpt)  exec bash $HERE/sgpt_point.sh  $SIZE $ARG ;;
  *) echo "unknown job kind: $KIND" >&2; exit 2 ;;
esac
