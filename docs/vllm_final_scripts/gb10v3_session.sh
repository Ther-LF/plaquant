#!/bin/bash
# GB10 v3 P0-1 timed session: all methods in ONE boot, back-to-back.
# Order: AdaMix r0.125 anchor -> BF16 -> FP8 -> GPTQ -> NVFP4 -> (r0.0, r1.0) -> T2 gate.
set -u
B=/home/luofan/gb10_sweep/bench
S=/home/luofan/gb10_sweep
LOG=$B/gb10v3_session.log
source $S/env.sh
cd /home/luofan/plaquant
BOOT=$(cat /proc/sys/kernel/random/boot_id)
echo "SESSION_START boot_id=$BOOT $(date '+%F %T') uptime=$(uptime -p)" | tee $LOG

run() { # tag ckpt dtype
  echo "POINT $1 $(date '+%F %T') boot=$(cat /proc/sys/kernel/random/boot_id)" | tee -a $LOG
  bash $S/gb10v3_serve_point.sh "$2" "$1" "$3" >> $LOG 2>&1
  echo "POINT_DONE $1 $(date '+%F %T')" | tee -a $LOG
}

run gb10v3_pm3b_r0.125 /home/luofan/gb10_sweep/ckpt/promix-3b-r0.125 float16
run gb10v3_bf3b  /home/luofan/hf_cache/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331 bfloat16
run gb10v3_fp83b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-fp8 bfloat16
run gb10v3_gptq3b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-gptq_w4a16 bfloat16
run gb10v3_nv3b /home/luofan/baselines/quant/unsloth--Llama-3.2-3B-nvfp4 bfloat16
run gb10v3_pm3b_r0.0 /home/luofan/gb10_sweep/ckpt/promix-3b-r0.0 float16
run gb10v3_pm3b_r1.0 /home/luofan/gb10_sweep/ckpt/promix-3b-r1.0 float16

echo "T2_GATE start $(date '+%F %T')" | tee -a $LOG
python -m promix.vllm_integ.gates.gate_t2 --ckpt /home/luofan/gb10_sweep/ckpt/promix-3b-r0.125 \
  --graphs --out $B/gate_t2_gb10v3_pm3b_r0.125.json >> $LOG 2>&1
echo "T2_RC $? $(date '+%F %T')" | tee -a $LOG
echo "SESSION_END boot_id=$(cat /proc/sys/kernel/random/boot_id) $(date '+%F %T')" | tee -a $LOG
