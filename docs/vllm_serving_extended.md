# Extended serving protocol — long context, big batch, ShareGPT (rd-densesweep)

Status: **IN PROGRESS** (queue `queue_ext.txt`; tables harvested into this doc
as points land — `python promix/scripts/densesweep/harvest_extended.py`).

## Protocol

Two additions on top of the five-way/r-sweep serving protocol
(`docs/vllm_final_fiveway.md` §0; ISL = input sequence length, OSL = output
sequence length, all through the same vLLM 0.32 cu130 stack, one engine per
idle GPU):

1. **Fixed-length synthetic grid (the standard serving measurement)** —
   latency depends on request *shape*, not content, so fixed-length synthetic
   requests are the standard protocol (vLLM/QServe/TRT-LLM all measure this
   way). Grid: ISL ∈ {128, 1024, 2048, 4096} × OSL ∈ {128, 512, 1024} ×
   batch ∈ {16, 64, 128, 256, 512}, engine max-model-len 5120, fp16 KV,
   `max_num_seqs=512`, gpu-mem-util 0.90. Cells beyond the KV budget show up
   as vLLM preemption/recompute (throughput collapse) or per-case errors —
   both recorded; the **max cleanly-servable batch per context length** is a
   KV-capacity result in itself (4-bit weight rows keep tens of GiB more KV
   headroom than BF16).
2. **ShareGPT real-distribution point (the realistic mixed load)** — one
   throughput run per method against a `vllm serve` endpoint with vLLM's
   `bench serve --dataset-name sharegpt` (ShareGPT_V3 conversational ISL/OSL
   distribution, 512 prompts): request throughput (req/s), output tok/s, mean
   TTFT/TPOT. This matches the vLLM-paper/QServe "real workload" convention;
   it complements (does not replace) the fixed-length grid.

Methods: BF16 / FP8 W8A8 / GPTQ-Marlin W4A16 / AWQ-Marlin W4A16 / NVFP4 W4A4 /
ProMix (shipped r=1/8 **plus** the persistent r=1/16 and r=1/4 ratio
checkpoints — per-method comparison points; the full 20-point ProMix ratio
curve lives in the main r-sweep, §4 of the five-way doc), at 3B and 8B
(the five-way sizes). Peers: SharQ (3B, fixed-length grid only — its vLLM
integration is an out-of-tree registration wrapper bound to the bench harness)
and FP-Quant/QuTLASS NVFP4 (3B) where the peer checkpoints exist.
Qwen3-32B remains deferred (architecture blocker — head_dim 128 ≠
hidden/heads, no had100; see `docs/rd_scale_report.md`).

## Results

(harvested tables land here per committed point; raw JSONs in
`docs/vllm_final_data/ext_*.json` and `docs/vllm_final_data/sgpt_*.json`)
