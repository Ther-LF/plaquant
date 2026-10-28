# Rate-Distortion Allocation — End-to-End Performance Payoff (R1-off, 3B)

**Branch:** `feat/rd-perf` (off `feat/rd-alloc`) · **Hardware:** RTX PRO 6000 Blackwell (SM120), docker `luofan-plaquant`, GPU 1 · **Date:** 2026-10-28

**Question:** push the average FP8 ring-budget DOWN (more of the network served
in NVFP4 instead of MXFP8) at iso-accuracy — how much REAL end-to-end speedup
does it buy on the real packed NVFP4+MXFP8 fused kernel?

**Harness:** `promix/scripts/rd_perf.py`. Accuracy (PPL) via the exact
`rd_validity` R1-off path (per-ring measured-MSE ΔD allocation → reorder →
GPTQ → wikitext-2 PPL, same protocol). Latency via the real packed fused-kernel
model (`full_network.wire_real_kernel`), FULL model forward, cudaEvent timing,
warmup + median: prefill (TTFT, M=2048 long-prompt forward) and decode (TPOT,
per-token M=1 KV-cache forward, decode-GEMV weight-streaming path).
`CUDA_VISIBLE_DEVICES=1`, iters=15, warmup=5, decode prompt=256 gen=48.

Latency is weight-value-independent and depends only on each Linear's **K_high
count** (FP8 FLOPs/bytes), not on *which* rings are FP8 — in deployment the ring
selection is folded offline into the PCA rotation (no per-forward gather). So the
latency build installs the per-Linear K_high counts via the opt-in
`model._rd_khigh_override` seam with NO runtime reorder; the accuracy build keeps
`rd_validity`'s reorder path (PPL *does* depend on the selection).

## Results

Baseline = **uniform-1/8** real-kernel model (shipped split). Speedup `x` =
`baseline_latency / row_latency` (>1 = faster).

| allocation            | avg FP8 | wikitext-2 PPL | prefill (TTFT) | decode (TPOT) | prefill x | decode x |
|-----------------------|:------:|:--------------:|:--------------:|:-------------:|:---------:|:--------:|
| uniform-1/8 (baseline)| 0.1250 | **8.3924**     | 52.88 ms       | 24.17 ms      | 1.000     | 1.000    |
| measured-MSE  1/8     | 0.1250 | **8.3124**     | 53.12 ms       | 23.84 ms      | 0.995     | 1.014    |
| measured-MSE  1/12    | 0.0882 | 8.3740         | 52.86 ms       | 24.52 ms      | 1.000     | 0.986    |
| measured-MSE  1/16    | 0.0697 | 8.3774         | 52.48 ms       | 24.06 ms      | **1.008** | **1.005**|
| measured-MSE  1/24    | 0.0513 | 8.4328         | 52.63 ms       | 23.78 ms      | 1.005     | 1.017    |

(`target_rings`: 1288 / 859 / 644 / 429 of 10304 total over 168 reallocated
Linears; realized avg FP8 < target at low budgets because the single global
ΔD threshold concentrates FP8 on high-ΔD layers. All rows built & ran the real
fused kernel directly — `fallback=False` everywhere, i.e. the kernel handles
K_high==0 pure-NVFP4 Linears.)

## Iso-accuracy payoff (THE number)

Smallest budget whose measured-MSE PPL is still ≤ uniform-1/8's PPL (8.3924):

- **b = 1/16 (avg FP8 0.0697), PPL 8.3774 ≤ 8.3924.**
- e2e speedup at that iso-accuracy point: **prefill ×1.008, decode ×1.005.**

**The performance payoff is ~1.00× — negligible (within run-to-run noise).**
1/24 (8.4328) is the first budget to exceed the uniform PPL, so it is *not*
iso-accuracy.

## Interpretation (honest negative result)

The rate-distortion allocation DOES buy real accuracy headroom: at the matched
1/8 budget the measured-MSE allocation beats uniform (8.3124 vs 8.3924), and it
holds ≤ uniform-1/8 PPL all the way down to a **1/16 average FP8 budget** (half
the FP8). But that headroom does **not** convert to end-to-end speedup on the
current SM120 kernel: across 1/8→1/24 the full-model prefill and decode latency
stay flat at ×1.00 ± ~1.5%.

Why: the fused 128×128 kernel runs BOTH an NVFP4 (low) and an MXFP8 (high) GEMM
over a fixed total K. Shrinking K_high just moves channels from the FP8 segment
to the FP4 segment — total K is unchanged, so the wall-clock barely moves; the
decode-GEMV weight-byte reduction (8.5→4.5 bit on a minority of rings) is
likewise marginal against the whole-model forward (attention, norms, RoPE,
lm_head run unchanged). Realizing a payoff would need a kernel whose cost
actually scales with the FP8 fraction (short-circuit / specialize the high
segment when K_high is small, or a per-layer budget-specialized GEMM config) —
the allocation is static per-model, so such specialization is feasible.

## Caveats

- **R1-off lower bound.** These PPLs use R1-off (pure PCA, identity main
  rotation) — the cheapest regime, slightly worse than the shipped R-on recipe
  (R-on 3B ≈ 8.30 vs R1-off uniform 8.3924 here). This curve is therefore the
  **R1-off lower bound on the payoff**; the R-on version (built in parallel on
  `feat/rd-alloc`) will shift accuracy and move the iso-accuracy knee. Report
  this as the R1-off perf signal, not the final number.
- The accuracy signal reproduces the `rd_validity` gate (measured-MSE < uniform
  at matched budget).
- Zero-regression verified at runtime: with `rd_alloc.enabled=false` / no
  override, the uniform real-kernel K_high per Linear == the ring-aligned 1/8
  split (`ZERO-REGRESSION uniform K_high==1/8 split: OK`).

## Reproduce

```bash
docker exec -d -w /home/luofan/plaquant -e CUDA_VISIBLE_DEVICES=1 \
  -e PYTHONPATH=/home/luofan/plaquant -e HF_HOME=/home/luofan/hf_cache \
  luofan-plaquant bash -c 'python3 promix/scripts/rd_perf.py \
  --config promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml --r1_off \
  --budgets 0.125,0.08333,0.0625,0.041667 --prefill_M 2048 \
  --decode_prompt 256 --decode_gen 48 --iters 15 --warmup 5 \
  --out /home/luofan/RD_PERF.json > /home/luofan/RD_PERF.log 2>&1'
```
