# ProMix → vLLM Integration — Phase 2 (CUDA graphs) + Phase 3 (three-way benchmark)

**Status: COMPLETE.** Graphs/compile work with `enforce_eager=False` (no vLLM
changes, no fallback needed); post-graphs T2 PPL gate GREEN on 3B **and** 8B;
the three-way same-strong-stack benchmark (ProMix-in-vLLM vs vLLM-NVFP4 vs
vLLM-BF16) is measured for 3B + 8B with the NVFP4/BF16 cross-check inside the
known regime. Raw JSONs: `docs/vllm_integ_phase23_data/`.

Builds on Phase 1 (`docs/vllm_integ_phase1.md`): plugin
`promix/vllm_integ/{plugin,kernels,shared_g,export}.py`, T0/T1 bit-exact,
eager T2 PPL |Δ|=0.0038.

---

## 1. Phase 2 — CUDA-graph / torch.compile compatibility

### What was built
| file | change |
|---|---|
| `promix/vllm_integ/ops.py` (new) | `torch.ops.vllm.promix_linear(x, handle)` via vLLM's `direct_register_custom_op` + fake impl. The WHOLE per-layer apply (down_proj Hadamard / o_proj reorder → `quant_pack_float` → fused dual-format GEMM) runs inside ONE opaque custom op, so Dynamo fullgraph tracing and piecewise/FULL CUDA-graph capture never see a pybind call. Layers are addressed by an int handle into a process-local registry filled in `process_weights_after_loading` (runs before vLLM compiles). |
| `promix/vllm_integ/plugin.py` | `apply()` calls the custom op (escape hatch `PROMIX_EAGER_APPLY=1` restores the Phase-1 direct path); registers the op at import. |
| `pyproject.toml` (new, repo root) | `vllm.general_plugins` entry point `promix_quant = promix.vllm_integ.plugin:register` → the method + op load in **spawned** workers (`uv pip install --no-deps -e .` into the vLLM venv). `vllm serve` / default multiprocessing now work; Phase-1's `VLLM_ENABLE_V1_MULTIPROCESSING=0` is no longer required. |
| `gates/gate_t2.py` | `--graphs` flag = the graph-equivalence re-gate (enforce_eager=False), writes `gate_t2_graphs.json`. |

### Result: graphs WORK — no blocker, no eager fallback
With the one custom op, vLLM 0.32 runs the full default pipeline on the promix
checkpoint: Dynamo fullgraph bytecode transform → inductor compile (AOT
artifacts cached/reloaded) → **51 PIECEWISE + 51 FULL CUDA graphs captured**
(FULL_AND_PIECEWISE), FlashInfer autotune included. No dynamic-shape issues
(each captured size replays the same kernel sequence; all branching lives
inside the op), no sync-in-capture (the exts were already
capture-safe: `cudaStreamIsCapturing`-guarded sync + cached workspaces, warmed
up by vLLM's pre-capture passes).

Why one coarse op instead of three fine ops (`quant_pack_float`,
`fused_gemm_prepacked`, Hadamard): identical capture behavior (the op body is
replayed inside the graph either way), zero tensor-schema churn for the
multi-output pack, and the fp16 Hadamard (fast_hadamard_transform, also a raw
pybind ext) is covered by the same wrapper for free. Inductor can't fuse into
an opaque region either way; the GEMM/pack kernels are already fused CUDA.

### Post-graphs T2 gate (full wikitext-2, 141 chunks / 288,627 positions)
| model | PPL vLLM-graphs | PPL shared-g ref | \|ΔPPL\| (gate ≤0.1) | top-1 |
|---|---|---|---|---|
| 3B | 8.3703 | 8.3674 | **0.0029 PASS** (eager was 0.0038) | 89.79 % |
| 8B | 6.6402 | 6.6377 | **0.0025 PASS** | 89.57 % |

Graphs do not change numerics beyond the eager-level noise. Top-1 stays in the
Phase-1 root-caused regime (~90 % = W4A4 activation quantization amplifying the
~0.17 % engine-glue top-1 difference; the unquantized glue control was
99.83 %) — unchanged conclusion, reported as measured.

8B notes: exported with the Phase-1 exporter + shared-g
(`promix/configs/llama-3-8b-fp-perhead.yaml`, GPTQ, `--no-perproj-ppl`).
**Shared-g real-kernel PPL = 6.6363** (fake-quant W4A4 baseline was 6.72).
Checkpoint 6.1 GiB vs 16 GiB bf16.

### Eager penalty (what Phase 2 bought)
promix 3B, graphs vs `--enforce-eager` (`promix_3b_eager.json`):
decode TPOT **6.1 → 24.6 ms (b1) / 7.2 → 24.9 ms (b16)** ≈ **3.5-4× decode
slowdown in eager**; prefill TTFT only +10-14 % (42.9 vs 38.7 ms b1, 540 vs
475 ms b16). Graphs were mandatory for any fair serving number.

---

## 2. Phase 3 — the three-way benchmark

Protocol = the committed `docs/baselines_scripts/bench_vllm.py` protocol,
extended (`promix/vllm_integ/bench_serve.py`): one vLLM 0.32 engine per config,
graphs ON, prefix caching OFF, identical-copy exact-length token prompts,
greedy, ignore_eos, 1 warmup + median of 3. Cases: prefill-heavy ISL/OSL
1024/128, decode-heavy 128/1024, batch {1,4,8,16}, plus TTFT-only L=2048 sweep.
TPOT = (full − TTFT)/(OSL−1); gen tok/s = B·OSL/full. GPU 7 (RTX Pro 6000
Blackwell), otherwise-idle box. Configs: **promix** = exported checkpoint
(fp16), **nvfp4** = llmcompressor NVFP4 W4A4 via compressed-tensors (bf16),
**bf16** = unsloth base model. Render tables:
`python -m promix.vllm_integ.harvest_three_way --dir docs/vllm_integ_phase23_data --size 3b|8b`.

### 3B (Llama-3.2-3B)

| case | TTFT pm/nv/bf16 (ms) | TPOT pm/nv/bf16 (ms) | gen tok/s pm/nv/bf16 | pm/bf16 | pm/nvfp4 | nv/bf16 |
|---|---|---|---|---|---|---|
| isl1024_osl128_b1 | 32.9 / 33.0 / 24.3 | 6.12 / 3.17 / 5.50 | 158 / 294 / 177 | 0.89x | 0.54x | 1.66x |
| isl1024_osl128_b4 | 80.5 / 64.0 / 80.3 | 6.42 / 3.19 / 5.99 | 571 / 1091 / 609 | 0.94x | 0.52x | 1.79x |
| isl1024_osl128_b8 | 130.0 / 86.7 / 148.6 | 6.75 / 3.52 / 6.11 | 1037 / 1919 / 1108 | 0.94x | 0.54x | 1.73x |
| isl1024_osl128_b16 | 239.1 / 151.2 / 290.6 | 7.22 / 4.22 / 6.85 | 1772 / 2979 / 1765 | 1.00x | 0.59x | 1.69x |
| isl128_osl1024_b1 | 8.8 / 7.0 / 9.9 | 5.93 / 3.16 / 5.45 | 169 / 316 / 183 | 0.92x | 0.53x | 1.72x |
| isl128_osl1024_b4 | 16.8 / 13.4 / 19.2 | 6.07 / 3.07 / 5.87 | 658 / 1300 / 680 | 0.97x | 0.51x | 1.91x |
| isl128_osl1024_b8 | 33.3 / 38.9 / 30.4 | 6.24 / 3.26 / 5.82 | 1278 / 2425 / 1369 | 0.93x | 0.53x | 1.77x |
| isl128_osl1024_b16 | 42.8 / 41.7 / 46.3 | 6.59 / 3.68 / 6.30 | 2414 / 4299 / 2525 | 0.96x | 0.56x | 1.70x |

| prefill TTFT L2048 | promix | nvfp4 | bf16 | pm/bf16 | pm/nvfp4 | nv/bf16 |
|---|---|---|---|---|---|---|
| b1 | 38.7 ms | 32.9 ms | 45.4 ms | 1.18x | 0.85x | 1.38x |
| b4 | 125.7 ms | 82.2 ms | 160.1 ms | 1.27x | 0.65x | 1.95x |
| b8 | 243.2 ms | 150.5 ms | 310.9 ms | 1.28x | 0.62x | 2.07x |
| b16 | 474.7 ms | 278.9 ms | 616.9 ms | **1.30x** | 0.59x | 2.21x |

### 8B (Llama-3-8B)

| case | TTFT pm/nv/bf16 (ms) | TPOT pm/nv/bf16 (ms) | gen tok/s pm/nv/bf16 | pm/bf16 | pm/nvfp4 | nv/bf16 |
|---|---|---|---|---|---|---|
| isl1024_osl128_b1 | 43.1 / 37.1 / 49.4 | 10.17 / 5.17 / 11.37 | 96 / 184 / 86 | 1.12x | 0.52x | 2.15x |
| isl1024_osl128_b4 | 133.8 / 84.6 / 176.2 | 10.11 / 5.68 / 11.46 | 361 / 635 / 314 | 1.15x | 0.57x | 2.02x |
| isl1024_osl128_b8 | 241.5 / 142.8 / 338.5 | 10.33 / 5.98 / 11.85 | 659 / 1134 / 555 | 1.19x | 0.58x | 2.04x |
| isl1024_osl128_b16 | 462.2 / 266.6 / 675.7 | 11.09 / 6.90 / 12.80 | 1095 / 1793 / 890 | **1.23x** | 0.61x | 2.01x |
| isl128_osl1024_b1 | 13.0 / 8.6 / 17.0 | 9.72 / 5.13 / 11.30 | 103 / 195 / 88 | 1.16x | 0.53x | 2.20x |
| isl128_osl1024_b4 | 26.8 / 18.5 / 35.6 | 9.88 / 5.50 / 11.27 | 404 / 726 / 354 | 1.14x | 0.56x | 2.05x |
| isl128_osl1024_b8 | 52.7 / 44.2 / 56.7 | 10.03 / 5.69 / 11.57 | 795 / 1398 / 689 | 1.15x | 0.57x | 2.03x |
| isl128_osl1024_b16 | 73.4 / 45.6 / 93.8 | 10.45 / 6.29 / 12.19 | 1523 / 2527 / 1304 | 1.17x | 0.60x | 1.94x |

| prefill TTFT L2048 | promix | nvfp4 | bf16 | pm/bf16 | pm/nvfp4 | nv/bf16 |
|---|---|---|---|---|---|---|
| b1 | 67.5 ms | 36.8 ms | 91.7 ms | 1.36x | 0.55x | 2.49x |
| b4 | 243.7 ms | 133.9 ms | 354.1 ms | 1.45x | 0.55x | 2.64x |
| b8 | 473.7 ms | 264.5 ms | 699.6 ms | 1.48x | 0.56x | 2.65x |
| b16 | 932.5 ms | 518.7 ms | 1393.7 ms | **1.49x** | 0.56x | 2.69x |

### Cross-check vs the known regime → PASS
vLLM-NVFP4/BF16 prefill TTFT: 3B 1.38x(b1)→2.21x(b16) vs the committed
1.35→2.14x; 8B 2.49→2.69x vs 2.16→2.87x. Decode tok/s ratios (1.7-2.2x) match
the committed decode baselines (3B b8: 2343/1432 = 1.64x then, 1.73-1.77x now).
The ProMix rows sit on a reproduced stack.

### Where ProMix lands (headline)
* **ProMix vs BF16** — prefill TTFT **1.18-1.30x (3B) / 1.36-1.49x (8B)**
  faster at every batch; end-to-end serving throughput 0.89-1.00x (3B) /
  **1.12-1.23x (8B)**. At 8B ProMix beats BF16 across the board (decode TPOT
  10.2-11.1 vs 11.3-12.8 ms); at 3B decode is a wash (quant tax ≈ bandwidth
  saving on a 3B-sized model — consistent with the earlier 1B/3B e2e findings).
* **ProMix vs vLLM-NVFP4** — **0.51-0.61x** throughput (NVFP4 ~1.7-1.9x
  faster), prefill TTFT 0.55-0.85x. ProMix does NOT beat the pure-NVFP4
  production path; its value is the accuracy side (ResQ rotation + MXFP8 ring:
  real-kernel 8B PPL 6.64 / 3B 8.36 at W4A4) at a 1.2-1.5x-over-BF16 serving
  cost point.

### Honest overhead notes (why ProMix trails pure NVFP4)
1. **Per-layer extra work vLLM-NVFP4 doesn't do:** activation quant+pack as a
   separate kernel per linear (vLLM fuses fp4 quant into RMSNorm/act epilogues
   via its `fuse_norm_quant`/`fuse_act_quant` inductor passes, and we disable
   none of that for the baselines), down_proj online fp16 Hadamard, o_proj
   column gather, bf16→fp16 epilogue cast + torch-level glue inside the op.
2. **Dual-format mainloop:** the MXFP8 ring (r=1/8 of K) runs at half the FP4
   TC rate and adds a phase; pure NVFP4 reads 4 bits/weight everywhere
   (checkpoints: 8B promix 6.1 GiB vs nvfp4 5.7 GiB; 3B 3.2 vs 2.3 GiB — the
   fp16 lm_head/embeddings and MXFP8 ring cost capacity too).
3. **Decode path:** the plugin routes every M through the prefill-oriented
   128×128 fused GEMM (+2 pack launches); at decode M=1-16 that is massively
   under-utilized, while vLLM-NVFP4 uses tuned FlashInfer fp4 GEMMs. The
   promix decode GEMV (`decode_gemv_ext`, M=1 W4A16, 1.3x over fp16 at kernel
   level) is NOT wired into the plugin (needs decode-layout weight buffers in
   the exporter) — the single biggest known lever for the decode rows.
4. **Small-M prefill dispatch:** chunked-prefill slices arrive at vLLM capture
   sizes; the per-shape `config_dispatch` (b1_tma for qkv·2048/small-M) is not
   wired in the plugin — it always runs the co-design 128×128 path.
5. fp16 (ours) vs bf16 (baselines): same width, negligible.

### Reproduce
```bash
# Phase 2 setup (once, inside the container):
docker exec -w /home/luofan/plaquant-vllm-integ luofan-plaquant \
  uv pip install --python /home/luofan/baselines/venv/bin/python --no-deps --no-build-isolation -e .
# post-graphs T2 re-gate (3B / 8B):
/home/luofan/baselines/venv/bin/python -m promix.vllm_integ.gates.gate_t2 \
  --ckpt /home/luofan/vllm_integ/promix-llama-3.2-3b --graphs
# 8B export (container base env, GPU 6; needs rotation/*llama-3-8b* artifacts):
MASTER_PORT=29517 python -m promix.vllm_integ.export \
  --config promix/configs/llama-3-8b-fp-perhead.yaml --no-perproj-ppl \
  --out /home/luofan/vllm_integ/promix-llama-3-8b
# three-way bench (one line per config; see bench/run_{3b,8b}.sh on the remote):
/home/luofan/baselines/venv/bin/python -m promix.vllm_integ.bench_serve \
  --model <ckpt> --tag <tag> --dtype float16|auto|bfloat16 --out <json>
```

Measurement hygiene: the first 3B pass ran concurrently with the 8B export and
showed bimodal b1 TTFTs; all tables above are from re-runs on an
otherwise-idle box (raw contended run kept on the remote under
`bench/contended_3b/`, not committed).

### Follow-ups (ranked)
1. Wire the decode GEMV + per-shape dispatch into the plugin (decode rows).
2. Fuse activation quant+pack into the producer (RMSNorm epilogue) as vLLM
   does for its fp4 path — removes the biggest per-layer tax.
3. TP>1 sharding of packed buffers + per-tensor g (plugin asserts TP=1).
4. lm_head W8 (PPL-safe per the 1B study) to close the capacity gap.
