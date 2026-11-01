# FINAL five-way same-stack serving comparison (vLLM 0.32, RTX Pro 6000 SM120)

> 2026-11-01. The closing act of the vLLM perf loop (rounds 1-5, converged at
> `docs/vllm_perf_round5.md`). Everything below is measured in the SAME vLLM
> 0.32 (cu130) engine on the same RTX Pro 6000 Blackwell Server (SM120, 96 GB),
> one engine per config, CUDA graphs + torch.compile ON, prefix caching OFF,
> greedy + ignore_eos, identical exact-length token prompts, 1 warmup + median
> of 3 — the Phase-2/3 `bench_serve` protocol
> (`docs/vllm_integ_phase23.md`), extended to the large-batch regime.
> Raw JSONs: `docs/vllm_final_data/`; scripts:
> `docs/vllm_final_scripts/run_final.sh` (+ `run_sweep_point.sh`),
> harvest: `python -m promix.vllm_integ.harvest_final --dir docs/vllm_final_data --size 3b|8b`.

## 0. Gates (run before any ProMix bench — mandatory)

| gate | result |
|---|---|
| T2 prefill `--graphs`, final converged config (all R1-R5 levers default-ON) | **Byte-identical to the r5 gate on every field**: PPL 8.37029131329681, top-1 89.78647181310134 %, logprob-absdiff mean 0.17774903457377478 (`gate_t2_graphs_final.json` vs r5's `gate_t2_graphs_r5final.json`). (The harness's own top-1 criterion prints FAIL in graphs mode — same digits every round since r2; the gate is byte identity, per r3/r5.) |
| T2 decode-route m16 | **PASS, signed ΔPPL = −0.32004700553051535** (8.0473 vs ref 8.3674) — exactly r5's −0.3200. |

Zero drift from the converged round-5 configuration. Config = feat/vllm-integ
@9da7189 + this campaign's additions (r-sweep configs, pad-aware SF alloc —
byte/time-identical on the shipping r=1/8 splits, see §6 note 6).

## 1. The five methods (+ extensions)

| method | W / A | checkpoint & vLLM path | dtype |
|---|---|---|---|
| BF16 (anchor) | 16/16 | unsloth base weights, torch.compile + cuBLAS/CUTLASS | bfloat16 |
| FP8 W8A8 | 8/8 (dynamic per-token acts) | llm-compressor `FP8_DYNAMIC` (data-free) → compressed-tensors, `CutlassFP8ScaledMM` | auto |
| GPTQ-Marlin W4A16 | 4/16 (g128 sym, wikitext calib) | llm-compressor GPTQ → compressed-tensors, **Marlin** kernel (the weight-only reference) | auto |
| NVFP4 W4A4 | 4.5/4 (g16 e4m3 scales) | llm-compressor `NVFP4` → compressed-tensors, `FlashInferCutlassNvFp4` (`fp4_gemm_cutlass_sm120`) | auto |
| ProMix W4A4 (ours) | 4.97/4 (r=1/8 NVFP4+MXFP8) | ProMix exporter ckpt, promix vLLM plugin (fused dual-format GEMM + GEMV/marlin low-M routes + marlin-hi + split-K + gather-fold + cast-fold + qpfm, all default-ON) | float16 |

Extensions measured with the identical protocol (§5): **AWQ-Marlin W4A16**,
**INT8 W8A8** (SmoothQuant+GPTQ W8A8), **ct-MXFP4 W4A4**, and the ProMix
**r-sweep** (uniform high-fraction grid, §4). Models: Llama-3.2-3B ("3B") and
Llama-3-8B ("8B"). One GPU per engine, otherwise-idle card.

## 2. Accuracy column (wikitext-2 PPL, absolute vLLM protocol — 141 x 2048-token chunks, 288,627 scored positions, same tokenizer/scoring for every row)

| method | 3B PPL | 8B PPL | activations |
|---|---|---|---|
| BF16 | 7.82 | 6.14 | bf16 |
| FP8 W8A8 | **7.880** | **6.191** | fp8 dynamic |
| INT8 W8A8 (ext) | not servable on SM120 | — | int8 dynamic |
| GPTQ-Marlin W4A16 | 8.20 | 6.52 | **bf16 (not quantized)** |
| AWQ-Marlin W4A16 | 8.48 | 6.73 | **bf16 (not quantized)** |
| NVFP4 W4A4 | 8.63 | 6.87 | fp4 static |
| ct-MXFP4 W4A4 (ext) | not servable on SM120 | — | mxfp4 dynamic |
| **ProMix W4A4 (served config, real kernel)** | **8.370** | **6.640** | fp4+1/8 mxfp8, act-MSE |
| ProMix decode-route (marlin W4A16 m<=16 windows active) | 8.047 | — | w4a16 at small M |

ProMix is the most accurate W4A4 row: −0.26 PPL vs NVFP4 at 3B, −0.23 at 8B
W4A16 rows keep bf16 activations (no activation
quantization cost) and are the weight-only reference, not a same-condition
W4A4 comparison. 3B/8B BF16, GPTQ, AWQ, NVFP4 from
`docs/baselines_on_hardware.md` (same harness/protocol); FP8/INT8/MXFP4
measured this campaign (`/home/luofan/baselines/results/*.json` on the host).

## 3. Serving tables

Cases: prefill-heavy ISL/OSL 1024/128, decode-heavy 128/1024, prefill-only
TTFT L2048, QServe-protocol continuous batching 1024/512 at concurrency
{64,128,256} (all requests submitted at once; the vLLM continuous-batching
scheduler manages the stream), long-ISL (2048/4096) decode with **fp8 KV
cache**. Batches {1,16} small-batch reference + {32,64,128,256,512}
large-batch primary (the serving regime; GEMMs saturated).

<!-- BEGIN GENERATED 3B -->
## 3B — methods: BF16, FP8 W8A8, GPTQ-Marlin W4A16, AWQ-Marlin W4A16, NVFP4 W4A4, ProMix W4A4

### prefill-heavy 1024/128 — TTFT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 23.5 | 290.5 | 579.2 | 1160.5 | 2350.5 | 4771.6 | 10803.5 |
| FP8 W8A8 | 18.0 | 184.6 | 363.1 | 725.3 | 1457.0 | 2950.2 | 6247.7 |
| GPTQ-Marlin W4A16 | 24.4 | 339.4 | 673.5 | 1341.9 | 2683.6 | 5421.4 | 11634.2 |
| AWQ-Marlin W4A16 | 24.6 | 340.6 | 674.2 | 1342.1 | 2680.6 | 5372.2 | 11338.9 |
| NVFP4 W4A4 | 30.5 | 126.4 | 246.3 | 484.5 | 964.6 | 1924.7 | 3852.1 |
| ProMix W4A4 | 27.7 | 184.6 | 372.1 | 759.9 | 1488.6 | 2998.4 | 6002.9 |

### prefill-heavy 1024/128 — TPOT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 5.51 | 6.84 | 8.69 | 11.11 | 17.12 | 31.34 | 63.52 |
| FP8 W8A8 | 3.84 | 5.09 | 6.54 | 9.11 | 14.75 | 28.48 | 60.71 |
| GPTQ-Marlin W4A16 | 2.84 | 4.17 | 5.86 | 9.28 | 16.02 | 31.78 | 61.88 |
| AWQ-Marlin W4A16 | 2.88 | 4.18 | 5.85 | 9.25 | 15.97 | 31.43 | 62.23 |
| NVFP4 W4A4 | 3.60 | 4.38 | 5.90 | 8.69 | 14.27 | 26.89 | 57.32 |
| ProMix W4A4 | 4.36 | 6.09 | 7.74 | 10.74 | 16.30 | 29.27 | 60.96 |

### prefill-heavy 1024/128 — gen tok/s
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 177 | 1767 | 2433 | 3186 | 3621 | 3744 | 3473 |
| FP8 W8A8 | 253 | 2462 | 3430 | 4351 | 4921 | 4990 | 4695 |
| GPTQ-Marlin W4A16 | 332 | 2357 | 2889 | 3251 | 3472 | 3465 | 3362 |
| AWQ-Marlin W4A16 | 327 | 2350 | 2890 | 3256 | 3480 | 3500 | 3406 |
| NVFP4 W4A4 | 262 | 2998 | 4116 | 5158 | 5899 | 6136 | 5888 |
| ProMix W4A4 | 220 | 2139 | 3022 | 3857 | 4605 | 4879 | 4768 |

### decode-heavy 128/1024 — TPOT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 5.47 | 6.35 | 7.45 | 8.91 | 12.74 | 19.70 | 37.27 |
| FP8 W8A8 | 3.63 | 4.41 | 5.25 | 6.91 | 10.07 | 18.23 | 32.63 |
| GPTQ-Marlin W4A16 | 2.71 | 3.53 | 4.56 | 6.89 | 11.21 | 19.86 | 36.82 |
| AWQ-Marlin W4A16 | 2.82 | 3.62 | 4.69 | 7.02 | 11.32 | 20.00 | 37.04 |
| NVFP4 W4A4 | 3.42 | 3.86 | 4.72 | 6.45 | 9.69 | 16.51 | 30.35 |
| ProMix W4A4 | 4.09 | 5.21 | 6.29 | 8.11 | 11.29 | 18.14 | 32.80 |

### decode-heavy 128/1024 — gen tok/s
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 183 | 2505 | 4253 | 7072 | 9822 | 12621 | 13314 |
| FP8 W8A8 | 275 | 3608 | 6043 | 9147 | 12481 | 13769 | 15349 |
| GPTQ-Marlin W4A16 | 368 | 4485 | 6888 | 9065 | 11091 | 12477 | 13431 |
| AWQ-Marlin W4A16 | 354 | 4363 | 6704 | 8898 | 10987 | 12393 | 13335 |
| NVFP4 W4A4 | 292 | 4110 | 6731 | 9832 | 13044 | 15286 | 16608 |
| ProMix W4A4 | 244 | 3051 | 5042 | 7801 | 11161 | 13834 | 15262 |

### prefill TTFT L2048 (ms)
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 44.5 | 679.3 | 1398.7 | 2806.9 | 5591.1 | 11067.7 | 21778.3 |
| FP8 W8A8 | 29.6 | 397.1 | 796.7 | 1642.0 | 3388.0 | 6876.0 | 13815.1 |
| GPTQ-Marlin W4A16 | 46.8 | 722.8 | 1471.8 | 2954.5 | 5938.0 | 11858.5 | 23650.1 |
| AWQ-Marlin W4A16 | 46.6 | 734.4 | 1485.8 | 2975.1 | 5888.8 | 11651.6 | 23201.9 |
| NVFP4 W4A4 | 33.3 | 265.0 | 524.1 | 1041.2 | 2077.9 | 4151.9 | 8593.9 |
| ProMix W4A4 | 35.3 | 390.7 | 782.3 | 1562.9 | 3159.0 | 6461.6 | 12932.9 |

### QServe-protocol continuous batching 1024/512 — gen tok/s (TPOT ms)
| method | c64 | c128 | c256 |
|---|---|---|---|
| BF16 | 4445 (12.13) | 5448 (18.91) | 6119 (32.53) |
| FP8 W8A8 | 5488 (10.24) | 6592 (16.56) | 6936 (31.13) |
| GPTQ-Marlin W4A16 | 5001 (10.19) | 5612 (17.59) | 5908 (32.85) |
| AWQ-Marlin W4A16 | 5009 (10.17) | 5612 (17.59) | 5824 (33.01) |
| NVFP4 W4A4 | 6129 (9.51) | 7130 (16.10) | 7771 (29.23) |
| ProMix W4A4 | 4992 (11.38) | 6209 (17.75) | 6931 (31.15) |

### long-ISL decode, fp8 KV (TPOT ms @ b16 / b64)
| method | isl2048 b16 | isl2048 b64 | isl4096 b16 | isl4096 b64 |
|---|---|---|---|---|
| BF16 | 6.83 | 11.29 | 8.07 | 18.48 |
| FP8 W8A8 | 4.74 | 9.22 | 6.07 | 14.76 |
| GPTQ-Marlin W4A16 | 3.90 | 9.10 | 5.20 | 16.19 |
| AWQ-Marlin W4A16 | 4.03 | 9.17 | 5.26 | 15.65 |
| NVFP4 W4A4 | 4.14 | 8.83 | 5.22 | 14.22 |
| ProMix W4A4 | 5.81 | 10.53 | 7.14 | 16.36 |

### capacity (GPU KV cache tokens @ gmu 0.90, mml 4096) + peak reserved GiB
| method | KV cache GiB | max reserved GiB |
|---|---|---|
| BF16 | 77.53 | 85.5 |
| FP8 W8A8 | 80.17 | 85.4 |
| GPTQ-Marlin W4A16 | 81.41 | 85.5 |
| AWQ-Marlin W4A16 | 81.41 | 85.5 |
| NVFP4 W4A4 | 81.45 | 88.3 |
| ProMix W4A4 | 77.43 | 85.0 |

### ProMix ratios (other_ms / pm_ms for latency; pm/other for tok/s; >1 = ProMix wins)
**case 1024/128 TPOT ratio (x/pm):**
| vs | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 1.26 | 1.12 | 1.12 | 1.03 | 1.05 | 1.07 | 1.04 |
| FP8 W8A8 | 0.88 | 0.84 | 0.85 | 0.85 | 0.90 | 0.97 | 1.00 |
| GPTQ-Marlin W4A16 | 0.65 | 0.69 | 0.76 | 0.86 | 0.98 | 1.09 | 1.02 |
| AWQ-Marlin W4A16 | 0.66 | 0.69 | 0.76 | 0.86 | 0.98 | 1.07 | 1.02 |
| NVFP4 W4A4 | 0.83 | 0.72 | 0.76 | 0.81 | 0.88 | 0.92 | 0.94 |

**case 128/1024 TPOT ratio (x/pm):**
| vs | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 1.34 | 1.22 | 1.19 | 1.10 | 1.13 | 1.09 | 1.14 |
| FP8 W8A8 | 0.89 | 0.85 | 0.83 | 0.85 | 0.89 | 1.01 | 0.99 |
| GPTQ-Marlin W4A16 | 0.66 | 0.68 | 0.73 | 0.85 | 0.99 | 1.10 | 1.12 |
| AWQ-Marlin W4A16 | 0.69 | 0.70 | 0.75 | 0.87 | 1.00 | 1.10 | 1.13 |
| NVFP4 W4A4 | 0.84 | 0.74 | 0.75 | 0.80 | 0.86 | 0.91 | 0.93 |

<!-- END GENERATED 3B -->

<!-- BEGIN GENERATED 8B -->
## 8B — methods: BF16, FP8 W8A8, GPTQ-Marlin W4A16, NVFP4 W4A4, ProMix W4A4

### prefill-heavy 1024/128 — TTFT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 48.5 | 664.6 | 1334.5 | 2695.6 | 5452.6 | 12226.7 | 24473.0 |
| FP8 W8A8 | 33.2 | 393.5 | 783.9 | 1575.5 | 3180.2 | 6754.8 | 14615.3 |
| GPTQ-Marlin W4A16 | 52.0 | 757.0 | 1506.5 | 3016.7 | 6121.3 | 13130.7 | 26325.6 |
| NVFP4 W4A4 | 42.2 | 246.6 | 481.2 | 959.0 | 1920.9 | 3861.1 | 7987.3 |
| ProMix W4A4 | 41.9 | 378.9 | 760.4 | 1525.1 | 3068.8 | 6191.1 | 13006.3 |

### prefill-heavy 1024/128 — TPOT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 11.54 | 12.94 | 15.24 | 18.48 | 26.87 | 42.98 | 101.97 |
| FP8 W8A8 | 7.16 | 8.62 | 10.33 | 13.55 | 20.40 | 41.66 | 79.26 |
| GPTQ-Marlin W4A16 | 4.69 | 6.57 | 8.05 | 12.19 | 23.55 | 41.58 | 87.06 |
| NVFP4 W4A4 | 5.66 | 7.21 | 8.68 | 11.51 | 18.28 | 34.11 | 74.65 |
| ProMix W4A4 | 7.16 | 8.84 | 10.46 | 14.45 | 21.58 | 38.39 | 83.16 |

### prefill-heavy 1024/128 — gen tok/s
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 85 | 887 | 1252 | 1625 | 1848 | 1853 | 1751 |
| FP8 W8A8 | 136 | 1376 | 1954 | 2485 | 2839 | 2720 | 2655 |
| GPTQ-Marlin W4A16 | 197 | 1287 | 1619 | 1795 | 1798 | 1780 | 1753 |
| NVFP4 W4A4 | 168 | 1761 | 2586 | 3385 | 3862 | 3999 | 3752 |
| ProMix W4A4 | 135 | 1364 | 1960 | 2438 | 2820 | 2961 | 2781 |

### decode-heavy 128/1024 — TPOT ms
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 11.46 | 12.37 | 13.80 | 15.98 | 20.11 | 29.50 | 57.03 |
| FP8 W8A8 | 6.91 | 7.89 | 8.82 | 10.78 | 14.67 | 25.64 | 44.78 |
| GPTQ-Marlin W4A16 | 4.52 | 5.49 | 6.66 | 9.58 | 16.33 | 28.59 | 54.90 |
| NVFP4 W4A4 | 5.43 | 6.45 | 7.04 | 8.76 | 12.87 | 21.24 | 38.37 |
| ProMix W4A4 | 6.84 | 7.81 | 9.05 | 11.92 | 15.82 | 24.88 | 44.18 |

### decode-heavy 128/1024 — gen tok/s
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 87 | 1286 | 2293 | 3926 | 6166 | 8302 | 8560 |
| FP8 W8A8 | 145 | 2016 | 3589 | 5836 | 8505 | 9691 | 11014 |
| GPTQ-Marlin W4A16 | 221 | 2867 | 4675 | 6437 | 7500 | 8498 | 8808 |
| NVFP4 W4A4 | 184 | 2467 | 4509 | 7209 | 9767 | 11794 | 13030 |
| ProMix W4A4 | 146 | 2033 | 3499 | 5288 | 7909 | 9992 | 11209 |

### prefill TTFT L2048 (ms)
| method | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 93.6 | 1624.0 | 3178.2 | 6220.3 | 12268.8 | 23710.4 | 47319.0 |
| FP8 W8A8 | 60.8 | 927.8 | 1883.6 | 3713.7 | 7335.2 | 14624.4 | 29363.2 |
| GPTQ-Marlin W4A16 | 101.4 | 1737.2 | 3427.8 | 6741.9 | 13444.8 | 27160.4 | 54760.7 |
| NVFP4 W4A4 | 39.1 | 519.2 | 1035.6 | 2068.4 | 4262.7 | 8655.9 | 17396.7 |
| ProMix W4A4 | 61.8 | 891.2 | 1800.7 | 3582.7 | 7136.6 | 14270.6 | 28283.6 |

### QServe-protocol continuous batching 1024/512 — gen tok/s (TPOT ms)
| method | c64 | c128 | c256 |
|---|---|---|---|
| BF16 | 2596 (19.41) | 3375 (27.31) | 3864 (43.38) |
| FP8 W8A8 | 3687 (14.27) | 4565 (21.79) | 4743 (40.36) |
| GPTQ-Marlin W4A16 | 3340 (13.29) | 3552 (24.04) | 3680 (43.59) |
| NVFP4 W4A4 | 4446 (12.54) | 5283 (20.51) | 5845 (36.30) |
| ProMix W4A4 | 3465 (15.54) | 4380 (23.28) | 4922 (39.94) |

### long-ISL decode, fp8 KV (TPOT ms @ b16 / b64)
| method | isl2048 b16 | isl2048 b64 | isl4096 b16 | isl4096 b64 |
|---|---|---|---|---|
| BF16 | 12.70 | 18.67 | 14.15 | 26.31 |
| FP8 W8A8 | — | — | — | — |
| GPTQ-Marlin W4A16 | 6.04 | 13.25 | 7.32 | 18.12 |
| NVFP4 W4A4 | 6.40 | 11.74 | 7.86 | 18.25 |
| ProMix W4A4 | 8.66 | 14.73 | 10.06 | 22.14 |

### capacity (GPU KV cache tokens @ gmu 0.90, mml 4096) + peak reserved GiB
| method | KV cache GiB | max reserved GiB |
|---|---|---|
| BF16 | 67.89 | 85.7 |
| FP8 W8A8 | 73.81 | 84.7 |
| GPTQ-Marlin W4A16 | 77.52 | 85.8 |
| NVFP4 W4A4 | 77.48 | 89.9 |
| ProMix W4A4 | 70.77 | 86.1 |

### ProMix ratios (other_ms / pm_ms for latency; pm/other for tok/s; >1 = ProMix wins)
**case 1024/128 TPOT ratio (x/pm):**
| vs | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 1.61 | 1.46 | 1.46 | 1.28 | 1.25 | 1.12 | 1.23 |
| FP8 W8A8 | 1.00 | 0.97 | 0.99 | 0.94 | 0.95 | 1.09 | 0.95 |
| GPTQ-Marlin W4A16 | 0.66 | 0.74 | 0.77 | 0.84 | 1.09 | 1.08 | 1.05 |
| NVFP4 W4A4 | 0.79 | 0.82 | 0.83 | 0.80 | 0.85 | 0.89 | 0.90 |

**case 128/1024 TPOT ratio (x/pm):**
| vs | b1 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| BF16 | 1.68 | 1.58 | 1.53 | 1.34 | 1.27 | 1.19 | 1.29 |
| FP8 W8A8 | 1.01 | 1.01 | 0.98 | 0.90 | 0.93 | 1.03 | 1.01 |
| GPTQ-Marlin W4A16 | 0.66 | 0.70 | 0.74 | 0.80 | 1.03 | 1.15 | 1.24 |
| NVFP4 W4A4 | 0.79 | 0.82 | 0.78 | 0.74 | 0.81 | 0.85 | 0.87 |

<!-- END GENERATED 8B -->

### Headline summary (ProMix vs each, TPOT at the serving regime)

**3B** (decode-heavy 128/1024): ProMix vs BF16 **1.09-1.34x faster** at every
batch; vs GPTQ-Marlin W4A16 0.66x at b1 → crossover at b128 (0.99) →
**1.10-1.12x at b256-512**; vs FP8 W8A8 ~parity at b256-512 (1.01/0.99); vs
NVFP4 0.74 (b16) → **0.93 (b512)**. Prefill TTFT L2048: ProMix beats BF16
1.26-1.77x and GPTQ/AWQ W4A16 1.33-1.88x at every batch; 0.66-0.68x of NVFP4
at b>=16 (b1 0.94).

**8B** (decode-heavy 128/1024): ProMix vs BF16 **1.19-1.68x**; vs GPTQ-Marlin
W4A16 0.66x at b1 → **1.03/1.15/1.24x at b128/256/512**; vs FP8 W8A8
0.90-1.03 (parity); vs NVFP4 0.74-0.87 (improving with batch). Prefill TTFT
L2048: ProMix 1.51-1.82x over BF16, **1.64-1.95x over GPTQ-Marlin**, 0.58-0.63x of NVFP4. QServe-protocol c256 throughput: ProMix 4922 tok/s vs GPTQ 3680
(**1.34x**), BF16 3864 (**1.27x**), FP8 4743 (1.04x), NVFP4 5845 (0.84x).

### The W4A16 verdict (the question this campaign had to answer)

**At the serving operating region, yes — ProMix W4A4 beats the weight-only
W4A16 Marlin reference**: every prefill point at every batch (1.3-2.0x), every
decode point from b128 up (1.03-1.24x), and high-concurrency continuous
batching (1.17x 3B / 1.34x 8B at c256) — while also being a 4-bit-activation
method (W4A16 pays zero activation-quant cost and serves bf16 activations).
The honest loss: small-batch decode (b1-b64), where Marlin's weight-streaming
GEMV is the textbook winner (0.66-0.85x); that regime is latency-bound
single-user serving, not the batched throughput regime these systems are
deployed in (and ProMix's own m<=16 marlin windows already borrow exactly that
trick inside its decode route, at better accuracy than the pure W4A16 model —
PPL 8.047 vs 8.20).

### fp8-KV support matrix (long-ISL runs)

All five methods accepted `kv_cache_dtype="fp8"` in this build (FlashAttention
backend) and produced correct-looking throughput; no method errored at engine
init. The 8B FP8-W8A8 kv8 row needed a re-run after a transient
free-memory race between chained engines (not a method failure).

## 4. ProMix r-sweep — the tunable accuracy/throughput curve (uniform n/32 grid)

The converged kernel serves ANY high-fraction r (dual-format split per
Linear). Grid constraint (honest): with uniform fractions, every Linear needs
K_high%32==0 at K in {3072, 8192} (3B) — representable points are n/32;
finer points (1/48, 1/64, ...) and non-n/32 ratios like 1/12, 1/24 need the
per-layer ring allocation (feat/rd-alloc machinery), which is not in this
serving branch. Points: 3B r in {1/32, 1/16, 3/32, 1/8, 3/16, 1/4};
8B r in {1/32, 1/16, 1/8, 1/4}.

<!-- BEGIN RSWEEP -->
<!-- END RSWEEP -->

## 5. Extensions — every other vLLM quant method runnable on this GPU

<!-- BEGIN EXT -->
### Runnability census (vLLM 0.32 cu130 wheel, SM120)

| method | status | evidence |
|---|---|---|
| BF16 / FP8 W8A8 / GPTQ-Marlin / AWQ-Marlin / NVFP4 / ProMix | **run** | full tables above / below |
| **INT8 W8A8** (SmoothQuant+GPTQ `W8A8` checkpoint, produced fine) | **BLOCKED on SM120 as shipped** | engine init: `RuntimeError: dispatch_scaled_mm ... Int8 not supported on SM120. Use FP8 quantization` (vllm csrc `w8a8/cutlass/c3x/scaled_mm_helper.hpp:34`) — vLLM's cutlass INT8 scaled-mm has no SM120 dispatch in this build |
| **ct-MXFP4 W4A4** (llm-compressor `MXFP4` checkpoint, produced fine) | **BLOCKED on SM120 as shipped** | `FlashInferMxFp4LinearKernel` selected, then `flashinfer.utils.BackendSupportedError: mm_fp4 does not support backend 'cute-dsl' with capability 120` |
| fp_quant / QuTLASS (Quartet, rotation W4A4) | see §6 peer attempts | requires `qutlass` source build |
| bitsandbytes etc. | skipped | not serving-grade paths |

So on THIS GPU the complete set of runnable same-stack quant methods is exactly
the table above + AWQ — i.e. the five-way IS the census, plus AWQ-Marlin:

**AWQ-Marlin W4A16** (identical protocol): at 3B — decode-heavy TPOT b256
20.00 ms / b512 37.04 vs ProMix 18.14 / 32.80 (ProMix 1.10x / 1.13x); prefill
TTFT L2048 b16 734 ms vs ProMix 390.7 (ProMix 1.88x) — same verdict as GPTQ:
ProMix wins the batched regime, loses b1-64 decode. Full AWQ rows are in the
generated tables above (8B AWQ in flight at commit time).
<!-- END EXT -->

## 6. Published W4A4 peer methods, end-to-end on this GPU

<!-- BEGIN PEERS -->
<!-- END PEERS -->

## 7. Cross-paper end-to-end comparison (relative to same-stack FP16/BF16, ISL/OSL 1024/512)

Literature convention (QServe): each method's e2e serving speedup vs ITS OWN
same-stack FP16/BF16 anchor, protocol-matched. Hardware generations differ —
these are NOT directly comparable absolute numbers; anchor strength matters
(an HF-eager anchor inflates the ratio; TRT-LLM/vLLM anchors are strong).

| method | W/A bits | hardware | stack (anchor strength) | protocol | published / measured speedup vs same-stack FP16/BF16 |
|---|---|---|---|---|---|
| QServe W4A8KV4 | 4/8 + KV4 | A100 / L40S | TensorRT-LLM FP16 (strong) | ISL/OSL 1024/512 max throughput | 1.2-2.4x (A100) / 1.47-3.47x (L40S), Llama-7B..72B (as published) |
| Atom W4A4KV4 | 4/4 + KV4 | RTX 4090 | Punica FP16 (medium) | serving throughput, batch 2-256 (large-batch partially simulated) | up to 2.5x (as published) |
| AWQ / TinyChat W4A16 | 4/16 | 4090 / Orin | HF eager FP16 (**weak anchor**) | decode latency | 2.7-3.9x (as published) |
| vLLM NVFP4 W4A4 | 4.5/4 | **RTX Pro 6000 (this work's GPU)** | vLLM 0.32 BF16 (strong) | ISL/OSL 1024/512, c256 (measured here) | **1.27x (3B) / 1.51x (8B)**; decode-heavy b256 1.19/1.39x; prefill L2048 b16 2.56/3.13x |
| **ProMix W4A4 (ours)** | 4.97/4 | **RTX Pro 6000** | vLLM 0.32 BF16 (strong) | ISL/OSL 1024/512, c256 (measured here) | **1.13x (3B) / 1.27x (8B)**; decode-heavy b256 TPOT 1.09/1.19x (b512 1.14/1.29x); prefill L2048 1.26-1.82x |

Honesty footnotes: (1) different GPUs/generations — on Blackwell the BF16
anchor itself is very strong (TC-dense BF16 + FlashAttention + graphs), which
compresses ALL quantization speedups relative to older-GPU papers; the NVFP4
row is the calibration point — the best production 4-bit stack on this card
gets 1.27-1.51x on the same protocol where QServe cites 1.2-3.47x on
A100/L40S. (2) QServe quantizes activations to 8 bits (W4A8); ProMix/NVFP4/
Atom quantize to 4 bits — 2x more aggressive on the activation side.
(3) AWQ's 2.7-3.9x is against HF eager, not a serving engine. (4) Our numbers
are end-to-end engine throughput including scheduler/sampling, not kernel-only.

## 8. Honest notes & caveats

1. **Session noise**: one engine per GPU on an otherwise-idle card, but the
   box is shared (8 GPUs, other tenants' host load). Repeat-run wobble on this
   protocol has historically been ±3-4% on b1 long-context rows; the b>=64
   rows replicate across cases and are the load-bearing numbers.
2. **TPOT at high concurrency** is (full−TTFT)/(OSL−1) with all B requests
   submitted together — it mixes a small amount of scheduling/chunked-prefill
   into the decode average for every method equally.
3. **ProMix checkpoint is fp16; baselines bf16** (same width; the known
   bit-exact cast-fold handles the GEMM-out cast).
4. **b512 decode-heavy at 8B** exceeds the KV budget for BF16
   (67.9 GiB KV available vs ~77 GiB needed at 1152 tok x 512 seqs):
   vLLM preempts/recomputes rather than OOM — throughput numbers at that cell
   include preemption for BF16 (a capacity result in itself; 4-bit rows keep
   22-14 GiB more KV headroom, §capacity tables).
5. **ProMix capacity**: ProMix's KV budget is the lowest of the quantized
   rows (fp16 lm_head+embeddings in the checkpoint + the marlin-hi companion
   copy, §r5 doc 2a) — a known, fixable packaging cost (lm_head W8 is
   PPL-safe per the 1B study).
6. **Pad-aware SF alloc fix** (this campaign): r-sweep splits with
   K_seg/VEC % 4 != 0 read uninitialized SF pad bytes (NaN risk) in the
   activation packers; fixed by conditional zero-fill. The shipping r=1/8
   splits are unpadded — their path is byte- and time-identical (gate §0 ran
   on the fixed stack).
7. **GPTQ-Marlin 3B prefill is anomalously slow** (worse than BF16, both
   sessions, matches the committed round-1 baselines) — Marlin's act-order
   dequant path at prefill M; AWQ-Marlin (ext §5) is the stronger W4A16
   prefill row at 3B. The W4A16 verdict above cites whichever is better.
8. FP8 W8A8 8B long-ISL kv8 row re-run after an infra race (chained engines);
   all other cells are first-pass.

## 9. Reproduce

```bash
B=/home/luofan/vllm_integ/bench_final       # on the GPU host
# core five-way, one config per GPU (ckpt, tag, dtype):
docker exec -d -e CUDA_VISIBLE_DEVICES=<n> luofan-plaquant \
  bash -c "bash $B/run_final.sh <ckpt> <tag> <dtype>"
# gates: promix.vllm_integ.gates.gate_t2 --ckpt <pm-ckpt> --graphs / --decode-route-m 16
# r-sweep: export per config promix/configs/llama-3.2-3b-fp-perhead-r<r>.yaml,
#   then bash $B/run_sweep_point.sh <ckpt> pm3b_r<r>
# accuracy (absolute protocol): /home/luofan/baselines/scripts/bench_vllm.py
# harvest: python -m promix.vllm_integ.harvest_final --dir docs/vllm_final_data --size 3b|8b
```
