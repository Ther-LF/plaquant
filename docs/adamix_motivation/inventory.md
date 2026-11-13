# AdaMix Motivation — Inventory & Start-of-Run Audit

Run context: 2026-11-13. CPU-only (all 8 Pro6000 GPUs occupied by another team's
rebuttal and must not be touched — verified: `nvidia-smi` shows 68–90 GB used and
0–90% util on every card). This audit covers task book §2.1 + §8 step 1. Pure
audit, no GPU.

## 1. Code / repo state

| item | value |
|---|---|
| Mac repo | `/Users/luofan/plaquant`, branch `feat/rd-alloc` |
| HEAD commit | `f3cd816` ("tracker: Qwen14B 'head-dim bug' corrected ...") |
| Pro6000 isolated worktree | `/home/luofan/plaquant-adamix`, new branch `feat/adamix-motiv`, forked from `origin/feat/rd-alloc` @ `f3cd816` |
| other worktrees (DO NOT TOUCH) | `plaquant-densesweep` (feat/rd-densesweep @ bc422de), `plaquant-accmatrix`, `plaquant-qwenmid`, `plaquant-vllm-integ` — only their `rotation/` artifacts are read |
| uncommitted in adamix worktree | `promix/scripts/adamix_e1a.py` (new, this run), `rotation` symlink → densesweep rotation dir, `docs/adamix_motivation/` (deliverables) |

No pre-existing uncommitted diffs on the fork point; the worktree is a clean
`f3cd816` plus this run's additions.

## 2. Model / checkpoint / basis / rotation

Model (Llama-3.2-3B), local checkpoint — no HF download needed:

| item | value |
|---|---|
| path | `/home/luofan/models/Llama-3.2-3B` |
| shards | `model-00001-of-00002.safetensors` (4.97 GB), `model-00002-of-00002.safetensors` (1.46 GB) |
| config | hidden_size=3072, intermediate_size=8192, num_hidden_layers=28, num_attention_heads=24, num_key_value_heads=8, head_dim=128, torch_dtype=bfloat16, tie_word_embeddings=true |
| note | `tie_word_embeddings=true` → loader unties and clones lm_head (promix `load_model`); E1a does not touch lm_head |

Rotation artifacts (resolved via the densesweep `rotation/` dir, symlinks into
`plaquant-vllm-integ/rotation/`). sha256 (first 20 hex):

| file | sha256[:20] | contents |
|---|---|---|
| `U-wikitext-512-Llama-3.2-3B.bin` | `3a402945c1d33f8b31b7` | dict: `config` (str), `attn_mlp` **(3072×3072 shared PCA basis P)**, per-layer `self_attn.value` (8×128×128), `self_attn.key_pos` (128×128), `mlp.down_proj` (256×256) |
| `E-wikitext-512-Llama-3.2-3B.bin` | `d480add1e1cde80632b9` | eigenvalues |
| `R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-3B.bin` | `e2976a5b5469f5be0811` | dict: `R1_1`, `R1_2` (+ optional `R1_0`) learned Stiefel rotation blocks |

Key finding for E1a: the shipped PCA basis for the residual stream is the single
shared `attn_mlp` (3072×3072) applied to q/k/v/gate/up inputs. down_proj in the
shipped pipeline uses a **256-block PCA (`mlp.down_proj` 256×256) + online
Hadamard**, i.e. NOT a single global PCA — this shapes the down_proj caveat in
E1a (see §4 of the report).

## 3. Devices / environment (verified)

| item | value |
|---|---|
| host | 128 CPU cores, ~444 GB RAM free |
| GPUs | 8× RTX Pro 6000; ALL occupied (68–90 GB used) → **not used**, `CUDA_VISIBLE_DEVICES=""` on every job |
| python | system `python3`: torch 2.11.0+cu130, transformers 4.48.1, fast_hadamard_transform present (CPU paths work) |
| HF | local checkpoint used; `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME=/home/luofan/hf_cache` set as fallback (no token) |

## 4. Reusable existing data (task book §2.1) and its limits

Audited against the on-disk reports in `docs/`. Reusable for *context*, but
**none is a drop-in substitute** for the E1a/E2 protocol here (different ring
size, different R-state, PPL-only gates):

| source | reusable | limit (confirmed) |
|---|---|---|
| `docs/paper_algorithm.md`, `paper_kernel.md`, `paper_evidence_index.md` | algorithm / kernel / evidence entry points | must be re-checked against current code + checkpoint |
| `promix/scripts/rd_validity.py` + `docs/rd_validity_report.md` | **the reusable machinery** — `build_base`/`collect_stats`/`allocate` hooks; proven forward-pre-hook per-ring streaming | default `RING=64` (not 512); per-ring stat is `e/(n*ring)` (plain MSE, **NOT** task-loss weighted); reports R1-OFF, not ring512 main config |
| `promix/scripts/rd_ron.py`, `docs/rd_ron_report.md` | ring-R-on substrate (`build_uniform_ring_R1`), 64/128/256/512 compare | per-model benefit differs; old quant config not mixable with final tuned config |
| `rd_buildup.py` + `docs/rd_buildup_ablation.md` + `rd_buildup_data/` | cumulative ablation, 3-seed, perf breakdown | not a matched factorial; naive-serving is an *estimate* |
| `docs/rd_accuracy_tuning.md` | final precision recipe | verify against serving checkpoint identity |
| `docs/rd_metrics_report.md`, `rd_metrics/` | multi-model/metric results | dense-R-on, 64-group OLD config; 8B does not broadly benefit |
| `docs/rd_calib_robustness.md` | calibration-shift evidence | not a ring512 experiment |
| `docs/rd_torchfp4_report.md`, `rd_torchfp4_data/torchfp4_gemm.json` | native 2-GEMM+add perf (same HW) | mostly large-M, GEMM-only, fixed ratio — **E4, GPU-only, out of scope this run** |
| `promix/eval/gemm_h2h_torchfp4.py` | kernel perf harness start | **E4, GPU-only, out of scope** |
| `docs/rd_operator_pareto.{md,csv}` | quality/operator-perf reference | per-row config + cost basis must be audited |

Implementation traps confirmed relevant and handled in E1a:
- per-group stat is plain MSE `e/(n·G)`, not loss-weighted — named accurately.
- `clamp_chosen` (keep ≥1 NVFP4 group so K_low≠0) matters for *full-model* masks
  (E1b/E2-PTQ), **not** for the E1a isolated LDelta math (no kernel executes).
- G=512 on 3072-dim layers → only 6 groups; `round(1/8·6)=1` ⇒ rho_actual≈0.167,
  not 0.125 (reported honestly per-run). down_proj (8192 → 16 groups) hits
  round(2)=2 ⇒ rho=0.125 exactly.

## 5. What is present vs missing for this run's scope

Present / usable now (CPU):
- 3B checkpoint + shared PCA basis + learned rotation (all hashed above).
- Deployed fake-quant: `fake_quantize_nvfp4` (bs16, FP8-E4M3 block scale, FP4-E2M1
  elements, per-block MSE scale search ON by default), `fake_quantize_mxfp8`
  (bs32, E8M0 pow2 scale, E4M3 elements). Verified signatures/behavior in
  `promix/quantize/quant_utils.py`.
- Ring/blockdiag rotation builder `build_uniform_ring_R1` + `compose_rotation`
  (`promix/quantize/ring_rotation.py`); Sylvester-Hadamard, needs power-of-2 ring
  — G=512 OK; 3072=6·512 and 8192=16·512 both divisible.

Missing / blocked for this run:
- **GPU-only**: E4 kernel timing (native-mix2 vs fused), full-model PPL matrix
  (E1b, E2-PTQ 72-config). Deferred — all cards occupied.
- 8B checkpoint not audited here (follow-up; also CPU-doable but ~2.5× slower).
- No disjoint-C4 calibration split wired for E1a (uses wikitext-2 train fit/diag
  split per §3.2); C4 PPL is an E1b/E2 (GPU) concern.

## 6. E1a entry point (this run)

`promix/scripts/adamix_e1a.py` (new). CPU-only, isolated-linear 4×2 diagnostic;
reuses the fake-quant + ring-rotation modules and the forward-pre-hook collection
pattern from `rd_validity.collect_stats`. See `runs/e1a_3b/manifest.json` for the
exact parameters, hashes, and the down_proj local-PCA caveat.
