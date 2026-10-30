# rd-qwen: Cross-Family Validation — ProMix W4A4 on Qwen3-8B

**Mission**: prove the W4A4 (NVFP4 main + MXFP8 high) quantization quality, the
measured-MSE allocation result, and the size-scaling trend are not
Llama-specific, by extending ProMix to a second model family (Qwen).

**Status**: feasibility PASSED; experiment COMPLETE. **Cross-family verdict:
the story holds on Qwen.**

---

## 1. Feasibility assessment: TRACTABLE (Qwen3-8B)

### Model availability

Only **Qwen/Qwen3-8B** is in the remote HF cache
(`/home/luofan/hf_cache/models--Qwen--Qwen3-8B`) — config + tokenizer only
(16 MB, no weight shards; weights must be fetched, HF reachable from the
container, verified HTTP 200). No Qwen2.5 model of any size is cached. So the
one-model run targets **Qwen3-8B** (36 layers, hidden 4096, 32 Q / 8 KV heads,
head_dim 128, intermediate 12288, untied embeddings) — directly comparable to
our Llama-3-8B anchor.

### Why the rotation math closes for Qwen3 (per-diff audit)

The `feat/rd-alloc` promix pipeline is fully ResQ-independent and addresses
modules by name (`model.model.layers[i].self_attn.{q,k,v,o}_proj`,
`mlp.{gate,up,down}_proj`, `input_layernorm`, `post_attention_layernorm`,
`model.norm`, `lm_head`) — **identical names in Qwen3**. Per architecture diff:

| Qwen3 diff vs Llama | Interaction with ProMix/ResQ | Verdict |
|---|---|---|
| **q_norm / k_norm** (per-head RMSNorm on q,k after projection, before RoPE) | R1 is fused on the **input** side of q/k/v_proj — their outputs are numerically unchanged, so q_norm/k_norm see identical tensors. R2 touches only the value path (v_proj output / o_proj input), which has no norm. `fuse_layer_norms` never visits them (they are not adjacent to a residual-stream linear); `add_actquant` leaves them fp16 (RMSNorm, not `nn.Linear`). | **No interaction. Closes exactly.** |
| key_pos basis (basis.py collects k_proj-output + RoPE, i.e. WITHOUT k_norm) | Only consumed by `setup_k_quant` (KV-cache quant). W4A4 runs use `k_bits=16` → never read. | Non-issue for W4A4; KV4-on-Qwen3 would need k_norm threaded into the basis hook (noted, out of scope). |
| No attention bias (`attention_bias=false`) | Same as Llama. (QKV **bias is a Qwen2.5 property**, not Qwen3: supporting Qwen2.5 would additionally require rotating `v_proj.bias` by R2 in `rotation._apply_had_to_linear`, which currently rotates weight only on the output path. R1 input-side fusion is bias-transparent — bias sits on the output side.) | Non-issue for Qwen3. |
| GQA 32Q/8KV | `_apply_had_to_linear` already handles kv_heads/kv_groups (same 32/8 structure as Llama-3-8B); `basis.py` collects per-kv-head H_value. | Already supported. |
| `config.head_dim` explicit (128) | All rotation/ptq sites derive head_dim = hidden//heads. For Qwen3-8B 4096/32 = 128 = config.head_dim — holds. (Qwen3-4B would NOT: 2560/32=80 vs head_dim=128 — plugin asserts.) | Holds for 8B. |
| intermediate_size 12288 | down_proj online Hadamard needs 12288 = K × 2^n: **12 × 1024 → had12** (was missing from `hadamard_data.json`; added via Paley I, q=11, verified ±1 entries and H·Hᵀ = 12·I). (Contrast Qwen2.5-7B: 18944 = 37 × 2^9 — **no Hadamard decomposition available**; Qwen2.5-7B would be blocked on the down_proj online Hadamard.) | Fixed with had12. |
| RoPE (theta 1e6, same rotate_half convention) | `basis.py` reuses `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` — identical formula to Qwen3's. | Non-issue. |
| tie_word_embeddings=false (8B) | loader's untie handling is a no-op. | Non-issue. |
| RMSNorm (same as Llama) | `fuse_layer_norms` fuses input/post-attn/final norms identically. | Non-issue. |

**Honest caveats**
1. **transformers**: Qwen3 needs ≥ 4.51; the shared container has 4.48.1. Run
   uses an isolated `pip install --target` of transformers 4.51.3 with
   PYTHONPATH precedence (shared env untouched).
2. **fp16 build dtype**: `load_model` builds the quantized model in fp16 (as
   for all Llama runs). Qwen family has a history of fp16 activation overflow
   (mostly Qwen2-72B); watched empirically — contingency is bf16 build.
3. The conclusion is **Qwen3-8B-specific** in two places: head_dim==hidden//heads
   (fails for Qwen3-4B) and no QKV bias (fails for Qwen2.5 until
   `_apply_had_to_linear` rotates v_proj.bias — a ~5-line bounded fix, plus a
   had-size check per intermediate_size; Qwen2.5-7B's 18944 has NO valid
   Hadamard decomposition, so Qwen2.5 would also need a different down_proj
   online rotation).

### Changes made (all on `feat/rd-qwen`)

- `promix/models/qwen.py` — Qwen3Plugin (+ registry entry in
  `promix/models/__init__.py`), with the head_dim assertion and the full arch
  caveat list in the docstring.
- `promix/configs/qwen3-8b-fp-perhead.yaml` — mirrors
  `llama-3-8b-fp-perhead.yaml` (same W4A4 = NVFP4 main + MXFP8 high, 1/8,
  per-head o_proj PCA, k/v 16-bit).
- `promix/quantize/hadamard_data.json` — added **had12** (Paley I, q=11;
  verified H·Hᵀ = 12·I through the same structure as the had108 addition).
- `promix/scripts/rd_ron.py::resolve_paths` — Qwen guard BEFORE the Llama size
  branches ("qwen3-8b" contains "8b" and would have silently remapped onto the
  llama-3-8b artifacts).
- `promix/scripts/rd_metrics.py` — opt-in `--no_c4` flag (run-host download
  robustness); default unchanged.

### Protocol (identical to the Llama 1B/3B/8B/13B suite)

`rd_metrics.py` on the dense-R-on shipped regime: (a) bf16 bridge (no
rotation, no quant), (b) uniform-1/8 W4A4, (c) measured-MSE-1/8 at the matched
FP8 ring budget. Step 0 (`basis.py`) generates the Qwen3-8B PCA basis +
random-init R offline first (Step 1 rotation training intentionally skipped —
same as the shipped Llama protocol).

### Llama reference (what "reproducing the pattern" means)

From `docs/rd_metrics_report.md` (wikitext, measured − uniform):
1B **−0.077**, 3B **−0.030**, 8B **+0.019**, 13B **−0.0006** — i.e. a small
gain at 1–3B that decays to break-even by 8B+. For Qwen3-8B the scaling trend
predicts **~0 to small** allocation delta, and a uniform-vs-bf16 gap of the
same order as Llama-8B (bf16 6.14 → uniform 6.59, +0.45 ≈ +7%; absolute PPLs
are not comparable across tokenizers — only within-family gaps are).

---

## 2. Results (Qwen3-8B, dense-R-on, same protocol as the Llama 1B–13B suite)

Pipeline ran END-TO-END on a fresh substrate with zero quantization-logic
changes: 16 GB weight download → Step 0 PCA basis
(`U/R/E-wikitext-512-Qwen3-8B.bin`, 512×2048, ~2.3 h) →
`rd_metrics.py --configs bf16,uniform,measured --no_zeroshot`. Raw JSON:
`docs/rd_metrics/RD_METRICS_qwen3_8b.json`. The rd_metrics bf16 leg reproduced
the standalone bridge exactly (9.7247 / 13.2993) — evaluator consistency gate
passed. Both quantized legs realized avg_fp8 = 0.1250 exactly
(165888/1327104 channels; measured budget 2304/18432 rings).

| point | wikitext | C4 |
|---|---|---|
| bf16 bridge | **9.7247** | **13.2993** |
| uniform-1/8 W4A4 | 9.9665 | 13.7319 |
| measured-MSE-1/8 | 9.9777 | 13.6805 |

(Absolute PPLs are not comparable to Llama-3-8B's 6.14/8.88 — different
tokenizer, vocab 151936 vs 128256, and training mix; only within-family gaps
carry meaning.)

**(a) Does ProMix W4A4 quantize Qwen well?** YES — better (relative) than
Llama-8B. Uniform-1/8 gap vs bf16: wikitext **+0.242 (+2.5%)**, C4 **+0.433
(+3.3%)**, vs Llama-3-8B's +0.451 (+7.4%) / +0.753 (+8.5%). The PCA+rotation
+ NVFP4/MXFP8 split transfers to the Qwen family with no degradation — no
arch-specific accuracy pathology from qk-norm, GQA, the 1e6-theta RoPE, or the
had12 down_proj Hadamard.

**(b) Does the measured-MSE allocation result reproduce the Llama pattern?**
YES — measured − uniform: wikitext **+0.011**, C4 **−0.051**. Break-even with
mixed signs across the two PPL families, numerically right where the Llama
scaling trend predicts for an 8B (Llama-8B: wikitext +0.019, C4 +0.004; the
gain only exists at 1–3B and decays to ~0 by 8B). The cross-family datapoint
lands ON the trend, not off it.

**(c) Arch-specific breakage?** NONE at runtime. The entire run used the
unmodified quantization/rotation/GPTQ/eval code; the only Qwen-specific
changes were artifacts/plumbing (plugin + config + had12 + resolve_paths
guard). Caveats that stay on the record: fp16 build dtype showed no overflow
on Qwen3-8B (watched); key_pos basis is pre-k_norm (irrelevant here, blocks
only future KV4-on-Qwen3); Qwen3-4B (head_dim≠hidden/heads) and Qwen2.5 (QKV
bias + intermediate 18944 has no Hadamard decomposition) remain out of scope.

### Cross-family verdict

All three claims survive the family swap on Qwen3-8B: (1) W4A4
(NVFP4+MXFP8-1/8) quantization quality is NOT Llama-specific — the uniform gap
is actually smaller on Qwen; (2) the measured-MSE allocation's size-scaling
behavior is NOT Llama-specific — break-even at 8B exactly as the Llama 8B/13B
points predict; (3) the ResQ-style PCA+rotation substrate (fuse_norm → U·R1 →
per-head R2 → online Hadamard) closes mathematically and empirically on a
second architecture with qk-norm and GQA. What this does NOT yet show: a Qwen
point at 1–3B where the allocation gain is nonzero for Llama (no small Qwen
model cached; worth one follow-up run if a small Qwen is fetched —
Qwen3-1.7B is a tractable candidate: hidden 2048 / 16 heads = head_dim 128 ✓,
intermediate 6144 = 12 × 512 → had12 ✓).

### Reproduce

```bash
# remote (docker luofan-plaquant, Pro 6000), isolated dir + isolated transformers:
git clone -b feat/rd-qwen https://github.com/Ther-LF/plaquant.git /home/luofan/plaquant-qwen
docker exec luofan-plaquant pip install -q --target=/home/luofan/plaquant-qwen/_pydeps transformers==4.51.3
# prune _pydeps to transformers* only (system hub/tokenizers/fsspec are compatible;
# the extra wheels pulled by --target would shadow and break datasets 2.20)
docker exec -d -e CUDA_VISIBLE_DEVICES=<idle> -e HF_HOME=/home/luofan/hf_cache \
  -e PYTHONPATH=/home/luofan/plaquant-qwen/_pydeps:/home/luofan/plaquant-qwen \
  -w /home/luofan/plaquant-qwen luofan-plaquant bash -c \
  'python -m promix.quantize.basis --config promix/configs/qwen3-8b-fp-perhead.yaml \
     --output_dir ./rotation --nsamples 512 > basis.log 2>&1'   # ~2.3 h
docker exec -d ... bash -c \
  'python promix/scripts/rd_metrics.py --config promix/configs/qwen3-8b-fp-perhead.yaml \
     --configs bf16,uniform,measured --no_zeroshot \
     --out /home/luofan/plaquant-qwen/RD_METRICS_qwen3_8b.json > metrics.log 2>&1'  # ~1.5 h
```
