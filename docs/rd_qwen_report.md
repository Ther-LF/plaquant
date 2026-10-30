# rd-qwen: Cross-Family Validation — ProMix W4A4 on Qwen3-8B

**Mission**: prove the W4A4 (NVFP4 main + MXFP8 high) quantization quality, the
measured-MSE allocation result, and the size-scaling trend are not
Llama-specific, by extending ProMix to a second model family (Qwen).

**Status**: feasibility PASSED (tractable); experiment phase in progress.

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

## 2. Results

(pending — filled in by the experiment phase)
