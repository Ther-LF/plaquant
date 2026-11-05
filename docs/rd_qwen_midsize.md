# rd-qwenmid: Qwen3 mid-size (14B / 32B) workstream

Status as of 2026-11-05. Branch `feat/rd-qwenmid` (merged with
`origin/feat/rd-alloc`). This document records the rotation-closure resolution,
the per-candidate GO/NO-GO verdict, and the accuracy/baseline campaign.

## 1. Rotation closure — RESOLVED (bit-exact for both matched and divergent head_dim)

### 1.1 The symptom and the red herring

The prior closure bisection reported cos ~0.9937 for a tiny divergent-head_dim
Qwen3 and feared the rotation was not orthogonality-preserving. A clean
stage-by-stage bisection (control = matched head_dim, divergent = head_dim 128
vs hidden//heads 64) shows this is **not** a rotation defect:

| model     | stage 1: norm-fuse only | stage 3: full rotation pipeline |
|-----------|-------------------------|---------------------------------|
| ctrl (matched)   | cos 0.99578547 | cos 0.99578547 (identical to 8 d.p.) |
| div  (divergent) | cos 0.99372649 | cos 0.99372649 (identical to 8 d.p.) |

The full rotation pipeline (basis-fuse + column reorder + per-head Hadamard +
down_proj online Hadamard) reproduces the **norm-fused baseline bit-for-bit**
(cos identical to 8 decimal places). The gap from 1.0 is entirely introduced by
stage 1 — `fuse_layer_norms` — and is present even for matched head_dim.

(The intermediate stage-2 cos ~0.17/0.31 reported in the raw bisect is a
measurement artifact: `fuse_basis_to_model` and `rearrange_columns` are a
matched pair — the model is only valid once both have run, so measuring between
them is meaningless.)

### 1.2 Root cause of the 0.9937: embedding mean-subtraction in fuse_layer_norms

`promix/quantize/fuse_norm.py::fuse_layer_norms` mean-centers the embedding rows:

```python
for W in [model.model.embed_tokens]:
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(...)
```

This is the standard QuaRot/ResQ step (it drives the residual stream toward
zero-mean so the orthogonal rotations compose cleanly). RMSNorm is **not**
shift-invariant, so on a freshly-initialised *random* tiny model this
perceptibly changes the logits (cos 0.9937). On *trained* models the embedding
mean along the hidden dim is tiny, so it is the established near-exact
approximation — promix Llama-3.2-1B matches ResQ PPL exactly, and Qwen3-8B
already ships with good accuracy under the identical pipeline.

Proof: pre-center the embedding (`E -= E.mean(-1)`) so the mean-subtraction
becomes a no-op, then run the full pipeline against the pre-centered reference:

| model              | cos         | argmax agree | max_abs  |
|--------------------|-------------|--------------|----------|
| ctrl (matched)     | 0.99999994  | 1.0000       | 4.6e-06  |
| div  (divergent)   | 1.00000000  | 1.0000       | 2.4e-06  |

The rotation is bit-exact orthogonality-preserving for **both** matched (14B)
and divergent (32B) head_dim.

### 1.3 The head_dim fix (origin commit `bbcf057`)

`rotation.py::_apply_had_to_linear` derived the per-head Hadamard group count
from `init_shape[0]` (out_features):

```python
num_kv_groups = init_shape[0] // (had_dim * num_kv_heads)   # OLD
num_kv_groups = (init_shape[-1] // had_dim) // num_kv_heads  # FIX
```

Head blocks live on the **input** side (in_features = q_heads * head_dim). The
old formula only worked when out == in == hidden (Llama). On Qwen3 with an
explicit `config.head_dim`, o_proj has out != in (Qwen3-32B: out 5120, in 8192),
so the old count under-counted groups and left head blocks unrotated. The fix
is validated by the divergent-model closure above (cos 1.00000000). This fix was
already on `origin/feat/rd-qwenmid` as `bbcf057`; the local WIP was an identical
duplicate and was reset onto it.

### 1.4 down_proj online Hadamard matrices — verified orthogonal

The down_proj intermediate dims factor into a small Hadamard x block structure.
All matrices in `promix/quantize/hadamard_data.json` verified with +-1 entries
and `H @ H.T == n*I` exactly (max off-diagonal 0):

| key    | order | used by | H H^T == nI |
|--------|-------|---------|-------------|
| had12  | 12    | 8B  (12288 = 12 x 1024)  | yes |
| had68  | 68    | 14B (17408 = 68 x 256)   | yes |
| had100 | 100   | 32B (25600 = 100 x 256)  | yes |
| had28  | 28    | -       | yes |
| had108 | 108   | -       | yes |

had68 (Paley I, q=67) and had100 (Paley II, q=49=7^2) are the mid-size pieces;
both are genuine Hadamard matrices, so the down_proj online rotation closes.

## 2. Per-candidate GO / NO-GO

| candidate  | head_dim    | down_proj Had | rotation closure | verdict |
|------------|-------------|---------------|------------------|---------|
| Qwen3-8B   | 128 == 4096/32 (matched) | had12  | closes (shipped) | GO (already shipped) |
| Qwen3-14B  | 128 == 5120/40 (matched) | had68  | closes (cos 1.0, matched case) | **GO** |
| Qwen3-32B  | 128 vs 5120/64=80 (divergent) | had100 | closes with bbcf057 fix (cos 1.0, divergent case) | **GO** |

No NO-GO. The explicit-head_dim threading (`promix.utils.config_head_dim` used
by basis/rotation/ptq/kv_quant) plus the `bbcf057` per-head group-count fix make
the divergent 32B tractable without an unbounded refactor. 14B is the simpler
matched case and is the primary ladder point to land first.

Caveats carried from `models/qwen.py` (unchanged, do not affect W4A4 closure):
- KV4 on Qwen3 is not yet supported (key_pos basis is collected without k_norm;
  W4A4 configs use k_bits=16 and never read it).
- Qwen2.5 (QKV bias) would additionally need v_proj.bias rotated by R2.

## 3. Accuracy / baseline campaign

Protocol (matches the acctune/densesweep shipping recipe): ring512 substrate +
global measured-deltaD ring allocation + act-MSE d3u2, wiki+C4 fake-quant PPL.
Driver: `promix/scripts/rd_acctune.py` (now carries both `--budget_frac` and the
`--stats_cache` / `rv.clamp_chosen` densesweep machinery after the rd-alloc
merge).

- bf16 bridge PPL (wiki+C4) = protocol-matched baseline.
- 5%-grid (budget_frac 0.05..1.00, 20 points) ProMix accuracy curve.
- vLLM-NVFP4 baseline.

Prerequisite: the Qwen3-14B PCA basis + initial rotation
(`U-/E-/R-...-Qwen3-14B.bin`) must be generated first (none existed). This is in
progress. Numbers will be appended here as the campaign (queued behind the
box's densesweep/accmatrix jobs on the shared GPUs) completes.

### Basis-generation fix (transformers >= 5.x)

Computing the Qwen3-14B PCA basis surfaced a second, independent bug in
`basis.py` (not a rotation/closure issue). `compute_basis` runs a manual
layer-by-layer forward and feeds each layer's output back as the next layer's
input; it took `layer(...)[0]`, assuming the transformers <= 4.51 tuple
`(hidden, ...)` return. transformers >= 5.x decoder layers return a **bare
Tensor**, so `[0]` sliced the batch dimension -> 2D hidden -> the next layer
reshaped q to `[seq, num_heads, head_dim].transpose(1,2)`, putting num_heads on
the RoPE axis and crashing `apply_rotary_pos_emb` (size 40 = Qwen3-14B num_heads
vs head_dim 128). Fixed by unwrapping defensively (commit `5d1dd73`); verified
the isolated 14B-dims layer forward keeps `[1, seq, hidden]`. This unblocks
basis generation for Qwen3 (and any model) under the box's transformers 5.17.

### Accuracy / baseline results (appended as they land)

The Qwen3-14B PCA basis is recomputing (CPU-bound FP32 covariance, ~340 s/layer
x 40 layers ~= 3.8 h on the shared box). A chained, detached campaign
(`/home/luofan/qwen14b_campaign.sh`, GPU-pinned, `nice`) waits for the basis,
then runs the bf16 bridge + shipped r=0.125 point, then the 5% budget grid
(0.05..1.00, 20 points), each writing `/home/luofan/qwenmid_work/14b_r<R>.json`
(logs under `.../logs/`). Collate those JSONs here once they land. (Numbers are
intentionally not reported until the runs complete and are verified.)

## 4. Commits

- `bbcf057` rd-qwenmid: per-head Hadamard group-count fix (out->in features) — validated bit-exact here.
- `4f16081` rd-qwenmid: merge origin/feat/rd-alloc; resolve rd_acctune.py clamp_chosen/budget_frac conflict keeping both sides.
- `a20c93a` rd-qwenmid: this closure-resolution doc.
- `5d1dd73` rd-qwenmid: basis.py layer-output unwrap fix for transformers >=5.x (unblocks 14B basis).
