# Adaptive High-Fraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed `high_fraction=1/8` on the main path (q/k/v/gate/up) with a per-layer, energy-threshold-derived high-precision cut, and produce a (avg bit-width, PPL) Pareto curve vs the fixed-fraction baseline.

**Architecture:** A single threshold τ drives per-layer high/low splits. Step 0 emits each layer's per-channel variance in the shared PCA basis (nearly free, from already-collected covariances). A pure helper derives per-layer `K_high` from τ. The main-path rotation uses identity R1 (so the per-layer cut length is unconstrained by a fixed block boundary). PTQ wires per-layer `K_high` into each main-path quantizer.

**Tech Stack:** Python, PyTorch 2.11 (cu130), transformers 4.48.1; runs in the `luofan-plaquant` Docker container on the Pro 6000 (SM120). No local torch — all tests run via `docker exec`.

**Spec:** `docs/superpowers/specs/2026-10-09-adaptive-high-fraction-design.md`

## Global Constraints

- Block alignment: `K_high` must be a multiple of 64 (satisfies MXFP8 K%32=0 and NVFP4 K_low%64=0 simultaneously, since hidden_dim 2048 % 64 = 0).
- High-precision channels are always the **contiguous tail** of the shared-U ordering (U columns sorted ascending by variance). Only the cut length varies per layer; the channel set does not.
- Scope: main path only (q/k/v/gate/up). o_proj (per-head), down_proj (Hadamard), KV (FP16) are untouched this round.
- Model: `unsloth/Llama-3.2-1B-Instruct`, hidden_dim=2048, 16 layers. Eval: wikitext-2 PPL.
- Existing fixed-`high_fraction` behavior must be byte-for-byte unchanged when `adaptive_high_fraction` is false (no regression).
- Tests run in the container: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest <path> -v`.

---

### Task 1: Pure adaptive helpers (`adaptive.py`)

**Files:**
- Create: `promix/quantize/adaptive.py`
- Test: `tests/test_adaptive.py`

**Interfaces:**
- Produces:
  - `per_channel_variance_in_basis(U, H) -> Tensor(d,)` — `diag(Uᵀ H U)`.
  - `tail_energy_fraction(var_curve) -> Tensor(n,)` — cumulative fraction captured by the last k channels, k=1..n.
  - `derive_khigh(var_curve, tau, align=64, hidden_dim=None) -> int`.
  - `derive_per_layer_khigh(var_curves, tau, align=64, hidden_dim=None) -> list[int]`.
  - `main_path_avg_bits(khigh_list, hidden_dim, low_bits=4, high_bits=8) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adaptive.py
import torch
from promix.quantize.adaptive import (
    per_channel_variance_in_basis, tail_energy_fraction,
    derive_khigh, derive_per_layer_khigh, main_path_avg_bits,
)

def test_variance_in_basis_identity():
    H = torch.diag(torch.tensor([1., 2., 3., 4.]))
    U = torch.eye(4)
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(v, torch.tensor([1., 2., 3., 4.], dtype=v.dtype))

def test_variance_in_basis_rotates():
    # U^T H U diagonal equals eigenvalues when U are eigenvectors
    H = torch.tensor([[2., 1.], [1., 2.]])
    evals, U = torch.linalg.eigh(H)
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(torch.sort(v).values,
                          torch.sort(evals.double()).values, atol=1e-10)

def test_tail_energy_endpoints():
    v = torch.tensor([0., 0., 1., 3.])      # total 4
    cum = tail_energy_fraction(v)           # last1=3/4, last2=4/4, ...
    assert torch.isclose(cum[0], torch.tensor(0.75, dtype=cum.dtype))
    assert torch.isclose(cum[-1], torch.tensor(1.0, dtype=cum.dtype))

def test_derive_khigh_alignment_and_threshold():
    # 16 channels, all energy in last 4 → need k>=4, round up to 64 then
    # clamp to hidden_dim (16) → but with align=4 for this small case:
    v = torch.zeros(16); v[-4:] = 1.0
    assert derive_khigh(v, tau=0.9, align=4, hidden_dim=16) == 4
    # ascending 1..16, tau=0.5: smallest tail k with sum>=68 is k=5 → align4 → 8
    v2 = torch.arange(1, 17, dtype=torch.float64)
    assert derive_khigh(v2, tau=0.5, align=4, hidden_dim=16) == 8

def test_derive_khigh_clamped_to_align_min():
    v = torch.zeros(16); v[-1] = 1.0
    # need k=1 → round up to align 4
    assert derive_khigh(v, tau=0.9, align=4, hidden_dim=16) == 4

def test_derive_per_layer_list():
    curves = [torch.zeros(16), torch.zeros(16)]
    curves[0][-4:] = 1.0      # needs 4
    curves[1][-8:] = 1.0      # needs 8
    ks = derive_per_layer_khigh(curves, tau=0.9, align=4, hidden_dim=16)
    assert ks == [4, 8]

def test_avg_bits():
    # 2 layers, hidden 16: K_high 4 and 8
    # layer0: (12*4 + 4*8)/16 = (48+32)/16 = 5.0
    # layer1: (8*4  + 8*8)/16 = (32+64)/16 = 6.0 → mean 5.5
    assert abs(main_path_avg_bits([4, 8], hidden_dim=16) - 5.5) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py -v`
Expected: FAIL with `ModuleNotFoundError: promix.quantize.adaptive`.

- [ ] **Step 3: Write the implementation**

```python
# promix/quantize/adaptive.py
"""Energy-threshold-driven per-layer high-precision cut (main path).

See docs/superpowers/specs/2026-10-09-adaptive-high-fraction-design.md.
All curves are in shared-U column order (ascending variance), so the
high-variance channels are the TAIL and the quantizer's last-K slice
matches `derive_khigh`'s tail-energy accounting.
"""
import torch


def per_channel_variance_in_basis(U, H):
    """diag(Uᵀ H U): per-channel variance after projecting covariance H
    onto basis U (columns = basis vectors). Returns a (d,) tensor whose
    c-th entry is the variance along U[:, c]."""
    U = torch.as_tensor(U, dtype=torch.float64)
    H = torch.as_tensor(H, dtype=torch.float64)
    M = H @ U                      # (d, d)
    return (U * M).sum(dim=0)      # diag(Uᵀ H U)


def tail_energy_fraction(var_curve):
    """Cumulative fraction of total variance captured by the LAST k channels,
    for k = 1..n. Returns a (n,) tensor; entry [k-1] = frac for last k."""
    v = torch.as_tensor(var_curve, dtype=torch.float64).flatten().clamp_min(0)
    total = v.sum()
    if total <= 0:
        raise ValueError("variance curve sums to <= 0")
    cum = torch.cumsum(torch.flip(v, dims=[0]), dim=0) / total
    return cum


def _round_up(x, m):
    return ((int(x) + m - 1) // m) * m


def derive_khigh(var_curve, tau, align=64, hidden_dim=None):
    """Smallest multiple of `align` such that the last K_high channels capture
    >= tau of the variance. Clamped to [align, hidden_dim]."""
    v = torch.as_tensor(var_curve, dtype=torch.float64).flatten().clamp_min(0)
    n = hidden_dim if hidden_dim is not None else v.numel()
    cum = tail_energy_fraction(v)
    meets = (cum >= tau).nonzero()
    k = int(meets[0].item()) + 1 if len(meets) else n
    return max(align, min(_round_up(k, align), n))


def derive_per_layer_khigh(var_curves, tau, align=64, hidden_dim=None):
    """Per-layer K_high list from a list of variance curves."""
    return [derive_khigh(v, tau, align=align, hidden_dim=hidden_dim)
            for v in var_curves]


def main_path_avg_bits(khigh_list, hidden_dim, low_bits=4, high_bits=8):
    """Mean over layers of (K_low*low_bits + K_high*high_bits)/hidden_dim."""
    per = [((hidden_dim - k) * low_bits + k * high_bits) / hidden_dim
           for k in khigh_list]
    return sum(per) / len(per)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add promix/quantize/adaptive.py tests/test_adaptive.py
git commit -m "feat(adaptive): pure helpers for per-layer energy-threshold K_high"
```

---

### Task 2: Step 0 emits per-layer variance curves (`basis.py`)

**Files:**
- Modify: `promix/quantize/basis.py` (full_shared branch, after `evec_attn_mlp` is computed)
- Test: `tests/test_adaptive.py` (add one integration-style test of the wiring helper)

**Interfaces:**
- Consumes: `per_channel_variance_in_basis` from Task 1.
- Produces: eval bundle gains keys `layer.{i}.attn_mlp_var` (each a `(hidden_dim,)` CPU tensor, shared-U column order) for i in 0..n_layers-1.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_adaptive.py
from promix.quantize.adaptive import per_channel_variance_in_basis

def test_variance_matches_eigvals_when_U_is_eigenbasis():
    # When U diagonalizes H, diag(U^T H U) == eigenvalues. This is the
    # property basis.py relies on to store per-layer curves consistently
    # with the shared eigenvalue ordering.
    torch.manual_seed(0)
    A = torch.randn(8, 8, dtype=torch.float64)
    H = A @ A.T                          # SPD
    evals, U = torch.linalg.eigh(H)      # ascending
    v = per_channel_variance_in_basis(U, H)
    assert torch.allclose(v, evals, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py::test_variance_matches_eigvals_when_U_is_eigenbasis -v`
Expected: PASS (Task 1 already provides the helper; this locks the contract basis.py depends on).

- [ ] **Step 3: Wire it into basis.py**

In the `full_shared` branch, immediately after:
```python
        basis_dict["attn_mlp"] = evec_attn_mlp.cpu()
        eval_dict["config"] = "full_shared_rotation"
        eval_dict["attn_mlp"] = eval_attn_mlp.cpu()
```
add:
```python
        # Per-layer per-channel variance in the SHARED basis, for adaptive
        # high-fraction (see adaptive.py / the adaptive-high-fraction spec).
        # H_attn[i], H_mlp[i] are still resident here; no extra forward pass.
        from promix.quantize.adaptive import per_channel_variance_in_basis
        U_am = evec_attn_mlp.to(torch.float64).cuda()
        for i in range(nlayers):
            Hi = ((H_attn[i] + H_mlp[i]).to(torch.float64).cuda()
                  / (2 * nbatches * seqlen))
            var_i = per_channel_variance_in_basis(U_am, Hi)
            eval_dict[f"layer.{i}.attn_mlp_var"] = var_i.cpu()
        del U_am
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Regenerate the bundle and verify the key exists**

IMPORTANT: when the `U-…bin` basis file exists, basis.py short-circuits and
skips the entire covariance collection — so the new per-layer variance code
(which needs `H_attn/H_mlp`) will NOT run. Delete BOTH the U and E files to
force a full recompute (~16 min covariance + eigendecomp on GPU 7):
```bash
docker exec luofan-plaquant bash -c "cd /home/luofan/plaquant/rotation && rm -f U-wikitext-512-Llama-3.2-1B-Instruct.bin E-wikitext-512-Llama-3.2-1B-Instruct.bin"
```
Also remove the symlinks that pointed at the old U/E (Stage 1b created
`*-fp-perhead-*` symlinks) so they are re-pointed after regen; the config
`llama-3.2-1b-fp-perhead.yaml` uses native names via basis.py, so just rerun:
```bash
docker exec -w /home/luofan/plaquant -e CUDA_VISIBLE_DEVICES=7 -e HF_HOME=/home/luofan/hf_cache -e PYTHONPATH=/home/luofan/plaquant -e PYTHONUNBUFFERED=1 luofan-plaquant python -m promix.quantize.basis --config promix/configs/llama-3.2-1b-fp-perhead.yaml --output_dir ./rotation --nsamples 512
```
Then verify:
```bash
docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -c "import torch; E=torch.load('rotation/E-wikitext-512-Llama-3.2-1B-Instruct.bin'); assert 'layer.0.attn_mlp_var' in E and E['layer.0.attn_mlp_var'].shape[0]==2048; print('OK', E['layer.0.attn_mlp_var'].shape)"
```
Expected: `OK torch.Size([2048])`. Budget ~16 min on GPU 7.

- [ ] **Step 5: Commit**

```bash
git add promix/quantize/basis.py tests/test_adaptive.py
git commit -m "feat(basis): emit per-layer attn_mlp variance in shared basis"
```

---

### Task 3: Identity-R1 path for adaptive mode (`rotation.py`)

**Files:**
- Modify: `promix/quantize/rotation.py` (`fuse_basis_to_model`, around lines 182-211)
- Test: `tests/test_adaptive.py`

**Interfaces:**
- Produces: `fuse_basis_to_model(..., identity_main_rotation=False)`. When True, the main-path transform is `T = U_attn` (R1 skipped), so no fixed [main|high] block boundary is imposed and per-layer cut lengths are unconstrained.
- Helper (extracted for testability): `build_main_transform(U_attn, R_dict, identity_main_rotation) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_adaptive.py
from promix.quantize.rotation import build_main_transform

def test_build_main_transform_identity_returns_U():
    U = torch.randn(8, 8, dtype=torch.float64)
    T = build_main_transform(U, R_dict=None, identity_main_rotation=True)
    assert torch.allclose(T, U)

def test_build_main_transform_applies_R1_when_not_identity():
    U = torch.eye(4, dtype=torch.float64)
    R_dict = {"R1_1": torch.eye(2, dtype=torch.float64) * 2,
              "R1_2": torch.eye(2, dtype=torch.float64) * 3,
              "R1_0": None}
    T = build_main_transform(U, R_dict=R_dict, identity_main_rotation=False)
    expected = torch.block_diag(torch.eye(2) * 2, torch.eye(2) * 3).double()
    assert torch.allclose(T, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py::test_build_main_transform_identity_returns_U -v`
Expected: FAIL with `ImportError: cannot import name 'build_main_transform'`.

- [ ] **Step 3: Extract helper and add the flag**

Add near the top of `rotation.py`:
```python
def build_main_transform(U_attn, R_dict, identity_main_rotation):
    """Compose the main-path transform T. With identity_main_rotation,
    T = U_attn (pure PCA, no trained R1 block boundary). Otherwise
    T = U_attn @ block_diag(R1_0?, R1_1, R1_2)."""
    if identity_main_rotation:
        return U_attn
    R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
    R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
    R1 = torch.block_diag(R1_1, R1_2)
    R1_0 = R_dict.get("R1_0")
    if R1_0 is not None:
        R1 = torch.block_diag(R1_0.cuda().to(torch.float64), R1)
    return torch.matmul(U_attn, R1)
```
Change `fuse_basis_to_model` signature to accept `identity_main_rotation=False`, and replace the inline R1 block (lines ~201-211) with:
```python
    R_dict = torch.load(rotation_path, weights_only=False)
    U_attn = build_main_transform(U_attn, R_dict, identity_main_rotation)
```
(Delete the now-redundant `R1_1/R1_2/assert/block_diag/matmul` lines. Keep the `R_dict` load since R2 keys are still read below.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py -k build_main_transform -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add promix/quantize/rotation.py tests/test_adaptive.py
git commit -m "feat(rotation): identity-R1 path for adaptive main-path cut"
```

---

### Task 4: PTQ adaptive plumbing (`ptq.py`)

**Files:**
- Modify: `promix/eval/ptq.py` (`configure_quantizers` loop; the `fuse_basis_to_model` call site)
- Test: `tests/test_adaptive.py` (unit-test the per-layer K lookup logic via an extracted helper)

**Interfaces:**
- Consumes: `derive_per_layer_khigh` (Task 1), the `layer.{i}.attn_mlp_var` eval keys (Task 2), `fuse_basis_to_model(..., identity_main_rotation=...)` (Task 3).
- Produces: new `quantize` config knobs `adaptive_high_fraction: bool` and `energy_threshold: float`; helper `layer_index_of(name) -> int|None` and `main_path_khigh_map(eval_bundle, tau, n_layers) -> dict[int,int]`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_adaptive.py
from promix.eval.ptq import layer_index_of

def test_layer_index_of():
    assert layer_index_of("model.layers.3.self_attn.q_proj.module") == 3
    assert layer_index_of("model.layers.11.mlp.gate_proj") == 11
    assert layer_index_of("lm_head") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py::test_layer_index_of -v`
Expected: FAIL with `ImportError` / `AttributeError`.

- [ ] **Step 3: Implement the plumbing**

Add helpers to `ptq.py`:
```python
import re

def layer_index_of(name):
    """Extract the decoder layer index from a module name, or None."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None

def main_path_khigh_map(eval_bundle, tau, n_layers, hidden_dim=2048, align=64):
    """Per-layer K_high for the main path from the attn_mlp_var curves."""
    from promix.quantize.adaptive import derive_khigh
    out = {}
    for i in range(n_layers):
        key = f"layer.{i}.attn_mlp_var"
        if key not in eval_bundle:
            raise KeyError(
                f"{key} missing from eval bundle; re-run Step 0 (basis.py) "
                "with the adaptive-variance patch to use adaptive_high_fraction."
            )
        out[i] = derive_khigh(eval_bundle[key], tau, align=align,
                              hidden_dim=hidden_dim)
    return out
```
In `configure_quantizers`, read the knobs and build the map once:
```python
    adaptive = qcfg.get('adaptive_high_fraction', False)
    if adaptive:
        tau = float(qcfg['energy_threshold'])
        eval_path = config['paths']['basis'].replace('U-', 'E-')  # eval bundle
        eval_bundle = torch.load(eval_path, weights_only=False)
        khigh_map = main_path_khigh_map(eval_bundle, tau, len(model.model.layers),
                                        hidden_dim=model_dim)
```
Then inside the `for name in qlayers:` loop, for main-path layers (q/k/v/gate/up — i.e. not o_proj/down_proj/lm_head/v_proj-out), override when adaptive:
```python
        is_main = any(p in name for p in
                      ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")) \
                  and "o_proj" not in name
        if adaptive and is_main:
            li = layer_index_of(name)
            if li is not None and li in khigh_map:
                layer_high_length = khigh_map[li]
                layer_low_length = 0
                layer_groupsize = -1
```
(Place this AFTER the existing per-layer defaults are set and BEFORE the `qlayers[name].quantizer.configure(...)` call, so it overrides them.)

At the `fuse_basis_to_model(...)` call site, thread the flag:
```python
    fuse_basis_to_model(model, basis_path, rotation_path,
                        high_fraction, low_fraction=low_fraction,
                        identity_main_rotation=qcfg.get('adaptive_high_fraction', False))
```

- [ ] **Step 4: Run the unit test + a config-load smoke check**

Run: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -m pytest tests/test_adaptive.py::test_layer_index_of -v`
Expected: PASS.

Smoke: `docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -c "import torch; from promix.eval.ptq import main_path_khigh_map; E=torch.load('rotation/E-wikitext-512-Llama-3.2-1B-Instruct.bin'); m=main_path_khigh_map(E,0.85,16); print(sorted(set(m.values())), 'layers:', len(m))"`
Expected: prints a list of per-layer K_high (multiples of 64) and `layers: 16`.

- [ ] **Step 5: Commit**

```bash
git add promix/eval/ptq.py tests/test_adaptive.py
git commit -m "feat(ptq): per-layer adaptive K_high plumbing + identity-R1 wiring"
```

---

### Task 5: Adaptive config + Pareto sweep (experiment)

**Files:**
- Create: `promix/configs/llama-3.2-1b-fp-adaptive.yaml`
- Create: `promix/scripts/run_adaptive_sweep.sh` (thin loop over τ)

**Interfaces:**
- Consumes: everything above; produces (avg_bits, PPL) points for τ ∈ {0.6, 0.7, 0.8, 0.85, 0.9, 0.95}.

- [ ] **Step 1: Create the adaptive config**

```yaml
# promix/configs/llama-3.2-1b-fp-adaptive.yaml
# Copy of llama-3.2-1b-fp-perhead.yaml with adaptive knobs added.
# (same model / quantize block as fp-perhead, plus:)
#   adaptive_high_fraction: true
#   energy_threshold: 0.85        # overridden per sweep run via a tiny sed
# eval tasks: [] (PPL only, for speed during the sweep)
```
Create by copying the perhead config and adding the two knobs + `tasks: []`:
```bash
docker exec -w /home/luofan/plaquant luofan-plaquant bash -c "sed -e 's/o_proj_pca: .*/o_proj_pca: \"per_head\"/' promix/configs/llama-3.2-1b-fp-perhead.yaml > promix/configs/llama-3.2-1b-fp-adaptive.yaml"
```
Then append `adaptive_high_fraction: true` and `energy_threshold: 0.85` under `quantize:` and set `tasks: []` (edit the file directly).

- [ ] **Step 2: Run the sweep**

For each τ, set `energy_threshold`, run Step 2 (PPL only), record PPL and avg_bits:
```bash
for TAU in 0.6 0.7 0.8 0.85 0.9 0.95; do
  docker exec -w /home/luofan/plaquant luofan-plaquant \
    sed -i "s/energy_threshold: .*/energy_threshold: $TAU/" promix/configs/llama-3.2-1b-fp-adaptive.yaml
  docker exec -d -w /home/luofan/plaquant -e CUDA_VISIBLE_DEVICES=7 -e HF_HOME=/home/luofan/hf_cache -e PYTHONPATH=/home/luofan/plaquant -e PYTHONUNBUFFERED=1 luofan-plaquant \
    bash -c "torchrun --nproc_per_node=1 --master_port=295$RANDOM promix/eval/ptq.py --config promix/configs/llama-3.2-1b-fp-adaptive.yaml > /home/luofan/adapt_$TAU.log 2>&1"
  # wait for completion, then grep PPL (serialize; one GPU)
done
```
(Run serially on one GPU, or fan out across GPUs 4/5/7. Record each PPL via `grep Perplexity`.)

- [ ] **Step 3: Compute avg_bits per τ and assemble the Pareto table**

For each τ, print the main-path avg bit-width:
```bash
docker exec -w /home/luofan/plaquant -e PYTHONPATH=/home/luofan/plaquant luofan-plaquant python -c "
import torch
from promix.eval.ptq import main_path_khigh_map
from promix.quantize.adaptive import main_path_avg_bits
E=torch.load('rotation/E-wikitext-512-Llama-3.2-1B-Instruct.bin')
for tau in [0.6,0.7,0.8,0.85,0.9,0.95]:
    m=main_path_khigh_map(E,tau,16)
    print(tau, round(main_path_avg_bits([m[i] for i in range(16)],2048),3))
"
```

- [ ] **Step 4: Assemble and interpret the Pareto curve**

Tabulate (τ, avg_bits, PPL) for adaptive and (fraction, avg_bits, PPL) for fixed {1/8, 1/4, 1/2} under the SAME main-path-only avg-bits accounting. State whether adaptive dominates (lower PPL at matched bits). A flat/overlapping result is a valid negative finding (see spec §7).

- [ ] **Step 5: Record results and commit**

Append the Pareto table + interpretation to the spec's results section (or a short `docs/superpowers/specs/2026-10-09-adaptive-high-fraction-results.md`), then:
```bash
git add promix/configs/llama-3.2-1b-fp-adaptive.yaml promix/scripts/run_adaptive_sweep.sh docs/superpowers/specs/2026-10-09-adaptive-high-fraction-results.md
git commit -m "exp(adaptive): Pareto sweep config, script, and results"
```

---

## Self-Review Notes

- **Spec coverage:** §1 Step-0 → Task 2; §2 K-derivation → Task 1; §3 identity R1 → Task 3; §4 ptq plumbing → Task 4; §5 avg-bits → Task 1 (`main_path_avg_bits`); §6 protocol + §7 success → Task 5. All sections mapped.
- **Alignment:** `align=64` everywhere for 1B (Global Constraints); tests use `align=4` only on tiny synthetic curves.
- **No regression:** adaptive gates behind `adaptive_high_fraction` (default false); fixed path untouched.
- **Known cost:** Task 2 Step 4 re-runs covariance collection (~16 min) because per-layer variance needs `H_attn/H_mlp`; flagged inline.
