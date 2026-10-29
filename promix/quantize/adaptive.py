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


# ---------------------------------------------------------------------------
# Rate-distortion mixed-precision allocation
# (docs/superpowers/plans/2026-10-28-rate-distortion-mixed-precision-allocation.md)
# ---------------------------------------------------------------------------
from .quant_utils import fake_quantize_nvfp4, fake_quantize_mxfp8

_FMT = {'nvfp4': fake_quantize_nvfp4, 'mxfp8': fake_quantize_mxfp8}


def ring_mse(x_rot, ring_size, fmt):
    """Per-ring measured quantization MSE on rotated activations/weights.

    Uses the deployment pipeline's real fake-quant (the same MSE-searched
    block scales the kernel sees), so the measured MSE matches what the
    model is actually served with. `fake_quantize_nvfp4`/`_mxfp8` quantize
    along the last dim with their own block sizes (16 / 32) and derive
    block scales internally; `ring_size` (64) is a whole multiple of both,
    so per-ring MSE is simply the mean squared dequant error over each
    64-channel group.

    x_rot: (M, d). Returns (d//ring_size,) float tensor.
    """
    if fmt not in _FMT:
        raise ValueError(f"fmt must be one of {list(_FMT)}, got {fmt}")
    x = torch.as_tensor(x_rot, dtype=torch.float32)
    M, d = x.shape
    if d % ring_size != 0:
        raise ValueError(f"d={d} not a multiple of ring_size={ring_size}")
    q = _FMT[fmt](x)                       # dequantized, same shape
    err2 = (x - q).pow(2)                  # (M, d)
    per_ring = err2.reshape(M, d // ring_size, ring_size).mean(dim=(0, 2))
    return per_ring                        # (d//ring_size,)


def ring_deltaD(x_rot, ring_size):
    """D_fp4 - D_fp8 per ring (error reduced by upgrading FP4 -> FP8)."""
    return ring_mse(x_rot, ring_size, 'nvfp4') - ring_mse(x_rot, ring_size, 'mxfp8')


# ---------------------------------------------------------------------------
# Loss-weighted rate-distortion score (spec §9 fallback / §4.5 refinement).
#
# Pure ring_deltaD ranks rings by LOCAL activation-MSE reduction, implicitly
# assuming every channel's error contributes equally to the final LM loss. It
# doesn't: a channel feeding a loss-sensitive path matters more than one whose
# error the network absorbs. The refinement weights each ring's ΔD by its
# sensitivity to the final cross-entropy loss, estimated with an empirical
# Fisher-diagonal proxy of the (rotated) activation entering each Linear:
#
#     F_c = (1/T) * sum_t (dL/da_c)^2          # per-channel, over T calib tokens
#     w_j = sum_{c in ring j} F_c              # per-ring loss-sensitivity weight
#     score_j = w_j * ΔD_j
#
# First-order Taylor: upgrading ring j FP4->FP8 reduces the activation error in
# its channels by ~ΔD_j (MSE), and the induced change in loss is, to leading
# order, sum_{c in ring} F_c * (per-channel error reduction) ≈ w_j * ΔD_j (ΔD_j
# is the ring-mean per-channel MSE reduction, w_j the ring-summed Fisher). So the
# product is the Fisher-proxy estimate of the LM-loss reduction from the upgrade,
# which is the quantity the allocation should actually be maximizing.
#
# ZERO-REGRESSION: these are additive, opt-in helpers. The pure-ΔD path
# (ring_deltaD -> allocate_by_budget) is untouched.
# ---------------------------------------------------------------------------
def ring_fisher_weight(grad_sq_sum, ring_size):
    """Per-ring loss-sensitivity weight from accumulated per-channel squared
    gradients of the LM loss w.r.t. the activation entering a Linear.

    grad_sq_sum: (d,) tensor = sum_t (dL/da_c)^2 over calibration tokens
                 (the un-normalized empirical Fisher diagonal; the 1/T factor
                 is a global constant that does not affect ring ranking).
    Returns (d//ring_size,): the Fisher summed over each ring's channels.
    """
    g = torch.as_tensor(grad_sq_sum, dtype=torch.float64).flatten().clamp_min(0)
    d = g.numel()
    if d % ring_size != 0:
        raise ValueError(f"d={d} not a multiple of ring_size={ring_size}")
    return g.reshape(d // ring_size, ring_size).sum(dim=1)


def ring_deltaD_weighted(deltaD, fisher_weight, normalize=False, eps=1e-12):
    """Loss-weighted rate-distortion score: w_j * ΔD_j.

    deltaD:       per-ring ΔD = D_fp4 - D_fp8 (local MSE reduction).
    fisher_weight: per-ring Fisher sum w_j from ring_fisher_weight.
    normalize:    if True, divide w by its mean over THESE rings before the
                  product (cosmetic only — a per-call positive rescale does not
                  change this layer's internal ring order; and because
                  allocate_by_budget ranks ALL rings globally, per-layer rescaling
                  WOULD alter cross-layer order, so the allocation driver keeps
                  normalize=False and ranks on the raw, globally-comparable
                  w_j*ΔD_j. The flag is provided for inspection/reporting.).

    The raw Fisher is comparable across layers (same loss, same tokens), so a ring
    in a loss-sensitive layer can outrank a higher-ΔD ring in an insensitive one —
    which is the whole point of the refinement.
    """
    dd = torch.as_tensor(deltaD, dtype=torch.float64).flatten()
    w = torch.as_tensor(fisher_weight, dtype=torch.float64).flatten().clamp_min(0)
    if dd.numel() != w.numel():
        raise ValueError(f"deltaD ({dd.numel()}) and fisher_weight ({w.numel()}) "
                         f"must have the same number of rings")
    if normalize:
        w = w / w.mean().clamp_min(eps)
    return w * dd


def allocate_by_budget(deltaD_per_layer, target_mean_fp8, deltaC_per_layer=None):
    """Global single-threshold rate-distortion allocation.

    Rank all rings by deltaD/deltaC (deltaC=None -> equal cost -> by deltaD),
    upgrade the top ones until the average FP8 ring-fraction == target_mean_fp8.
    Returns one bool mask per layer (True = FP8).
    """
    flat = []  # (score, layer_idx, ring_idx)
    for li, dd in enumerate(deltaD_per_layer):
        dd = torch.as_tensor(dd, dtype=torch.float64)
        if deltaC_per_layer is not None:
            dc = torch.as_tensor(deltaC_per_layer[li], dtype=torch.float64).clamp_min(1e-12)
            score = dd / dc
        else:
            score = dd
        for ri in range(score.numel()):
            flat.append((float(score[ri]), li, ri))
    total_rings = len(flat)
    n_fp8 = int(round(target_mean_fp8 * total_rings))
    flat.sort(key=lambda t: t[0], reverse=True)
    masks = [torch.zeros(len(torch.as_tensor(dd).flatten()), dtype=torch.bool)
             for dd in deltaD_per_layer]
    for _, li, ri in flat[:n_fp8]:
        masks[li][ri] = True
    return masks


def masks_to_khigh(masks, ring_size):
    """FP8 ring masks -> per-layer K_high (= #FP8 rings * ring_size)."""
    return [int(m.sum()) * ring_size for m in masks]


# ---------------------------------------------------------------------------
# Config plumbing (plan Task 6 Step 1): opt-in `rd_alloc` block reader.
# ---------------------------------------------------------------------------
_RD_ALLOC_DEFAULTS = {'enabled': False, 'ring_size': 64, 'mode': 'perf',
                      'target': 0.125}


def rd_alloc_config(config):
    """Read the opt-in ``rd_alloc`` block from a loaded config dict.

    Returns a dict with all defaults filled:
      ``{'enabled': bool, 'ring_size': int, 'mode': str, 'target': float}``.

    ZERO-REGRESSION CONTRACT (plan Global Constraints): when the block is
    absent OR ``enabled: false`` (the shipped default), this returns
    ``enabled=False``. Callers MUST then install NO ``model._rd_khigh_override``,
    so ``configure_quantizers`` takes the byte-identical uniform
    ``high_fraction`` split. Only ``enabled: true`` opts a config into the
    rate-distortion per-Linear K_high allocation.
    """
    raw = (config or {}).get('rd_alloc') or {}
    out = dict(_RD_ALLOC_DEFAULTS)
    for k in out:
        if k in raw and raw[k] is not None:
            out[k] = raw[k]
    out['enabled'] = bool(out['enabled'])
    out['ring_size'] = int(out['ring_size'])
    out['target'] = float(out['target'])
    out['mode'] = str(out['mode'])
    return out


# ---------------------------------------------------------------------------
# Operating-point selection (spec section 8): one ranked curve, three modes.
# Purely algorithmic -- eval_ppl_fn is injected (no model, no cost_model import)
# so it stays unit-testable. All results are clipped to the Global-Constraints
# feasible region: PPL <= floor AND average FP8 ring-fraction <= 1/8.
# ---------------------------------------------------------------------------
_MAX_MEAN_FP8 = 1.0 / 8  # low-precision floor: avg FP8 ring-fraction <= 1/8


def select_operating_point(deltaD_per_layer, deltaC_per_layer, mode,
                           target, eval_ppl_fn=None, floor=None):
    """Pick an operating point on the single-threshold rate-distortion curve.

    deltaD_per_layer:  list of per-ring deltaD tensors (one per layer).
    deltaC_per_layer:  optional per-ring cost (None -> equal-cost, rank by deltaD).
    mode:              'perf' | 'accuracy' | 'knee'.
    target:            perf -> avg FP8 budget; accuracy -> acceptable PPL;
                       knee -> acceptable PPL used only for fallback.
    eval_ppl_fn:       masks -> PPL (monotone-decreasing in #FP8 rings).
    floor:             hard PPL ceiling for the feasible region.

    Returns (masks, info). ``info`` carries mode, n_fp8, avg_fp8, feasible, and
    (where evaluated) ppl / fallback.
    """
    counts = [int(torch.as_tensor(dd).flatten().numel()) for dd in deltaD_per_layer]
    total_rings = sum(counts)
    max_fp8 = int(total_rings * _MAX_MEAN_FP8)  # avg FP8 <= 1/8 feasibility cap

    def masks_for(n):
        n = max(0, min(int(n), total_rings))
        frac = (n / total_rings) if total_rings else 0.0
        return allocate_by_budget(deltaD_per_layer, frac, deltaC_per_layer)

    def n_of(masks):
        return sum(int(m.sum()) for m in masks)

    def n_floor_min():
        """Smallest feasible n whose PPL <= floor (monotone => a threshold).
        Returns max_fp8 if even the feasible cap cannot reach the floor."""
        if eval_ppl_fn is None or floor is None:
            return 0
        for n in range(0, max_fp8 + 1):
            if eval_ppl_fn(masks_for(n)) <= floor:
                return n
        return max_fp8

    def _accuracy_n(acc_target):
        """Fewest FP8 rings meeting acc_target, then clipped to [n_floor, max_fp8].
        Returns (n, feasible)."""
        chosen, met = None, False
        for n in range(0, max_fp8 + 1):
            if eval_ppl_fn(masks_for(n)) <= acc_target:
                chosen, met = n, True
                break
        if chosen is None:
            chosen = max_fp8
        chosen = max(chosen, n_floor_min())      # enforce PPL <= floor
        chosen = min(chosen, max_fp8)            # enforce avg FP8 <= 1/8
        feasible = met and (floor is None or eval_ppl_fn is None
                            or eval_ppl_fn(masks_for(chosen)) <= floor)
        return chosen, feasible

    info = {'mode': mode, 'total_rings': total_rings, 'max_fp8': max_fp8}

    if mode == 'perf':
        budget_n = int(round(target * total_rings))
        n = max(n_floor_min(), min(budget_n, max_fp8))   # clip to feasible region
        masks = masks_for(n)
        info['target_budget'] = target

    elif mode == 'accuracy':
        n, feasible = _accuracy_n(target)
        masks = masks_for(n)
        info['target_ppl'] = target
        info['feasible'] = feasible

    elif mode == 'knee':
        ppls = [eval_ppl_fn(masks_for(n)) for n in range(0, max_fp8 + 1)]
        # improvement[i] = PPL drop from adding the (i+1)-th FP8 ring (cost +1).
        improvements = [ppls[i] - ppls[i + 1] for i in range(len(ppls) - 1)]
        knee_n = None
        for i in range(1, len(improvements)):
            prev = improvements[i - 1]
            if prev > 1e-12 and improvements[i] / prev < 0.5:
                knee_n = i          # marginal gain collapses at ring i+1 -> keep i
                break
        if knee_n is None:
            # No clean knee -> fall back to accuracy(target); target=None -> floor.
            acc_target = target if target is not None else floor
            if acc_target is None:
                n = max_fp8
                feasible = False
            else:
                n, feasible = _accuracy_n(acc_target)
            info['fallback'] = True
            info['feasible'] = feasible
        else:
            n = max(n_floor_min(), min(knee_n, max_fp8))  # clip to feasible region
            info['fallback'] = False
        masks = masks_for(n)

    else:
        raise ValueError(
            f"mode must be one of 'perf', 'accuracy', 'knee', got {mode!r}")

    actual = n_of(masks)
    info['n_fp8'] = actual
    info['avg_fp8'] = (actual / total_rings) if total_rings else 0.0
    if eval_ppl_fn is not None:
        info['ppl'] = eval_ppl_fn(masks)
    if 'feasible' not in info:
        ok_budget = info['avg_fp8'] <= _MAX_MEAN_FP8 + 1e-12
        ok_ppl = (floor is None or eval_ppl_fn is None
                  or info.get('ppl', float('-inf')) <= floor)
        info['feasible'] = ok_budget and ok_ppl
    return masks, info
