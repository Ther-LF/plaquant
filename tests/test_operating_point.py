import torch

from promix.quantize.adaptive import (
    allocate_by_budget,
    select_operating_point,
)


def _dd(n_layers=2, rings_per_layer=40):
    """Distinct descending deltaD per layer so allocation order is deterministic.
    The mock PPL functions below depend only on the FP8 ring COUNT, so exact
    values are irrelevant to the mode logic -- only the total ring count and the
    per-budget n_fp8 matter."""
    return [torch.linspace(1.0, 0.1, rings_per_layer) + 0.01 * li
            for li in range(n_layers)]
# total_rings = 80, feasible cap max_fp8 = 80 // 8 = 10


# ---- perf mode: reuses Task-2 budget logic -------------------------------

def test_perf_matches_allocate_by_budget():
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='perf', target=0.125, eval_ppl_fn=None, floor=None)
    expected = allocate_by_budget(dd, 0.125)
    assert [m.tolist() for m in masks] == [m.tolist() for m in expected]
    assert info['n_fp8'] == 10          # 0.125 * 80
    assert info['mode'] == 'perf'


def test_perf_clips_to_feasible_budget():
    # target budget 0.5 would be 40 FP8 rings; feasible region caps avg FP8 <= 1/8.
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='perf', target=0.5, eval_ppl_fn=None, floor=None)
    total = sum(m.numel() for m in masks)
    assert sum(int(m.sum()) for m in masks) == total // 8   # clipped to 10
    assert info['n_fp8'] == 10


# ---- accuracy mode: fewest FP8 meeting an acceptable PPL -----------------

def _mock_ppl(masks):
    n = sum(int(m.sum()) for m in masks)
    return 10.0 - 0.5 * n            # monotone: more FP8 -> lower PPL


def test_accuracy_returns_fewest_fp8_meeting_target():
    dd = _dd()
    # need 10 - 0.5n <= 7.0  ->  n >= 6  -> cheapest is 6
    masks, info = select_operating_point(
        dd, None, mode='accuracy', target=7.0, eval_ppl_fn=_mock_ppl, floor=7.0)
    assert info['n_fp8'] == 6
    assert _mock_ppl(masks) <= 7.0 + 1e-9
    assert info['feasible'] is True


def test_accuracy_clipped_by_ppl_floor():
    # loose acceptable target (9.9 -> n>=1) but strict floor 7.0 forces n>=6.
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='accuracy', target=9.9, eval_ppl_fn=_mock_ppl, floor=7.0)
    assert info['n_fp8'] == 6           # bumped up to satisfy PPL <= floor
    assert _mock_ppl(masks) <= 7.0 + 1e-9


def test_accuracy_infeasible_target_clipped_to_budget_cap():
    # min reachable PPL at feasible cap (n=10) is 5.0; target 4.0 is unreachable.
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='accuracy', target=4.0, eval_ppl_fn=_mock_ppl, floor=4.0)
    assert info['n_fp8'] == 10          # avg FP8 <= 1/8 cap
    assert info['feasible'] is False


# ---- knee mode: max-curvature interior point -----------------------------

def _mock_ppl_knee(masks):
    # Steep improvement for the first 3 rings, then flattens (concave).
    n = sum(int(m.sum()) for m in masks)
    return 10.0 - (1.0 * min(n, 3) + 0.1 * max(n - 3, 0))


def test_knee_returns_interior_point():
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='knee', target=7.0, eval_ppl_fn=_mock_ppl_knee, floor=10.0)
    n = info['n_fp8']
    assert 0 < n < 10                   # interior
    assert n == 3                       # curvature collapse after ring 3
    assert info.get('fallback') is False


def test_knee_falls_back_to_accuracy_when_no_clean_knee():
    # Linear PPL -> constant slope -> no curvature -> fall back to accuracy(target).
    dd = _dd()
    masks, info = select_operating_point(
        dd, None, mode='knee', target=7.0, eval_ppl_fn=_mock_ppl, floor=7.0)
    assert info.get('fallback') is True
    assert info['n_fp8'] == 6           # accuracy fallback target 7.0 -> n=6
