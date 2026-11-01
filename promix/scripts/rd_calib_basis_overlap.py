"""rd-calib: PCA-basis stability across calibration sets (CPU-only analysis).

Compares two Step-0 basis files (e.g. U-wikitext-512-* vs U-c4-512-* for the
same model) and quantifies how stable the learned PCA subspaces are:

  * top-1/8 subspace overlap: for each matched basis matrix, take the top-k
    eigenvector block (k = round(frac*d); eigen_decompose sorts ASCENDING, so
    the top-variance directions are the LAST k columns) and compute
        overlap = ||Ua_topᵀ Ub_top||_F² / k  ∈ [0, 1]
    (the mean squared cosine of the principal angles between the two
    subspaces; 1 = identical span, random-baseline = k/d = frac).
  * principal-angle spectrum summary (min/median/max cosine, #cos>0.9).

Covered keys (full_shared basis layout):
  - "attn_mlp"                 : the shared hidden-dim U (drives q/k/v/o/gate/up)
  - "layer.{i}.mlp.down_proj"  : per-layer intermediate-dim U (aggregated)
  - "layer.{i}.self_attn.value": per-head U (aggregated over layers+heads)

Usage:
  python promix/scripts/rd_calib_basis_overlap.py \
      --basis_a rotation/U-wikitext-512-Llama-3.2-3B.bin \
      --basis_b rotation/U-c4-512-Llama-3.2-3B.bin --frac 0.125
"""
import argparse
import json

import torch


def top_block(U, frac):
    """Last-k columns (ascending eigenvalue sort -> top variance at the end)."""
    d = U.shape[-1]
    k = int(round(frac * d))
    return U[..., :, d - k:], k, d


def subspace_stats(Ua, Ub, frac):
    """Overlap + principal-angle cosines between the top-frac blocks."""
    A, k, d = top_block(Ua.to(torch.float64), frac)
    B, kb, db = top_block(Ub.to(torch.float64), frac)
    assert (k, d) == (kb, db), f"shape mismatch {(k, d)} vs {(kb, db)}"
    M = A.transpose(-2, -1) @ B
    cos = torch.linalg.svdvals(M).clamp(0, 1)          # principal-angle cosines
    overlap = float((cos ** 2).sum() / k)
    return dict(k=k, d=d, overlap=overlap, random_baseline=k / d,
                cos_min=float(cos.min()), cos_median=float(cos.median()),
                cos_max=float(cos.max()),
                n_cos_gt_0p9=int((cos > 0.9).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_a", required=True)
    ap.add_argument("--basis_b", required=True)
    ap.add_argument("--frac", type=float, default=0.125)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    da = torch.load(args.basis_a, map_location="cpu")
    db = torch.load(args.basis_b, map_location="cpu")
    assert da.get("config") == db.get("config"), \
        f"granularity mismatch: {da.get('config')} vs {db.get('config')}"

    results = {"meta": dict(basis_a=args.basis_a, basis_b=args.basis_b,
                            frac=args.frac, config=da.get("config"))}

    # shared hidden-dim basis
    if "attn_mlp" in da:
        results["attn_mlp"] = subspace_stats(da["attn_mlp"], db["attn_mlp"],
                                             args.frac)
        r = results["attn_mlp"]
        print(f"attn_mlp (shared, d={r['d']}, k={r['k']}): "
              f"overlap={r['overlap']:.4f} (random {r['random_baseline']:.3f}) "
              f"cos min/med/max={r['cos_min']:.3f}/{r['cos_median']:.3f}/"
              f"{r['cos_max']:.3f}", flush=True)

    # per-layer down_proj
    dp_keys = sorted(k for k in da if k.endswith(".mlp.down_proj"))
    if dp_keys:
        per = {}
        for key in dp_keys:
            per[key] = subspace_stats(da[key], db[key], args.frac)
        ovs = [per[k]["overlap"] for k in dp_keys]
        results["down_proj"] = dict(
            mean_overlap=sum(ovs) / len(ovs), min_overlap=min(ovs),
            max_overlap=max(ovs), n_layers=len(ovs),
            random_baseline=per[dp_keys[0]]["random_baseline"], per_layer=per)
        print(f"down_proj ({len(ovs)} layers): mean overlap="
              f"{results['down_proj']['mean_overlap']:.4f} "
              f"min={min(ovs):.4f} max={max(ovs):.4f} "
              f"(random {results['down_proj']['random_baseline']:.3f})",
              flush=True)

    # per-head value basis (stacked [heads, hd, hd])
    v_keys = sorted(k for k in da if k.endswith(".self_attn.value"))
    if v_keys:
        ovs = []
        for key in v_keys:
            Ua, Ub = da[key], db[key]
            for h in range(Ua.shape[0]):
                ovs.append(subspace_stats(Ua[h], Ub[h], args.frac)["overlap"])
        results["value_per_head"] = dict(
            mean_overlap=sum(ovs) / len(ovs), min_overlap=min(ovs),
            max_overlap=max(ovs), n=len(ovs))
        print(f"value per-head ({len(ovs)} head-blocks): mean overlap="
              f"{results['value_per_head']['mean_overlap']:.4f} "
              f"min={min(ovs):.4f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"WROTE {args.out}", flush=True)
    print("RD_CALIB_BASIS_OVERLAP_DONE", flush=True)


if __name__ == "__main__":
    main()
