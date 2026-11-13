# AdaMix Motivation — experiment deliverables

CPU-only motivation evidence for the AdaMix (ProMix) paper. This directory holds
the inventory audit, the E1a local controlled representation x precision
diagnostic, and the E2 allocation math (no PPL). GPU experiments (E1b full-model
PPL, E2-PTQ, E4 kernel timing) are deferred — all Pro6000 cards were occupied.

See the task book: `outputs/adamix-motivation-experiments.md`.

## Layout

```
inventory.md                 # §2.1/§8 audit: commit, paths+hashes, devices, reusable data, gaps
protocol.yaml                # fixed config, formats, splits, allocation policies
runs/e1a_3b/                 # E1a 3B results (q/k/v/gate/up shared-PCA + down local-PCA)
  manifest.json              #   exact params, hashes, caveats, wall time
  metrics.json / summary.csv #   per-(layer,module,rep,split,policy) MSE/clip metrics
  group_stats.csv            #   per-group variance/maxabs/rms/mse_fp4/mse_fp8/signed_delta (STABLE gid)
  allocation.json            #   per-(module,rep) k_l + selected fp8 group ids
runs/e2math_3b/              # E2 allocation math (rep frozen = R3)
  e2_alloc_summary.json      #   Spearman(var,delta), A/B/C/D overlaps, byte accounting
  allocation.csv             #   per-(policy,module,group) selection
report.md                    # per-question reading (§9)
```

## Reproduce (CPU-only; all jobs set CUDA_VISIBLE_DEVICES="")

E1a (3B, ~30-55 min on 96 CPU cores; includes down_proj local-PCA phase):

```bash
cd /home/luofan/plaquant-adamix
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/luofan/hf_cache \
       CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 OMP_NUM_THREADS=96
python3 promix/scripts/adamix_e1a.py \
    --model /home/luofan/models/Llama-3.2-3B --short Llama-3.2-3B \
    --basis rotation/U-wikitext-512-Llama-3.2-3B.bin \
    --G 512 --max_tokens 2048 --fit_samples 1 --diag_samples 1 --include_down \
    --out_dir docs/adamix_motivation/runs/e1a_3b
```

E2 allocation math (reads the E1a run dir; seconds):

```bash
python3 promix/scripts/adamix_e2_alloc_math.py \
    --run_dir docs/adamix_motivation/runs/e1a_3b --rep R3 \
    --out_dir docs/adamix_motivation/runs/e2math_3b
```

8B is a direct follow-up: same commands with the 8B checkpoint/basis (CPU-doable,
~2.5x slower). E1b/E2-PTQ/E4 require a free GPU.
