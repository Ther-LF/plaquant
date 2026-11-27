# M31-ext report — format-aware cross-layer criterion (DD) + densified low-budget grid

Protocol: docs/paper_motivation_3_1_ext_experiments.md; runner promix/scripts/rd_m31.py (ext mode). All new artifacts live in docs/motivation_3_1_ext/; legacy docs/motivation_3_1/ untouched.

## Gate 1 — endpoint consistency (new runner vs legacy, tol 1e-4)

| model | r | d_wiki | d_c4 | d_E_mean | pass |
|---|---|---|---|---|---|
| 3b | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| 3b | 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| 8b | 0 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| 8b | 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |

Gate 1: **PASS**

## Gate 2 — exact quota conservation (H identical across strategies/scopes at each (model, r))

Gate 2 over 30 (model, r) groups (raw-representation runs, legacy+ext): **PASS**

## Gate 3 — no NaN / completeness

- ext runs: 120; with NaN layer errors: 0 
- non-ok ext runs: 0 
- failures.csv rows: 0
Gate 3: **PASS**

## DD score diagnostics

- 3b: negative-score channel fraction overall = 0.0080% (signed scores kept; per-module max = 0.0977%)
- 8b: negative-score channel fraction overall = 0.0096% (signed scores kept; per-module max = 0.1883%)

## Llama-3.2-3B — WikiText-2 PPL / C4 PPL / act_mse_global / E_mean

Endpoints: pure NVFP4 (r=0) wiki 14.3309, c4 23.1306, act_mse 0.0048626; pure MXFP8 (r=1) wiki 7.8537, c4 10.5024, act_mse 0.000459247

| r | metric | rnd within | ma within | ss within | dd within | ma across | ss across | dd across |
|---|---|---|---|---|---|---|---|---|
| 0.02 | wiki | 10.4853 (n=3) | 8.2225 | 8.2414 | 8.2744 | 17.6073 | 8.2276 | 17.6064 |
| 0.02 | c4 | 15.5209 (n=3) | 11.2640 | 11.2776 | 11.3523 | 28.1076 | 11.2447 | 28.3190 |
| 0.02 | act_mse | 4.773e-03 (n=3) | 3.736e-03 | 3.735e-03 | 4.503e-03 | 3.557e-03 | 3.557e-03 | 4.352e-03 |
| 0.02 | emean | 6.754e-03 (n=3) | 5.788e-03 | 5.785e-03 | 6.486e-03 | 5.905e-03 | 5.820e-03 | 6.530e-03 |
| 0.04 | wiki | 15.2073 (n=3) | 8.1990 | 8.1813 | 8.2273 | 16.9599 | 8.1895 | 18.3320 |
| 0.04 | c4 | 22.6177 (n=3) | 11.2034 | 11.1873 | 11.2959 | 24.7010 | 11.1786 | 27.0617 |
| 0.04 | act_mse | 4.678e-03 (n=3) | 3.431e-03 | 3.430e-03 | 4.214e-03 | 3.222e-03 | 3.222e-03 | 4.000e-03 |
| 0.04 | emean | 6.622e-03 (n=3) | 5.417e-03 | 5.414e-03 | 6.145e-03 | 5.587e-03 | 5.499e-03 | 6.203e-03 |
| 0.06 | wiki | 12.0946 (n=3) | 8.1618 | 8.1656 | 8.2051 | 17.0195 | 8.1734 | 17.1388 |
| 0.06 | c4 | 17.7467 (n=3) | 11.1357 | 11.1458 | 11.2373 | 26.7287 | 11.1302 | 26.8598 |
| 0.06 | act_mse | 4.583e-03 (n=3) | 3.225e-03 | 3.225e-03 | 4.004e-03 | 3.015e-03 | 3.015e-03 | 3.740e-03 |
| 0.06 | emean | 6.439e-03 (n=3) | 5.147e-03 | 5.143e-03 | 5.884e-03 | 5.371e-03 | 5.283e-03 | 5.939e-03 |
| 0.08 | wiki | 11.1536 (n=3) | 8.1432 | 8.1434 | 8.1884 | 16.2676 | 8.1624 | 18.6405 |
| 0.08 | c4 | 15.9931 (n=3) | 11.0969 | 11.1057 | 11.2100 | 25.7569 | 11.1201 | 28.2226 |
| 0.08 | act_mse | 4.493e-03 (n=3) | 3.075e-03 | 3.074e-03 | 3.802e-03 | 2.865e-03 | 2.864e-03 | 3.517e-03 |
| 0.08 | emean | 6.282e-03 (n=3) | 4.934e-03 | 4.928e-03 | 5.628e-03 | 5.205e-03 | 5.114e-03 | 5.733e-03 |
| 0.1 | wiki | 13.0362 (n=3) | 8.1243 | 8.1223 | 8.1760 | 16.6165 | 8.1434 | 14.7981 |
| 0.1 | c4 | 18.3564 (n=3) | 11.0691 | 11.0656 | 11.1870 | 24.9669 | 11.0922 | 25.7949 |
| 0.1 | act_mse | 4.401e-03 (n=3) | 2.958e-03 | 2.957e-03 | 3.656e-03 | 2.733e-03 | 2.732e-03 | 3.299e-03 |
| 0.1 | emean | 6.140e-03 (n=3) | 4.743e-03 | 4.739e-03 | 5.412e-03 | 5.059e-03 | 4.968e-03 | 5.493e-03 |
| 0.2 | wiki | 11.3552 (n=3) | 8.0828 | 8.0807 | 8.1268 | 18.1483 | 8.1121 | 17.8318 |
| 0.2 | c4 | 15.7583 (n=3) | 10.9716 | 10.9975 | 11.0622 | 26.0680 | 11.0215 | 26.4531 |
| 0.2 | act_mse | 3.985e-03 (n=3) | 2.592e-03 | 2.591e-03 | 3.050e-03 | 2.257e-03 | 2.256e-03 | 2.523e-03 |
| 0.2 | emean | 5.468e-03 (n=3) | 4.104e-03 | 4.103e-03 | 4.585e-03 | 4.513e-03 | 4.425e-03 | 4.699e-03 |
| 0.5 | wiki | 8.0293 (n=3) | 7.9789 | 7.9810 | 7.9772 | 15.6947 | 8.0176 | 16.3202 |
| 0.5 | c4 | 10.8868 (n=3) | 10.7714 | 10.7649 | 10.7828 | 25.2421 | 10.8219 | 24.6158 |
| 0.5 | act_mse | 2.655e-03 (n=3) | 1.714e-03 | 1.713e-03 | 1.863e-03 | 1.019e-03 | 1.019e-03 | 1.110e-03 |
| 0.5 | emean | 3.510e-03 (n=3) | 2.587e-03 | 2.574e-03 | 2.776e-03 | 3.016e-03 | 2.956e-03 | 3.084e-03 |

## Llama-3-8B — WikiText-2 PPL / C4 PPL / act_mse_global / E_mean

Endpoints: pure NVFP4 (r=0) wiki 7.7818, c4 10.5596, act_mse 0.00436966; pure MXFP8 (r=1) wiki 6.1835, c4 8.9410, act_mse 0.000412306

| r | metric | rnd within | ma within | ss within | dd within | ma across | ss across | dd across |
|---|---|---|---|---|---|---|---|---|
| 0.02 | wiki | 7.4067 (n=3) | 6.5361 | 6.5280 | 6.5808 | 7.2783 | 6.5333 | 7.4313 |
| 0.02 | c4 | 10.5342 (n=3) | 9.5493 | 9.5353 | 9.5887 | 10.1739 | 9.5148 | 10.3204 |
| 0.02 | act_mse | 4.278e-03 (n=3) | 3.346e-03 | 3.346e-03 | 3.904e-03 | 3.198e-03 | 3.198e-03 | 3.779e-03 |
| 0.02 | emean | 6.987e-03 (n=3) | 5.998e-03 | 5.997e-03 | 6.597e-03 | 6.007e-03 | 5.997e-03 | 6.556e-03 |
| 0.04 | wiki | 7.7356 (n=3) | 6.5187 | 6.5265 | 6.5662 | 7.0490 | 6.5145 | 7.6789 |
| 0.04 | c4 | 11.1923 (n=3) | 9.4997 | 9.5108 | 9.5773 | 10.2483 | 9.4964 | 10.5957 |
| 0.04 | act_mse | 4.217e-03 (n=3) | 3.185e-03 | 3.184e-03 | 3.776e-03 | 3.005e-03 | 3.004e-03 | 3.563e-03 |
| 0.04 | emean | 6.853e-03 (n=3) | 5.718e-03 | 5.717e-03 | 6.381e-03 | 5.766e-03 | 5.755e-03 | 6.304e-03 |
| 0.06 | wiki | 7.7967 (n=3) | 6.4889 | 6.4994 | 6.5556 | 7.0257 | 6.4944 | 7.3506 |
| 0.06 | c4 | 11.4206 (n=3) | 9.4720 | 9.4715 | 9.5516 | 10.1241 | 9.4633 | 10.6336 |
| 0.06 | act_mse | 4.125e-03 (n=3) | 3.027e-03 | 3.026e-03 | 3.606e-03 | 2.848e-03 | 2.847e-03 | 3.300e-03 |
| 0.06 | emean | 6.681e-03 (n=3) | 5.451e-03 | 5.448e-03 | 6.110e-03 | 5.560e-03 | 5.543e-03 | 6.004e-03 |
| 0.08 | wiki | 7.3619 (n=3) | 6.4909 | 6.4776 | 6.5358 | 7.1367 | 6.4913 | 7.6276 |
| 0.08 | c4 | 10.6237 (n=3) | 9.4475 | 9.4447 | 9.5335 | 10.2875 | 9.4592 | 10.9205 |
| 0.08 | act_mse | 4.061e-03 (n=3) | 2.949e-03 | 2.949e-03 | 3.487e-03 | 2.744e-03 | 2.744e-03 | 3.115e-03 |
| 0.08 | emean | 6.551e-03 (n=3) | 5.289e-03 | 5.287e-03 | 5.910e-03 | 5.423e-03 | 5.406e-03 | 5.793e-03 |
| 0.1 | wiki | 7.2831 (n=3) | 6.4777 | 6.4804 | 6.5086 | 7.3245 | 6.4841 | 7.9006 |
| 0.1 | c4 | 11.1366 (n=3) | 9.4294 | 9.4312 | 9.4944 | 10.4024 | 9.4244 | 10.8296 |
| 0.1 | act_mse | 3.966e-03 (n=3) | 2.853e-03 | 2.852e-03 | 3.326e-03 | 2.622e-03 | 2.621e-03 | 2.884e-03 |
| 0.1 | emean | 6.379e-03 (n=3) | 5.108e-03 | 5.105e-03 | 5.669e-03 | 5.264e-03 | 5.245e-03 | 5.530e-03 |
| 0.2 | wiki | 6.5130 (n=3) | 6.4320 | 6.4283 | 6.4640 | 7.2275 | 6.4424 | 7.8090 |
| 0.2 | c4 | 9.4795 (n=3) | 9.3526 | 9.3515 | 9.4047 | 10.3938 | 9.3705 | 11.0877 |
| 0.2 | act_mse | 3.563e-03 (n=3) | 2.519e-03 | 2.518e-03 | 2.717e-03 | 2.111e-03 | 2.111e-03 | 2.216e-03 |
| 0.2 | emean | 5.646e-03 (n=3) | 4.436e-03 | 4.431e-03 | 4.733e-03 | 4.600e-03 | 4.588e-03 | 4.714e-03 |
| 0.5 | wiki | 6.3695 (n=3) | 6.3296 | 6.3264 | 6.3312 | 7.2793 | 6.3552 | 7.1454 |
| 0.5 | c4 | 9.2571 (n=3) | 9.1761 | 9.1795 | 9.1812 | 10.3494 | 9.1937 | 10.3026 |
| 0.5 | act_mse | 2.392e-03 (n=3) | 1.683e-03 | 1.683e-03 | 1.741e-03 | 7.693e-04 | 7.701e-04 | 8.019e-04 |
| 0.5 | emean | 3.675e-03 (n=3) | 2.850e-03 | 2.838e-03 | 2.971e-03 | 2.932e-03 | 2.923e-03 | 2.949e-03 |

## Allocation audit — down_proj share and layers.1.mlp.down_proj

FP8 channel share given to each projection type under the ACROSS scope (fraction of that projection's channels set to MXFP8), plus the quota of layers.1.mlp.down_proj (the layer MA starved in M31).

### 3b

| r | strategy | q | k | v | gate | up | down | layers.1.mlp.down_proj h/d |
|---|---|---|---|---|---|---|---|---|
| 0.02 | ma (ext) | 0.033 | 0.033 | 0.033 | 0.028 | 0.028 | 0.000 | 0/8192 |
| 0.02 | ss (ext) | 0.033 | 0.033 | 0.033 | 0.028 | 0.028 | 0.000 | 32/8192 |
| 0.02 | dd (ext) | 0.031 | 0.031 | 0.031 | 0.032 | 0.032 | 0.000 | 0/8192 |
| 0.08 | ma (ext) | 0.131 | 0.131 | 0.131 | 0.116 | 0.116 | 0.000 | 0/8192 |
| 0.08 | ss (ext) | 0.130 | 0.130 | 0.130 | 0.116 | 0.116 | 0.001 | 32/8192 |
| 0.08 | dd (ext) | 0.129 | 0.129 | 0.129 | 0.119 | 0.119 | 0.000 | 0/8192 |
| 0.1 | ma (legacy) | 0.167 | 0.167 | 0.167 | 0.145 | 0.145 | 0.001 | 0/8192 |
| 0.1 | ss (legacy) | 0.166 | 0.166 | 0.166 | 0.146 | 0.146 | 0.001 | 32/8192 |
| 0.1 | dd (ext) | 0.164 | 0.164 | 0.163 | 0.150 | 0.150 | 0.000 | 0/8192 |
| 0.2 | ma (legacy) | 0.321 | 0.321 | 0.321 | 0.278 | 0.278 | 0.001 | 0/8192 |
| 0.2 | ss (legacy) | 0.321 | 0.321 | 0.321 | 0.277 | 0.277 | 0.001 | 32/8192 |
| 0.2 | dd (ext) | 0.320 | 0.320 | 0.320 | 0.281 | 0.281 | 0.000 | 0/8192 |
| 0.5 | ma (legacy) | 0.782 | 0.782 | 0.782 | 0.742 | 0.742 | 0.001 | 0/8192 |
| 0.5 | ss (legacy) | 0.781 | 0.781 | 0.781 | 0.743 | 0.743 | 0.002 | 32/8192 |
| 0.5 | dd (ext) | 0.788 | 0.788 | 0.788 | 0.734 | 0.734 | 0.000 | 0/8192 |

### 8b

| r | strategy | q | k | v | gate | up | down | layers.1.mlp.down_proj h/d |
|---|---|---|---|---|---|---|---|---|
| 0.02 | ma (ext) | 0.038 | 0.038 | 0.038 | 0.036 | 0.035 | 0.001 | 0/14336 |
| 0.02 | ss (ext) | 0.038 | 0.038 | 0.038 | 0.035 | 0.035 | 0.001 | 32/14336 |
| 0.02 | dd (ext) | 0.037 | 0.037 | 0.037 | 0.038 | 0.038 | 0.000 | 0/14336 |
| 0.08 | ma (ext) | 0.138 | 0.138 | 0.138 | 0.128 | 0.127 | 0.001 | 0/14336 |
| 0.08 | ss (ext) | 0.137 | 0.137 | 0.137 | 0.127 | 0.127 | 0.001 | 32/14336 |
| 0.08 | dd (ext) | 0.139 | 0.139 | 0.139 | 0.127 | 0.127 | 0.000 | 0/14336 |
| 0.1 | ma (legacy) | 0.177 | 0.177 | 0.177 | 0.163 | 0.163 | 0.001 | 0/14336 |
| 0.1 | ss (legacy) | 0.176 | 0.176 | 0.176 | 0.163 | 0.163 | 0.001 | 32/14336 |
| 0.1 | dd (ext) | 0.178 | 0.178 | 0.178 | 0.162 | 0.162 | 0.000 | 0/14336 |
| 0.2 | ma (legacy) | 0.357 | 0.357 | 0.356 | 0.322 | 0.322 | 0.001 | 0/14336 |
| 0.2 | ss (legacy) | 0.357 | 0.357 | 0.357 | 0.321 | 0.321 | 0.002 | 32/14336 |
| 0.2 | dd (ext) | 0.361 | 0.361 | 0.361 | 0.317 | 0.317 | 0.000 | 0/14336 |
| 0.5 | ma (legacy) | 0.882 | 0.882 | 0.882 | 0.799 | 0.799 | 0.001 | 0/14336 |
| 0.5 | ss (legacy) | 0.885 | 0.885 | 0.885 | 0.795 | 0.795 | 0.002 | 32/14336 |
| 0.5 | dd (ext) | 0.876 | 0.876 | 0.876 | 0.810 | 0.810 | 0.001 | 0/14336 |

## Findings (as-measured, incl. results running against the spec's expectations)

1. **DD (across) does NOT fix the cross-layer collapse.** On 3B it tracks MA (across): wiki PPL 17.6/18.3/17.1/18.6 at r=0.02-0.08 and 14.8-17.8 at 0.1-0.5, all worse than pure NVFP4 (14.33). On 8B the same pattern is milder (7.35-7.90 vs r=0 baseline 7.78, while within-scope reaches 6.48-6.58). SS (across) remains the only non-collapsing across criterion (3B 8.11-8.23; 8B 6.44-6.53).
2. **DD (across) also loses on its own same-unit objective**: its held-out act_mse_global is HIGHER than MA/SS (across) at every r (e.g. 3B r=0.1: 3.30e-3 vs 2.73e-3), and DD (within) is likewise worse than MA/SS (within). The spec's expectation (sec 6.4.1: DD across significantly lower act_mse_global) is contradicted. Cause: the per-channel DD score is measured independently on the ORIGINAL block layout, so it ignores the dominant interaction - moving a high-magnitude channel OUT of an NVFP4 block16 shrinks that block's scale and reduces every remaining element's error. MA/SS target exactly those scale-poisoning outlier channels and therefore minimize the realized post-packing error better than the format-aware but interaction-blind DD.
3. **Critical-layer protection**: DD (across) gives layers.1.mlp.down_proj 0 channels at every reported r (same as MA); SS (across) consistently gives it exactly 32 channels - and that single 32-channel difference separates collapse (MA/DD) from stability (SS). All three across criteria starve the down_proj projection type as a whole (<=0.2% share).
4. **Densified cliff**: on 3B the within-scope targeted criteria recover almost the entire NVFP4->MXFP8 PPL gap with the FIRST 2% of channels (14.33 -> 8.22-8.27 at r=0.02); RND (within) is far behind and high-variance in the cliff region (seed-mean 10.5-15.2 over 0.02-0.08). Same shape on 8B (7.78 -> 6.53-6.58 at r=0.02 vs RND 7.36-7.80).
5. **act-MSE-optimal != PPL-optimal**: at r>=0.2 the across criteria all have clearly lower act_mse_global than within (e.g. 3B r=0.5: 1.02-1.11e-3 vs 1.71-1.86e-3), yet MA/DD (across) are catastrophically worse in PPL - allocation quality is NOT summarized by the global activation-MSE objective alone; layer-coverage constraints matter.
6. **P2 (1B) not started**: the 3B/8B cliff region is fully resolved with large dynamic range between RND / targeted-within / across at r=0.02-0.08, so the optional larger-headroom model was unnecessary. DD_out (optional diagnostic variant) was likewise not run; the main DD criterion already shows the failure mode is packing interaction, not output weighting.

