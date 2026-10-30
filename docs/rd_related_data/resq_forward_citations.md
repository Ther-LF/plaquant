# Forward citations of ResQ (2412.14363) — quant-relevant, from Semantic Scholar graph (44 citers, 38 quant-relevant)
## CLOSEST to ProMix (error-driven high/low partition + PCA rotation + dual-format NVFP4/MXFP8 fused kernel) — read first
- ARCQuant (2601.07475) — NVFP4 + augmented residual channels  [NVFP4 + ResQ-residual: closest format+residual]
- MicroMix (2508.02343) — mixed-precision with MICROSCALING formats (MX)  [our NVFP4/MXFP8 cousin]
- MosaicQuant (2606.15652) — inlier/outlier disaggregation, unified 4-bit  [= our high/low partition]
- CoQuant (2604.26378) — joint weight-activation SUBSPACE projection, mixed-precision  [= PCA subspace + partition]
- dMX (2606.04115) — DIFFERENTIABLE mixed-precision assignment for low-prec FP  [= our allocation, learned]
- HCDCMQ — Hessian-aware channel-wise mixed-precision PTQ  [= our ΔD/loss-weighted allocation]
- AlignQuant (2610.07457) / TileMix (2608.17336) — tile-ALIGNED mixed-precision  [= our hardware-alignment co-design]
- OffQ (2606.07116) — structured outliers by offsetting; KurTail (2503.01483) — kurtosis-based  [= our outlier/kurtosis mechanism]
- GyRot (2607.27694) — rotation + fine-grained group quant; JustQuant (2609.33601) — "no rotation needed for 4-bit act" [counter-claim]
## Also relevant
- SharQ (2606.26587) FP4+sparsity; DualQuant (2610.13767); TWLA (2606.13054) ternary+low-act; SensitivityGuided MPQAT;
- Joint Pruning+MPQ (2606.07819); Multi-Bitwidth additive codebooks (2606.12876); Component-type attn sensitivity (2609.26173)
- RotateKV (2501.16383); KVLinC (2510.05373); Don't-Waste-Bits KV (2604.04722)  [KV-cache rotation/MP]
## Survey hubs
- Mixed-Precision Quantization for LMs: Techniques and Prospects (2510.16805)
- A Comprehensive Evaluation on Quantization Techniques (2507.17417)
