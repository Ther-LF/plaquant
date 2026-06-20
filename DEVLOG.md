# ProMix Development Log

## Project Goal

Build a fully self-contained mixed-precision INT4/INT8 quantized inference system for LLMs, based on the ResQ (ICML 2025) method. The system performs PCA + Hadamard rotation on activations, splits channels by variance into high-precision (INT8, 1/8 of hidden dim) and low-precision (INT4, 7/8), then uses custom CUTLASS fused kernels for dual-precision GEMM in a single kernel launch.

---

## Completed Work

### 1. ProMix PTQ Pipeline (Fully Working, Self-Sufficient)

**Result: PPL = 14.72 on Llama-3.2-1B (matches ResQ paper exactly)**

The pipeline has 3 stages:

| Step | Script | Time | Description |
|------|--------|------|-------------|
| 0 | `promix/scripts/run_basis.sh` | ~20 min | PCA basis computation (fully independent) |
| 1 | `promix/scripts/run_optimize_rotation.sh` | ~34 min | Rotation optimization (optional for 1B) |
| 2 | `promix/scripts/run_ptq_eval.sh` | ~3 min | Quantize + evaluate PPL |

Key files:
- `promix/quantize/basis.py` — PCA basis computation, no dependency on project-resq
- `promix/quantize/optimize_rotation.py` — SGDG optimization on Stiefel manifold
- `promix/eval/ptq.py` — Config-driven PTQ evaluation entry point
- `promix/eval/evaluator.py` — Layer-by-layer PPL evaluation
- `promix/models/loader.py` — Model loading with tie_word_embeddings fix
- `promix/quantize/rotation.py` — Rotation fusion into model weights
- `promix/quantize/fuse_norm.py` — RMSNorm fusion
- `promix/quantize/hadamard.py` — Hadamard utilities (includes had28 for 8B)
- `promix/quantize/quant_utils.py` — ActQuantizer + ActQuantWrapper

Multi-model support verified:
- Llama-3.2-1B: PPL = 14.72
- Llama-3.2-3B: PPL = 10.21
- Llama-3-8B: PPL = 6.72

Key technical discoveries:
- `tie_word_embeddings` must be disabled + lm_head cloned before rotation (otherwise rotation corrupts lm_head)
- Rotation optimization provides NO improvement for 1B model (random orthogonal R is already optimal)
- 8B model requires real Hadamard-28 matrix (±1 entries, not random orthogonal with norm=1)
- hadamard_data.json stores the pre-computed had28 matrix

### 2. Mixed-Precision Fused GEMM Kernel (SM80, Working)

**Location: `kernels/mixed_gemm_l20/`**

Performance: **1.12-1.19x speedup** vs CUTLASS SM80 2-launch baseline (correctness: cosine = 1.0)

Architecture:
- INT4 path: InstructionShape<16,8,64>, ThreadblockShape<64,64,128>
- INT8 path: InstructionShape<16,8,32>, ThreadblockShape<64,64,64>
- Two phases share INT32 accumulator, SMEM union for memory reuse
- B matrix stored as (N,K) contiguous = ColumnMajor(K,N)

Also exposes standalone INT4/INT8 GEMM functions:
- `cutlass_int4_gemm(A_packed, B_packed)` → (M, N) FP32
- `cutlass_int8_gemm(A, B)` → (M, N) FP32
- Outputs FP32 (not FP16) to avoid INT32→FP16 overflow for large K values

Build:
```bash
cd kernels/mixed_gemm_l20
rm -rf build *.so
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace
```

### 3. Real Inference Pipeline (Code Complete, Needs Remote Verification)

Replaces fake quant (quantize→dequant→FP16 GEMM) with actual INT GEMM:

```
ActQuantWrapper.forward(x):
  1. Online Hadamard (down_proj only)
  2. Split activation: main (K_main) + high (K_high)
  3. Per-token quantize → unsigned INT4/INT8
  4. Shift to signed: [-8,7] / [-128,127]
  5. Pack INT4 (2 values per byte)
  6. INT GEMM via CUTLASS kernel (Solution A: separate INT4 + INT8 calls)
  7. Dequant: output = s_x * s_w * (raw + (shift - zero) * colsum)
```

Files:
- `promix/inference/quant_ops.py` — Activation quantization + INT4 packing ops
- `promix/inference/weight_packer.py` — Pre-pack model weights to INT4/INT8 format
- `promix/inference/real_forward.py` — Real forward replacing fake quant, with kernel fallback
- `promix/inference/benchmark.py` — E2E benchmark (FP16 vs Real Quant)

The pipeline uses PyTorch fallback when the CUTLASS kernel .so is not available (for testing on machines without CUDA toolkit). When the kernel IS available, it calls `cutlass_int4_gemm` and `cutlass_int8_gemm`.

**Known limitation**: o_proj layers use per-group quantization (groupsize > 0) which is incompatible with the single-scale-per-row kernel. These layers keep fake quant for now.

### 4. Additional Quantization Features

- **KV cache quantization** (`promix/quantize/kv_quant.py`): QKRotationWrapper for K-cache rotation
- **GPTQ** (`promix/quantize/gptq.py`): Hessian-based optimal rounding with error compensation
- **Multi-config support**: Configs for W4A4, W4A4KV4, different models

### 5. Training Infrastructure

- `promix/train/optimizer.py` — SGDG (Stiefel manifold SGD via Cayley transform)
- `promix/train/modeling_llama_train.py` — Llama with trainable R1/R2 in forward
- `promix/train/quant_linear.py` — Quantized linear for training

---

## Architecture Overview

```
Input x (FP16)
    │
    ├── [Offline] PCA basis U sorts channels by variance
    ├── [Offline] Rotation R smooths distribution within subspace
    ├── [Offline] U×R fused into model weights (zero runtime cost)
    │
    ▼
ActQuantWrapper.forward(x):
    │
    ├── Online Hadamard (down_proj only, fast_hadamard_transform)
    │
    ├── Split: x_main[0:K_main] (INT4), x_high[K_main:] (INT8)
    │
    ├── Per-token quantize: scale = (max-min)/maxq, zero = round(-min/scale)
    │
    ├── Shift to signed: q - 2^(bits-1)
    │
    ├── GEMM:
    │   ├── cutlass_int4_gemm(q_main_packed, W_main_packed) → FP32
    │   └── cutlass_int8_gemm(q_high, W_high_int8) → FP32
    │
    └── Dequant:
        output = s_x_main * s_w_main * (raw_main + bias_main)
               + s_x_high * s_w_high * (raw_high + bias_high)
        where bias = (shift - zero) * colsum_w
```

Key dimensions (Llama-3.2-1B, high_fraction=0.125):

| Projection | K_high (INT8) | K_main (INT4) |
|-----------|--------------|--------------|
| q/k/v/o_proj | 256 | 1792 |
| gate/up_proj | 256 | 1792 |
| down_proj | 1024 | 7168 |

---

## Pending Work (Priority Order)

### P0: Verify Real Inference on Remote

- [ ] Connect to H20 remote, compile kernel, run `promix/inference/benchmark.py`
- [ ] Verify PPL matches fake quant (should be ≈14.72)
- [ ] Measure end-to-end latency: FP16 vs Real Quant
- [ ] Expected: compute-bound scenarios (large batch) should show INT4 speedup

### P1: Solve o_proj Per-Group Quantization

The o_proj layer has per-group quantization (groupsize=128), meaning different groups of K channels use different scales. After GEMM reduction over K, you can't separate per-group contributions.

Options:
- A) Modify kernel to accept per-group scales (fuse dequant into epilogue)
- B) Use per-channel quantization for o_proj (may hurt accuracy)
- C) Keep fake quant for o_proj only (current approach, acceptable for now)

### P2: SM90 WGMMA Fused Kernel (Higher Performance)

Plan exists at `.claude/plans/compiled-jumping-unicorn.md`:
- Dual accumulator: Phase 1 (INT4 WGMMA) → Phase 2 (INT8 WGMMA) → FP32 combine
- TileShape 128×128×128, warp-specialized with TMA
- SMEM budget: 160 KB (3 stages main + 2 stages high)
- Target: single kernel launch for maximum throughput on H20

### P3: Fuse Dequant into Kernel Epilogue (Solution C)

Currently dequant happens in Python after kernel returns. Fusing scale/bias/zero into the CUTLASS EVT epilogue saves:
- 2 kernel launches (currently: GEMM + element-wise dequant)
- Memory bandwidth for reading/writing intermediate (M, N) tensor

### P4: Decouple from project-resq Completely

Remaining dependencies (Step 1 only):
- `project-resq/fake_quant/train_utils/modeling_llama_train.py` — Llama with trainable R
- `project-resq/fake_quant/train_utils/optimizer.py` — SGDG optimizer

Already migrated to `promix/train/` but still references resq model architecture for the training forward pass.

### P5: Larger Model Validation

- [ ] Run 70B PTQ eval (basis computation done, need eval)
- [ ] Verify rotation optimization helps on 7B+ models
- [ ] Memory optimization for 70B (may need model parallelism or offloading)

### P6: lm-eval Integration

- Device mismatch issues with lm_eval harness (model on CUDA, harness expects different handling)
- Need to wrap quantized model properly for lm_eval's interface

---

## Reproduction Instructions

### Prerequisites
- NVIDIA GPU (H20/A100/L20 for inference, H20 for kernel compilation)
- Python 3.9+ with PyTorch 2.x + CUDA 12.x
- `pip install transformers datasets fast_hadamard_transform`

### Full Pipeline (PCA basis already in `./rotation/`)

```bash
# Step 0: Compute PCA basis (~20 min, single GPU, ~14GB VRAM)
CUDA_VISIBLE_DEVICES=0 python -m promix.quantize.basis \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation --nsamples 512

# Step 1: Rotation optimization (~34 min, optional for 1B)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29505 \
    -m promix.quantize.optimize_rotation \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation --max_steps 100 --learning_rate 1.5

# Step 2: PTQ evaluation (~3 min)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29507 \
    promix/eval/ptq.py --config promix/configs/llama-3.2-1b-resq.yaml
```

### Kernel Compilation (requires nvcc + CUTLASS)

```bash
cd kernels/mixed_gemm_l20
rm -rf build *.so
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace
python benchmark.py  # verify correctness + measure speedup
```

### Real Inference Benchmark

```bash
# After kernel compiled + PTQ pipeline run:
python -m promix.inference.benchmark \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --kernel_path kernels/mixed_gemm_l20/mixed_gemm_l20*.so
```

---

## File Reference

```
promix/
├── __init__.py
├── utils.py                         # DEV, cleanup_memory, HadamardTransform
├── configs/
│   ├── llama-3.2-1b-resq.yaml      # Main config (PPL=14.72)
│   ├── llama-3.2-1b.yaml           # Base model config
│   ├── llama-3.2-3b.yaml           # 3B config
│   ├── llama-3-8b.yaml             # 8B config
│   ├── llama-3-70b.yaml            # 70B config
│   ├── llama-3.2-1b-kv4.yaml       # With KV cache quantization
│   └── llama-3.2-1b-w4a4kv4.yaml   # Full W4A4KV4
├── models/
│   ├── loader.py                    # load_model, install_column_order_hooks
│   ├── base.py                      # ModelPlugin base class
│   └── llama.py                     # Llama-specific plugin
├── quantize/
│   ├── basis.py                     # Step 0: PCA basis (independent)
│   ├── optimize_rotation.py         # Step 1: rotation optimization
│   ├── quant_utils.py              # ActQuantizer, ActQuantWrapper
│   ├── rotation.py                  # Rotation fusion into weights
│   ├── fuse_norm.py                # RMSNorm fusion
│   ├── hadamard.py                 # Hadamard utils + had28
│   ├── hadamard_data.json          # Pre-computed Hadamard matrices
│   ├── kv_quant.py                 # KV cache quantization
│   └── gptq.py                     # GPTQ optimal rounding
├── eval/
│   ├── ptq.py                      # Main PTQ entry point
│   ├── evaluator.py                # Layer-by-layer PPL eval
│   └── data.py                     # Dataset loading
├── inference/
│   ├── quant_ops.py                # Activation quantize + pack
│   ├── weight_packer.py            # Pack weights for INT GEMM
│   ├── real_forward.py             # Real INT GEMM forward
│   └── benchmark.py               # E2E benchmark
├── train/
│   ├── optimizer.py                # SGDG (Stiefel manifold)
│   ├── modeling_llama_train.py     # Trainable Llama
│   └── quant_linear.py            # Quantized linear
└── scripts/
    ├── run_basis.sh                # Step 0 runner
    ├── run_optimize_rotation.sh    # Step 1 runner
    └── run_ptq_eval.sh            # Step 2 runner

kernels/
├── mixed_gemm_l20/                 # SM80 fused kernel (working)
│   ├── mixed_gemm_l20.cu          # CUTLASS kernel source
│   ├── setup.py                   # Build script
│   └── benchmark.py              # Correctness + perf test
└── mixed_gemm/                    # SM90 kernel (planned)
```
- **[2026-06-20 14:21:04]** [doc] create `DEVLOG.md`
