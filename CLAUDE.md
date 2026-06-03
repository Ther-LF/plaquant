# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PLAQuant implements mixed-precision INT4/INT8 quantized inference for LLMs on NVIDIA Hopper (SM90a) GPUs. It builds on the ResQ quantization method: after PCA + Hadamard rotation, activations are split into high-variance channels (8-bit, 1/8 of hidden dim) and low-variance channels (4-bit, 7/8 of hidden dim). Custom CUTLASS 3.x kernels fuse these dual-precision GEMMs into a single launch.

## Build Commands

### Custom CUDA Kernels (requires H20/Hopper GPU)

```bash
# Mixed-precision GEMM kernel
cd kernels/mixed_gemm && pip install -e .

# Mixed-precision FlashAttention kernel
cd kernels/flash_attn && pip install -e .
```

Both use PyTorch's `CUDAExtension` with CUTLASS 3.x headers from `third_party/cutlass/`. Target arch is `compute_90a` (the `a` suffix is required for Hopper WGMMA/TMA).

### ResQ Pipeline (project-resq/fake_quant/)

```bash
cd project-resq/fake_quant
pip install -r ../requirements.txt

bash 0_get_basis.sh       # PCA basis computation
bash 1_optimize_rotation.sh  # rotation matrix optimization
bash 2_eval_ptq.sh        # quantize + eval (wikitext, MMLU, etc.)
bash 4_collect_gemm.sh    # collect per-layer GEMM data for kernel tests
bash 4_bench_gemm.sh      # benchmark GEMM implementations
```

## Running Tests

```bash
# Full test suite (needs gemm_data/ from collect step and GPU)
pytest tests/test_mixed_gemm.py -v

# Filter by layer or batch size
pytest tests/test_mixed_gemm.py -v -k "q_proj"
pytest tests/test_mixed_gemm.py -v -k "bs1"
```

Tests load ground-truth tensors from `gemm_data/` (collected by `collect_gemm_data.py`) and validate kernel output against both real-quant reference and FP16 baseline.

## Architecture

### Kernel Design (kernels/mixed_gemm/)

The fused mixed-precision GEMM exploits the fact that INT4 and INT8 WGMMA instructions on Hopper produce accumulator fragments with identical thread-to-element mapping, allowing in-register accumulation across precisions.

**Epilogue Visitor Tree (EVT)** implements fused asymmetric dequantization:
```
D[m,n] = s_x[m] * s_w[n] * (acc[m,n] - zero_x[m] * colsum_w[n])
```

The shift-128 trick converts UINT8 activations to INT8 for Tensor Core compatibility, with bias correction via precomputed weight column sums.

### Quantization Pipeline (project-resq/fake_quant/)

- `get_basis.py` — PCA to identify high-variance subspace
- `ptq.py` — post-training quantization evaluation (torchrun for multi-GPU)
- `collect_gemm_data.py` — extracts per-layer quantized tensors as test fixtures
- `bench_gemm.py` — performance benchmarking (each test computes live, no cloning)
- `triton_gemm.py` — Triton INT8 GEMM reference kernels

### Key Dimensions (Llama-3.2-1B, high_fraction=0.125)

| Projection | K_high (INT8) | K_main (INT4) |
|-----------|--------------|--------------|
| q/k/v/o_proj | 256 | 1792 |
| gate/up_proj | 256 | 1792 |
| down_proj | 1024 | 7168 |

## Code Hygiene Rules

- **No temp files in repo** — write throwaway tests/debug scripts to `/tmp/`
- **No duplicate scripts** — one canonical shell script per pipeline step; use flags for modes
- **No old kernel versions** — delete replaced kernels in the same commit
- **No commented-out blocks >3 lines** — use git history
- **No unused imports** in modified Python files
- **bench_gemm.py** — every test must compute its output live and measure its own latency

## Remote Development

- GPU container: `gemini@general-1295685810-geminijob-0` (H20 SM90a)
- venv: `source /vllm-workspace/plaquant/.venv/bin/activate`
- CUDA workaround: `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so`
- GEMM test data: `/vllm-workspace/plaquant/project-resq/fake_quant/gemm_data/`

### 连接远程服务器

通过 tmux 后台 session 连接（非交互式环境必须用 tmux）：

```bash
# 创建 tmux session 并连接
tmux kill-session -t remote 2>/dev/null
tmux new-session -d -s remote -x 200 -y 50
tmux send-keys -t remote "~/gemini-go gemini@general-1295685810-geminijob-0 UERzVUR5cTFrd2hnUndvRnY3NVRUcEt1QWdFVUhqSzZteFVxVFdSUmFaT1dzaEFvNVpTWExsN2RjZVdSVWtGeTo6OjM0MzY6OjpzcGFuYWx1bzo6OnByb2plY3Q=" Enter

# 等待连接后执行命令
sleep 8
tmux send-keys -t remote "cd /vllm-workspace/plaquant && git pull" Enter

# 获取输出
sleep 5 && tmux capture-pane -t remote -p | tail -20
```

### 远程编译和运行 kernel

```bash
# 在远程编译（base 环境有 torch 2.9.1+cu128）
tmux send-keys -t remote "cd /vllm-workspace/plaquant/kernels/mixed_gemm_l20 && rm -rf build *.so && LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace 2>&1 | tail -5" Enter

# 运行 benchmark
tmux send-keys -t remote "LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python benchmark.py 2>&1" Enter
```

注意：
- 远程连接会因超时断开（默认 24h），需要重新连接
- 编译用 base 环境（不要 activate .venv，里面没有 torch）
- `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so` 是 H20 上的必需 workaround
- 先 `git push` 本地改动，再在远程 `git pull`

### 本地环境（L20）

- 8× NVIDIA L20 (SM89, Ada Lovelace, 48GB)
- 没有完整 CUDA toolkit（无 nvcc），不能本地编译 CUDA kernel
- 有 PyTorch（torch 2.x + CUDA 12.8），可以跑 Python 代码
- CUTLASS 源码在 `third_party/cutlass/`（用于代码阅读和 IDE 跳转）

## Technical Notes

- Asymmetric quantization: `a_sym=False` means activations are asymmetrically quantized
- Per-group layers (o_proj): `group_k=56`, padded to 64 for BLOCK_K alignment
- CUTLASS 3.x on H20 requires the `a` suffix: `-gencode arch=compute_90a,code=sm_90a`
- CUTLASS EVT Arguments order: children (input) nodes first, then the operation node last
