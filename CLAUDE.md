# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProMix (previously PLAQuant) implements mixed-precision INT4/INT8 quantized inference for LLMs on NVIDIA GPUs. It builds on the ResQ quantization method: after PCA + Hadamard rotation, activations are split into high-variance channels (8-bit, 1/8 of hidden dim) and low-variance channels (4-bit, 7/8 of hidden dim). Custom CUTLASS kernels fuse these dual-precision GEMMs into a single launch.

## Repository Structure

```
plaquant/
├── promix/                    # Our reimplementation (PTQ pipeline + model plugins)
│   ├── configs/               # YAML quantization configs
│   ├── models/                # Model plugin architecture (base.py, llama.py)
│   ├── quantize/              # Quantization logic (quant_utils.py, hadamard.py, etc.)
│   ├── eval/                  # PTQ evaluation (ptq.py — main entry)
│   └── scripts/               # Run scripts (run_basis.sh, run_ptq_eval.sh)
├── project-resq/              # Original ResQ code (submodule, reference)
├── kernels/
│   ├── mixed_gemm/            # SM90 Hopper fused GEMM (WIP)
│   └── mixed_gemm_l20/       # SM80 Ampere fused GEMM (working, 1.12-1.19x speedup)
├── third_party/
│   ├── cutlass/               # CUTLASS 4.5 (for kernel development)
│   ├── LLM-Infra-Reference/   # Knowledge base (CUTLASS analysis, Hopper docs)
│   └── ...                    # Other references
└── docs/                      # Design specs, optimization ideas
```

## Current Status

### ProMix PTQ Pipeline (✅ Working)
- **PPL = 14.72** (matches ResQ exactly)
- Config: `promix/configs/llama-3.2-1b-resq.yaml`
- Run: `bash promix/scripts/run_ptq_eval.sh` (on remote H20)
- Key fix: must untie word embeddings + clone lm_head before rotation (see ptq.py)
- Currently delegates to project-resq's scripts for basis/rotation generation

### Mixed-Precision Fused GEMM (✅ Working)
- `kernels/mixed_gemm_l20/` — INT4+INT8 single-launch fused GEMM
- Correctness: cosine = 1.0 (bit-exact)
- Performance: 1.12-1.19x vs CUTLASS SM80 2-launch baseline
- Tile: 64×64, Registers: 90-96, Stages: Low=5, High=4
- Benchmark: `python benchmark.py` (on remote H20)

### Pending Work
- ProMix: self-generate basis + rotation (not use pre-existing files)
- ProMix: lm-eval benchmarks (device issues with lm_eval wrapper)
- SM90 WGMMA fused kernel (kernels/mixed_gemm/)
- Real inference pipeline (quantized weights → fused kernel)

## Build Commands

### Mixed-Precision GEMM Kernel (SM80, on remote H20)

```bash
cd kernels/mixed_gemm_l20
rm -rf build *.so
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python benchmark.py
```

### ProMix PTQ Evaluation (on remote H20)

```bash
# Requires .venv activated + rotation files in project-resq/fake_quant/rotation/
cd /vllm-workspace/plaquant
source .venv/bin/activate
bash promix/scripts/run_ptq_eval.sh
```

### ResQ Pipeline (project-resq/fake_quant/)

```bash
cd project-resq/fake_quant
source /vllm-workspace/plaquant/.venv/bin/activate
bash 0_get_basis.sh       # PCA basis computation (~5 min)
bash 1_optimize_rotation.sh  # rotation optimization (~10 min)
bash 2_eval_ptq.sh        # quantize + eval
```

## Remote Development

### 远程服务器信息
- GPU container: `gemini@general-1295685810-geminijob-0` (8× H20 SM90a)
- 工作目录: `/vllm-workspace/plaquant/`
- venv: `source /vllm-workspace/plaquant/.venv/bin/activate` (Python 3.9, 有 fast_hadamard_transform 等)
- Base 环境: Python 3.12, torch 2.9.1+cu128 (用于编译 CUDA kernel)
- CUDA workaround: `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so`

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

### 远程执行模式

```bash
# 编译 kernel（用 base 环境，不要 activate .venv）
tmux send-keys -t remote "cd /vllm-workspace/plaquant/kernels/mixed_gemm_l20 && rm -rf build *.so && LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace 2>&1 | tail -5" Enter

# 跑 benchmark
tmux send-keys -t remote "LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python benchmark.py 2>&1" Enter

# 跑 PTQ eval（需要 venv）
tmux send-keys -t remote "source .venv/bin/activate && bash promix/scripts/run_ptq_eval.sh 2>&1 | tail -10" Enter

# 获取远程输出
sleep N && tmux capture-pane -t remote -p | tail -20
```

### 注意事项
- 远程连接会因超时断开（~24h），需要重新连接
- 编译用 base 环境（有 torch）；PTQ eval 用 .venv（有 fast_hadamard_transform）
- `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so` 是 H20 上的必需 workaround
- 先 `git push` 本地改动，再在远程 `git pull`
- GEMM test data: `/vllm-workspace/plaquant/project-resq/fake_quant/gemm_data/`

### 本地环境（L20）

- 8× NVIDIA L20 (SM89, Ada Lovelace, 48GB)
- 没有完整 CUDA toolkit（无 nvcc），不能本地编译 CUDA kernel
- 有 PyTorch（torch 2.x + CUDA 12.8），可以跑 Python 代码
- CUTLASS 源码在 `third_party/cutlass/`（用于代码阅读和 IDE 跳转）

## Key Technical Details

### ProMix PTQ 关键修复（tie_word_embeddings）

Llama-3.2-1B 的 `config.tie_word_embeddings = True`，导致 embed_tokens 和 lm_head 共享权重。rotation fuse 修改 embed_tokens 时会同时改坏 lm_head。必须：
```python
config.tie_word_embeddings = False
model = LlamaForCausalLM.from_pretrained(..., config=config)
model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
```

### Mixed-Precision GEMM Kernel 设计

- INT4 path: InstructionShape<16,8,64>, ThreadblockShape<64,64,128>
- INT8 path: InstructionShape<16,8,32>, ThreadblockShape<64,64,64>
- 两者共享 accumulator (INT32)，顺序执行 (先 low 后 high)
- SMEM 用 union 复用
- B 矩阵存储为 (N,K) contiguous = ColumnMajor(K,N)

### CUTLASS EVT Arguments 顺序
- children (input) nodes first, then the operation node last
- 即模板参数 `Sm90EVT<Op, Child0, Child1>` 对应 Args: `{Child0_args, Child1_args, Op_args}`

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

## Running Tests

```bash
# Full test suite (needs gemm_data/ from collect step and GPU)
pytest tests/test_mixed_gemm.py -v

# Filter by layer or batch size
pytest tests/test_mixed_gemm.py -v -k "q_proj"
pytest tests/test_mixed_gemm.py -v -k "bs1"
```
