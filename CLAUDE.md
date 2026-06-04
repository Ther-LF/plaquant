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
│   ├── quantize/              # Quantization logic
│   │   ├── basis.py           # Step 0: PCA basis computation (✅ fully independent)
│   │   ├── optimize_rotation.py # Step 1: rotation optimization (uses resq train model)
│   │   ├── quant_utils.py     # ActQuantizer + ActQuantWrapper
│   │   ├── fuse_norm.py       # RMSNorm fusion (delegates to resq)
│   │   ├── hadamard.py        # Hadamard utils (delegates to resq)
│   │   └── rotation.py        # Rotation application (delegates to resq)
│   ├── eval/                  # PTQ evaluation (ptq.py — main entry)
│   └── scripts/               # Run scripts
│       ├── run_basis.sh       # Step 0: compute PCA basis (~20 min)
│       ├── run_optimize_rotation.sh  # Step 1: optimize rotation (~34 min)
│       └── run_ptq_eval.sh    # Step 2: quantize + eval (~3 min)
├── project-resq/              # Original ResQ code (submodule, reference)
├── kernels/
│   ├── mixed_gemm/            # SM90 Hopper fused GEMM (WIP, plan exists)
│   └── mixed_gemm_l20/       # SM80 Ampere fused GEMM (working, 1.12-1.19x speedup)
├── third_party/
│   ├── cutlass/               # CUTLASS 4.5 (for kernel development)
│   ├── LLM-Infra-Reference/   # Knowledge base (CUTLASS analysis, Hopper docs)
│   └── ...                    # Other references
└── docs/                      # Design specs, optimization ideas
```

## Current Status

### ProMix PTQ Pipeline (✅ Complete, Self-Sufficient)
- **PPL = 14.72** (matches ResQ exactly)
- Config: `promix/configs/llama-3.2-1b-resq.yaml`
- Full pipeline: Step 0 → Step 1 → Step 2, all runnable from promix scripts
- Key fix: must untie word embeddings + clone lm_head before rotation (see ptq.py)

#### Step 0: PCA Basis (✅ Fully Independent)
- `promix/quantize/basis.py` — no dependency on project-resq at runtime
- Generates `U-wikitext-512-Llama-3.2-1B-Instruct.bin` (numerically identical to ResQ reference)
- Also generates initial random rotation R file and eigenvalue E file
- Runtime: ~20 min, single GPU, ~14 GB VRAM
- Run: `bash promix/scripts/run_basis.sh` or `python -m promix.quantize.basis --config ...`

#### Step 1: Rotation Optimization (✅ Working, Still Depends on ResQ Train Model)
- `promix/quantize/optimize_rotation.py`
- Uses project-resq's `train_utils/modeling_llama_train.py` (Llama with trainable R1/R2 in forward)
- Uses `train_utils/optimizer.py` (SGDG — Stiefel manifold SGD via Cayley transform)
- 100 steps, lr=1.5, cosine scheduler, gradient checkpointing
- Runtime: ~34 min, single GPU
- **Finding: For Llama-3.2-1B, rotation optimization provides no measurable improvement (PPL unchanged)**
- The initial random orthogonal R already works optimally for this small model
- Run: `bash promix/scripts/run_optimize_rotation.sh`

#### Step 2: PTQ Evaluation (✅ Working, Delegates to ResQ)
- `promix/eval/ptq.py` — uses ResQ's `ptq_model()`, `process_args_ptq()`, evaluator
- Run: `bash promix/scripts/run_ptq_eval.sh`

### Mixed-Precision Fused GEMM (✅ Working on SM80)
- `kernels/mixed_gemm_l20/` — INT4+INT8 single-launch fused GEMM
- Correctness: cosine = 1.0 (bit-exact vs reference)
- Performance: 1.12-1.19x vs CUTLASS SM80 2-launch baseline
- Tile: 64×64, Registers: 90-96, Stages: Low=5, High=4
- Benchmark: `python benchmark.py` (on remote H20)

### SM90 WGMMA Fused Kernel (📋 Planned, Not Started)
- Plan at: `.claude/plans/compiled-jumping-unicorn.md`
- Dual accumulator: Phase 1 (INT4 WGMMA) → Phase 2 (INT8 WGMMA) → FP32 combine → FP16 store
- TileShape 128×128×128, warp-specialized with TMA
- SMEM budget: 160 KB (3 stages main + 2 stages high)
- Target: single kernel launch replacing current 2-launch + add

## Pending Work (Priority Order)

1. **Real inference pipeline** — Load quantized model weights → pack INT4/INT8 → feed to fused kernel → end-to-end inference
2. **SM90 WGMMA fused kernel** — Higher performance on H20 (see plan file)
3. **Decouple from project-resq completely** — Rewrite `ptq_model`, `fuse_norm`, `rotation_utils` in promix
4. **lm-eval benchmarks** — Device mismatch issues with lm_eval wrapper (model on cuda, harness expects different device handling)
5. **Larger model validation** — Test on 7B/70B where rotation optimization actually matters

## Build Commands

### Mixed-Precision GEMM Kernel (SM80, on remote H20)

```bash
cd kernels/mixed_gemm_l20
rm -rf build *.so
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python setup.py build_ext --inplace
LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so python benchmark.py
```

### ProMix Full Pipeline (on remote H20)

```bash
cd /vllm-workspace/plaquant
source .venv/bin/activate

# Step 0: PCA basis (~20 min, use free GPU)
CUDA_VISIBLE_DEVICES=6 python -m promix.quantize.basis \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation --nsamples 512

# Step 1: Rotation optimization (~34 min, optional for 1B model)
CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29505 \
    -m promix.quantize.optimize_rotation \
    --config promix/configs/llama-3.2-1b-resq.yaml \
    --output_dir ./rotation --max_steps 100 --learning_rate 1.5

# Step 2: PTQ evaluation (~3 min)
CUDA_VISIBLE_DEVICES=6 LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29507 \
    promix/eval/ptq.py --config promix/configs/llama-3.2-1b-resq.yaml
```

### ResQ Pipeline (project-resq/fake_quant/, for reference only)

```bash
cd project-resq/fake_quant
source /vllm-workspace/plaquant/.venv/bin/activate
bash 0_get_basis.sh       # PCA basis computation
bash 1_optimize_rotation.sh  # rotation optimization
bash 2_eval_ptq.sh        # quantize + eval
```

## Remote Development

### 远程服务器信息
- GPU container: `gemini@general-1295685810-geminijob-0` (8× H20 SM90a, 96GB each)
- 工作目录: `/vllm-workspace/plaquant/`
- venv: `source /vllm-workspace/plaquant/.venv/bin/activate` (Python 3.9, 有 fast_hadamard_transform 等)
- Base 环境: Python 3.12, torch 2.9.1+cu128 (用于编译 CUDA kernel)
- CUDA workaround: `LD_PRELOAD=/usr/local/cuda/lib64/libcudart.so`
- GPU 0-5 通常被其他进程占用 (~83GB)，优先用 GPU 6 或 7

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
- Step 0 (basis) 不需要 torchrun，直接 python 即可
- Step 1 和 Step 2 需要 torchrun（distributed init）

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

### ResQ 三阶段旋转架构

```
Step 0 (PCA Basis U):
  - 收集 calibration 激活的协方差矩阵
  - 特征分解得到按方差排序的基向量
  - full_shared: attn+mlp 共享一个 U，down_proj/value/key_pos 每层独立

Step 1 (Rotation R):
  - 在 U 重排后的子空间内乘可学习正交矩阵 R
  - R1: hidden_dim 空间 (分为 R1_1=main, R1_2=high, R1_0=low)
  - R2: head_dim 空间 (分为 R2_1=main, R2_2=high, R2_0=low)
  - SGDG 优化器在 Stiefel 流形上训练 (Cayley transform)
  - 对 1B 模型无效果，对 7B+ 可能有用

Online Hadamard (固定，不需训练):
  - 仅用于 down_proj 输入 (intermediate_size 太大不适合 full PCA)
  - 使用 fast_hadamard_transform 库做快速在线旋转
```

### Mixed-Precision GEMM Kernel 设计 (SM80)

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

### 生成文件命名规则

```
U-{calib_dataset}-{nsamples}-{model_short_name}.bin    # PCA basis
E-{calib_dataset}-{nsamples}-{model_short_name}.bin    # Eigenvalues
R-high-{h_frac}-low-{l_frac}-sparse-{s_frac}-{model_short_name}.bin  # Rotation
```

示例: `U-wikitext-512-Llama-3.2-1B-Instruct.bin`

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
