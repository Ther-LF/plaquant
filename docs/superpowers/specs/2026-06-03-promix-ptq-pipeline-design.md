# ProMix PTQ Pipeline — Design Spec

**Author**: 罗凡  
**Date**: 2026-06-03  
**Status**: Approved

## Goal

在 `plaquant/promix/` 下一比一复刻 ResQ 的完整 PTQ 流程（PCA → rotation → quantize → eval），保持与 project-resq 的结果可对比，同时将 model 层独立为 plugin 模式方便后续扩展。

## Scope (Phase 1)

- PTQ pipeline 全流程（get_basis → optimize_rotation → ptq eval）
- 支持 Llama-3.2-1B（第一个模型）
- 评测：wikitext perplexity + lm-eval benchmarks（mmlu, boolq, piqa 等）
- 与 project-resq 结果做对比验证

## Out of Scope

- 真实推理（用 mixed_gemm kernel）— Phase 2
- 其他模型（Qwen2, Llama-3-8B）— 在 Phase 1 验证通过后添加
- 训练/QAT — 不做

## Directory Structure

```
promix/
├── __init__.py
├── configs/
│   └── llama-3.2-1b-resq.yaml
├── models/
│   ├── __init__.py
│   ├── base.py          # ModelPlugin 基类接口
│   └── llama.py         # Llama 系列实现
├── quantize/
│   ├── __init__.py
│   ├── basis.py         # PCA 基计算
│   ├── rotation.py      # 旋转矩阵优化 + 应用
│   ├── quant_utils.py   # ActQuantWrapper, 量化函数
│   ├── hadamard.py      # Hadamard 变换
│   └── fuse_norm.py     # RMSNorm 融合到权重
├── eval/
│   ├── __init__.py
│   ├── ptq.py           # PTQ 主入口
│   ├── perplexity.py    # wikitext PPL
│   └── lm_eval_wrapper.py
├── data/
│   ├── __init__.py
│   └── calibration.py   # 校准数据加载
└── scripts/
    ├── run_basis.sh
    ├── run_rotation.sh
    └── run_ptq_eval.sh
```

## Model Plugin Interface

```python
class ModelPlugin:
    model_type: str
    
    def get_layers(self, model) -> list
    def get_projections(self, layer) -> dict  # {"q_proj": module, ...}
    def get_embedding(self, model)
    def get_lm_head(self, model)
    def get_norm(self, model)  # final norm
    def get_hidden_size(self, model) -> int
    def get_intermediate_size(self, model) -> int
    def patch_forward(self, model, config)
```

## Pipeline Steps

### Step 0: PCA Basis (promix/quantize/basis.py)
- 对应：project-resq/fake_quant/get_basis.py
- 输入：pretrained model + calibration data (wikitext, 512 tokens × 128 samples)
- 输出：`U-wikitext-512-{model_name}.bin`（per-layer PCA 基矩阵）
- 运行：`bash scripts/run_basis.sh`

### Step 1: Rotation Optimization (promix/quantize/rotation.py)  
- 对应：project-resq/fake_quant/optimize_rotation.py
- 输入：model + PCA basis + calibration data
- 输出：`R-high-{frac}-low-{frac}-sparse-{frac}-{model_name}.bin`
- 运行：`bash scripts/run_rotation.sh`

### Step 2: PTQ Eval (promix/eval/ptq.py)
- 对应：project-resq/fake_quant/ptq.py
- 流程：load model → fuse norm → apply rotation → add quant wrappers → eval
- 输出：perplexity + benchmark scores
- 运行：`bash scripts/run_ptq_eval.sh`

## Config Format (YAML)

```yaml
model:
  name: "unsloth/Llama-3.2-1B-Instruct"
  type: "llama"

quantize:
  w_bits: 16
  a_bits: 4
  high_bits: 8
  low_bits: 2
  high_fraction: 0.125
  low_fraction: 0.0
  a_asym: true
  k_bits: 4
  v_bits: 4
  k_groupsize: 64
  v_groupsize: 64
  rotate_mode: "resq"
  rotation_granularity: "full_shared"

paths:
  basis: "./rotation/U-wikitext-512-Llama-3.2-1B.bin"
  rotation: "./rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin"

eval:
  tasks: ["mmlu", "boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]
  batch_size: 1
  max_length: 2048
```

## Validation Criteria

- wikitext perplexity 与 project-resq 的结果差异 < 0.1
- lm-eval benchmark scores 与 project-resq 一致（±0.5%）
- 同一份 rotation/basis 文件两边通用（格式兼容）

## Implementation Strategy

1. 先 scaffold 目录结构
2. 从 project-resq 复制核心逻辑，逐文件重构到新结构
3. 先跑通 Step 2（PTQ eval），因为可以直接用 project-resq 已有的 basis + rotation 文件
4. 再跑通 Step 0 和 Step 1
5. 对比验证
