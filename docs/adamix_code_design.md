# AdaMix 代码/设计总结（面向论文写作）

> 代码库名 **ProMix**（目录 `promix/`），论文名 **AdaMix**。本文所有 file:line 均按
> 分支 `feat/rd-alloc` 当前工作区实核；格式 `path:line` 可在 IDE 直接跳转。
> 注意：仓库根 CLAUDE.md 中 INT4/INT8、H20/SM80/SM90 相关内容已过时（项目已转向
> FP 格式 + SM120），见文末不一致清单。

## 1. 总览

AdaMix = **NVFP4 (E2M1, block-16) + MXFP8 (E4M3, block-32) 双格式混合精度 PTQ**
+ **单 launch 双 phase 融合 GEMM**（SM120 / RTX Pro 6000 Blackwell）。核心思想：
经 PCA + ring 块对角旋转后，激活按 64/512 通道一"环"为单位做**实测率失真
(measured-ΔD) 分配**——每环实测 NVFP4 与 MXFP8 的量化 MSE 之差，全局单阈值贪心
把预算内收益最大的环升为 MXFP8，其余留在 NVFP4；kernel 侧用一个共享 FP32
accumulator 的双 phase CUTLASS kernel 在一次 launch 内完成
`D = g · (A_lo@B_lo + A_hi@B_hi)`。

三段数据流：

```
┌───────────── 离线量化管线 ──────────────┐   ┌───── 导出/打包 ─────┐   ┌────── 在线推理 ──────┐
│ Step0 PCA basis (basis.py)              │   │ export.py            │   │ vLLM plugin.py        │
│   U/E/R 三产物, full_shared 粒度        │   │  shared-g 重打包     │   │  create_weights       │
│ 旋转底座 (rotation.py+ring_rotation.py) │   │  fused-group unify   │   │  process_weights →    │
│   fuse_norm → U@R1_ring 吸收进权重      │──▶│  weight_pack.py      │──▶│   MixedQuantLinear    │
│ ΔD 环分配 (rd_validity+adaptive.py)     │   │  model.safetensors   │   │  _gemm dispatch:      │
│ GPTQ frozen-scale (gptq.py)             │   │  + k_high map        │   │   fused/smallm/splitk │
│ 量化器 (quant_utils.py) + KV4           │   │  + column_order      │   │   /marlin/GEMV        │
└─────────────────────────────────────────┘   └──────────────────────┘   │ kernel: CodesignFused │
                                                                          └───────────────────────┘
```

## 2. 算法侧（量化管线）

### 2.1 PCA basis（Step 0）

- 设计思想：在校准集上逐层收集激活协方差，特征分解得到按方差**升序**排列的正交基
  U；量化时高方差通道位于尾部，天然形成 `[main | high]` 布局。full_shared 粒度下
  attn+mlp 全层共享一个 U（残差流只需一次全局旋转），down_proj / value / key_pos
  每层独立。
- 关键路径：
  - 入口 `compute_basis`：`promix/quantize/basis.py:95`；CLI `promix/quantize/basis.py:526`。
  - 协方差收集（hook q/k/v/up/down/o_proj 输入输出，double 精度累加
    `X^T X`）：`promix/quantize/basis.py:314-416`；RoPE 后 key 协方差
    `promix/quantize/basis.py:382`。
  - 特征分解 `eigen_decompose`（`torch.linalg.eigh` + 1% 对角阻尼，按特征值升序
    排序）：`promix/quantize/basis.py:68-91`。**CPU LAPACK 回退：本地 checkout 的
    eigen_decompose 全程 cuda，未见 CPU 回退代码 [待核]**（可能在远端
    metricsweep 工作区尚未合并）。
  - 三产物与命名：U/E/R 路径拼接 `promix/quantize/basis.py:145-156`；初始随机
    正交 R（QR 分解符号修正 `random_orthogonal_matrix`）`promix/quantize/basis.py:60-65`，
    R1_1/R1_2（hidden 空间 main/high）与 R2_1/R2_2（head 空间）
    `promix/quantize/basis.py:199-238`。
  - full_shared 分支 `promix/quantize/basis.py:423-475`：共享 `attn_mlp` 基 +
    每层 `layer.{i}.mlp.down_proj` / `self_attn.value`(per-head) /
    `self_attn.key_pos`；同时写入每层 `attn_mlp_var` 方差曲线
    （`promix/quantize/basis.py:436-444`，供 adaptive 高比例使用）。
  - `o_proj_pca: "full_global"` 可选：额外收集 o_proj 输入的 hidden×hidden 协方差
    （`promix/quantize/basis.py:303-310, 404-410`），存 `o_proj_global` 键。
- 论文对应：Method 的 "variance-ordered PCA basis"；U 的升序排列是后文"高方差环在
  尾部 → last-K 切片即 MXFP8 段"约定的来源。

### 2.2 旋转底座：ring512 块对角 R1

- 设计思想：稠密可训练 R1 对大模型收益小且使残差流旋转与"环"分配单位不对齐。
  AdaMix 用**固定 Sylvester-Hadamard 块对角 R1**：`T = U @ R1_ring`，每个
  ring_size×ring_size（部署值 512）块只在环内旋转，使任意环的 FP8/FP4 重分配都
  无需改旋转矩阵（R1 一次构建、全层共享、烘焙进 embedding，在线零开销）。
- 关键路径：
  - 环 R1 构建：`sylvester_hadamard` `promix/quantize/ring_rotation.py:29`，
    `build_uniform_ring_R1` `promix/quantize/ring_rotation.py:53`，
    `compose_rotation(U, R1)` `promix/quantize/ring_rotation.py:86`。
  - 主变换三模式（ring_r1 > identity > dense R1）：`build_main_transform`
    `promix/quantize/rotation.py:24-53`。
  - 旋转吸收进权重：`fuse_basis_to_model` `promix/quantize/rotation.py:234`——
    embed/lm_head 右乘 T（`rotation.py:56-67`），q/k/v/gate/up 输入侧右乘
    （`rotation.py:70-91`），o_proj/down_proj 输出侧左乘 T^T
    （`rotation.py:77-108`）；down_proj 权重输入侧预乘 Hadamard 与运行时
    online Hadamard 配对（`promix/quantize/rotation.py:102-108`）。
  - RMSNorm 融合（旋转可通过的前提）：`fuse_layer_norms`
    `promix/quantize/fuse_norm.py:27-50`（norm 权重吸入相邻 Linear，norm 置 1，
    embed 去均值）。
  - per-head 路径（o_proj）：v_proj 输出 / o_proj 输入按 head_dim 块乘
    `U_value @ R2`：`rotate_ov_proj` `promix/quantize/rotation.py:111`、
    `_apply_had_to_linear` `promix/quantize/rotation.py:122`（含 GQA 分组与
    Qwen3 显式 head_dim 修正 `rotation.py:144-159`）；o_proj 列重排
    `[low|main|high]` 物理连续：`rearrange_columns` / `_rearrange_o_proj`
    `promix/quantize/rotation.py:167-231`（`o_proj_pca: full_global` 时跳过，
    全局基已天然有序 `rotation.py:184-185`）。
  - online Hadamard（仅 down_proj 输入，intermediate 维太大不做全 PCA）：
    `get_hadK` / `matmul_hadU_cuda` `promix/quantize/hadamard.py:22,40`
    （非 2 幂宽度如 14336=28×512 用预存 28×28 Hadamard）；运行时挂载
    `setup_down_proj_hadamard` `promix/eval/ptq.py:410`，激活侧执行点在
    `ActQuantWrapper.forward` `promix/quantize/quant_utils.py:486-505`。
- 论文对应："rotation substrate"（固定、免训练、环对齐）；在线旋转开销 =
  down_proj Hadamard + KV QKRotationWrapper（§2.5）两处。

### 2.3 率失真分配：measured-ΔD ring allocation（核心贡献）

- 设计思想：以环（ring，部署 ring_size=512；默认/早期实验 64）为分配单位，
  **直接实测**每环在部署同款 fake-quant 下的 NVFP4 MSE 与 MXFP8 MSE，定义
  `ΔD_j = D_fp4(j) − D_fp8(j)`（升格收益）；因每环升格成本相同（字节数一样），
  全局把所有层所有环按 ΔD 排序，取 top-n 使平均 FP8 环占比等于预算 r，即单阈值
  最优解。相比方差代理（能量阈值），ΔD 捕捉到"方差高但两格式都量得好/都量不好"
  的环不值得升格这一事实。
- 分数定义与分配器（库层，`promix/quantize/adaptive.py`）：
  - 每环实测 MSE `ring_mse`：`promix/quantize/adaptive.py:69-91`（调用部署同款
    `fake_quantize_nvfp4/mxfp8`，ring_size 为 16/32 的公倍数所以逐环均方差良定义）。
  - `ring_deltaD = mse(nvfp4) − mse(mxfp8)`：`promix/quantize/adaptive.py:94-96`。
  - 全局单阈值分配 `allocate_by_budget(deltaD_per_layer, target_mean_fp8)`：
    `promix/quantize/adaptive.py:166-190`，返回每层 bool mask（True=FP8 环）。
  - 预算→K_high 映射 `masks_to_khigh`（K_high = FP8 环数 × ring_size）：
    `promix/quantize/adaptive.py:193-195`。
  - Fisher 加权细化（损失敏感度 × ΔD，可选）：`ring_fisher_weight`
    `promix/quantize/adaptive.py:123`、`ring_deltaD_weighted`
    `promix/quantize/adaptive.py:139`。
  - 运行点选择（perf/accuracy/knee 三模式 + 可行域 PPL≤floor 且平均 FP8≤1/8）：
    `select_operating_point` `promix/quantize/adaptive.py:239-345`。
  - 配置开关 `rd_alloc` 块（enabled/ring_size/mode/target，缺省关闭=零回归）：
    `rd_alloc_config` `promix/quantize/adaptive.py:205-227`；YAML 样例
    `promix/configs/llama-3.2-3b-mxfp8-nvfp4.yaml:51-67`。
- 统计收集与端到端驱动（`promix/scripts/rd_validity.py`，A/B/C 验证门，被所有
  下游脚本复用）：
  - `collect_stats`：对 q/k/v/gate/up/down_proj 的内层 nn.Linear 注册
    forward-pre hook，在**旋转后、量化前**（down_proj 为 online-Hadamard 后）的
    激活上逐环累加 `e4=(x−q4)²`、`e8=(x−q8)²` 与通道均值/方差，产出
    `deltaD = mse4 − mse8`：`promix/scripts/rd_validity.py:96-154`
    （mse4/mse8/deltaD 计算在 `rd_validity.py:146-148`）。o_proj 固定 uniform
    1/8 不参与重分配（控制变量，`rd_validity.py:22-25, 61`）。
  - `measured_alloc`（调 `allocate_by_budget`）：`promix/scripts/rd_validity.py:195-204`；
    对照组 `uniform_rings` `rd_validity.py:160`、方差代理 `variance_alloc`
    （tau 二分到预算）`rd_validity.py:165-192`。
  - `clamp_chosen`：每 Linear 至少保 1 个 NVFP4 环（纯 MXFP8 无 kernel 路径、
    两级权重 scale 空段崩溃），r=1.0 实际为 all-but-one-ring：
    `promix/scripts/rd_validity.py:207-224`。
  - `build_eval`：按 chosen 做环粒度列重排（非选中环在前、FP8 环在后），写
    `w._column_order` + `reorder_before_quant` + `model._rd_khigh_override`，
    再 GPTQ + configure：`promix/scripts/rd_validity.py:231-266`。
  - override 注入点（优先级最高，逐 Linear 覆盖 K_high）：
    `promix/eval/ptq.py:253-257`（`configure_quantizers` 内）。
- 预算 r ↔ K_high：`budget_frac` → `budget = round(r · total_rings)` →
  `allocate_by_budget` → 每 Linear `K_high = |chosen| × ring_size`；导出侧入口
  `promix/vllm_integ/export.py:388-461`（其中 **fused-group unify**
  `export.py:430-457`：vLLM 合并 GEMM（qkv、gate_up）成员必须同一 K split，
  按成员最大环数对齐、不足的按各自 ΔD 降序补、超出的裁回，cap 至 nr−1）。
- 论文对应：Method "measured rate-distortion allocation"（分数=ΔD、单位=环、
  等成本→全局单阈值）；Experiments 的 A/B/C gate（uniform vs variance-proxy vs
  measured）即 `rd_validity.py` 本体。

### 2.4 量化器（NVFP4 / MXFP8 fake-quant 与权重两级 scale）

- 设计思想：`bits` 字段可为字符串 `"nvfp4"/"mxfp8"`（`SUPPORTED_FP_FORMATS`
  `promix/quantize/quant_utils.py:29`），激活按 `[low|main|high]` 段路由到各自
  FP 格式；block scale 全部在 forward 内按 spec 推导（无需 find_params 预拟合）。
  fake-quant 是 kernel 位精确参考（kernel packer 对同一输入必须 bit-identical）。
- 关键路径（均在 `promix/quantize/quant_utils.py`）：
  - `fake_quantize_nvfp4`（block-16，scale=absmax/6 **向上取整到 E4M3 格点**
    `_ceil_to_value`，元素 round-to-nearest-even）：`quant_utils.py:1075-1139`；
    `fake_quantize_mxfp8`（block-32，E8M0 向上取 2 幂 `_e8m0_round_up`）：
    `quant_utils.py:1025-1072`。
  - `_ceil_to_value` 分块实现（4M 元素/块，避免 (numel,V) 比较矩阵在 70B 形状上
    ~17 GiB OOM，位级等价）：`quant_utils.py:653-677`。
  - **act-MSE scale 搜索**（Win1）：ceil 基准 + 下方 N 个 E4M3 格点（可选上方
    STEPS_UP 个）选逐块 MSE 最优；开关/深度 `ACT_MSE_SCALE_SEARCH/ACT_MSE_FAST/
    ACT_MSE_STEPS(_UP)` `quant_utils.py:704-718`，搜索体
    `quant_utils.py:1111-1132`；CUDA 同款快速实现 `kernels/mixed_gemm_sm120/act_mse.cuh`。
  - **weight-MSE 搜索**（Win2，仅重打包零前向开销）：`WEIGHT_MSE_SCALE_SEARCH`
    `quant_utils.py:721`，`_mse_best_scale` `quant_utils.py:820`，
    `select_block_scales_nvfp4/mxfp8` `quant_utils.py:934/896`。
  - **两级权重 scale**（Round-6）：NVFP4 有效 scale = 每张量全局 g × 每 16 通道
    E4M3 micro；MXFP8 high 段共享同一 g（micro=e8m0(ideal/g) 对任意 g 都是 2 幂），
    g 由 fused kernel 的 epilogue alpha 一次性携带——
    `WEIGHT_TWO_LEVEL` `quant_utils.py:741`，
    `_weight_two_level_nvfp4_eff` `quant_utils.py:745-778`，
    `_weight_two_level_mxfp8_eff` `quant_utils.py:781-796`。
  - 激活量化器 `ActQuantizer`（FP 字符串分支 + `[low|main|high]` 段切分 + KV
    per-head groupsize reshape）：`quant_utils.py:164-398`（FP forward 分支
    `quant_utils.py:229-291`）；Linear 包装 `ActQuantWrapper`
    （online Hadamard / reorder_before_quant / 融合生产者开关）：
    `quant_utils.py:401-532`。
  - KV 对齐工具 `kv_fp_high_length`（见 §2.5）：`quant_utils.py:97-136`。
- **GPTQ 集成**（frozen-block-scale FP GPTQ，spec §11 风格）：
  - block scale 在**未修正**权重上一次性冻结（`WeightQuantizer.find_params` FP
    分支 `promix/quantize/kv_quant.py:54-101`，两级 g 也在此产出并经
    `global_scale` 传给 high 段 `promix/quantize/gptq.py:60-68`）；
  - 逐列 Hessian 修正值对**同一冻结 scale** 做最近邻取整：
    `quantize_column_with_frozen_scale` `promix/quantize/quant_utils.py:981-1022`
    （WeightQuantizer 侧封装 `kv_quant.py:160-181`）；
  - 主循环 `GPTQ.fasterquant` `promix/quantize/gptq.py:52`（三段 low/main/high
    分段路由 FP 冻结取整 `gptq.py:160-200` 附近；FP 禁 groupsize
    `gptq.py:122-130`）；驱动 `gptq_fwrd` `promix/quantize/gptq.py:232`；
    配置入口 `run_gptq_if_enabled` `promix/eval/ptq.py:271`。
  - 打包侧用 GPTQ 冻结 scale 而非重推（bit-faithful）：`_frozen_scales_for`
    `promix/quantize/full_network.py:119` → `pack_weight_mixed(frozen_*_scale=...)`。
- 论文对应：Implementation 的量化格式定义 + "scale 向上取整 / MSE 搜索 /
  两级全局 g" 三个精度 lever；GPTQ 小节的 "frozen scale + Hessian 修正可交换"。

### 2.5 KV cache 量化（rd-kv4）

- 设计思想：K cache 必须在 RoPE 之后旋转（RoPE 与通道旋转不可交换），故运行时
  在 `apply_rotary_pos_emb` 后插入每层旋转 `U_key_pos @ R2`（INT8 量化存储），再
  对 K 做每 head 混合 FP 量化；V cache 由 v_proj 的 out_quantizer 处理（离线
  旋转已吸收）。FP 块对齐难点：per-head 下 `int(head_dim·hf)`（16 或 8）小于
  MXFP8 块 32 → `kv_fp_high` 策略 "up"（高段上取整到 32 的倍数）或 "zero"
  （纯 NVFP4 KV）。
- 关键路径：
  - `QKRotationWrapper`（pre-rotation INT8 → matmul 旋转 → per-head/per-token
    K 量化）：`promix/quantize/kv_quant.py:218-301`（forward
    `kv_quant.py:262`）。
  - monkeypatch 注入（替换 forward globals 里的 `apply_rotary_pos_emb`）：
    `add_qk_rotation_wrapper` `promix/quantize/kv_quant.py:318-330`。
  - 装配入口 `setup_k_quant`（读 U/R 文件、组 R2、INT8 量化旋转矩阵、FP 时调
    `kv_fp_high_length` 对齐）：`promix/quantize/kv_quant.py:333-424`
    （FP 对齐分支 `kv_quant.py:355-362`）。
  - 对齐策略本体 `kv_fp_high_length`（up: 高段 ceil 到 32 倍数且 main%16==0，
    否则 ValueError）：`promix/quantize/quant_utils.py:97-136`。
  - V 路径（v_proj out_quantizer，groupsize=head_dim 的 FP 段内切分）：
    `promix/eval/ptq.py:151-174` + `ActQuantizer.forward` 的 FP groupsize
    reshape `promix/quantize/quant_utils.py:243-250`。
- 论文对应：KV4 小节；"在线旋转开销"即 QKRotationWrapper 的两次 matmul +
  一次 INT8 预量化。

## 3. 系统侧（推理集成）

### 3.1 导出 / 打包

- `promix/vllm_integ/export.py`：构建量化模型（dense 路径 `prepare_ptq_model`
  或 ring+measured 路径 `export.py:388-463`，含 §2.3 的 fused-group unify
  `export.py:430-457`）→ **shared-g 重打包**（vLLM 合并组 qkv/gate_up 各成员共
  用一个每张量 g，`pack_all_targets_shared_g` `export.py:90-171`，依赖
  `promix/vllm_integ/shared_g.py` 的 `rederive_frozen_scales_shared_g`）→
  real-kernel wiki PPL 复核（`export.py:494-505`）→ 写 checkpoint
  `export_checkpoint` `export.py:189-287`：`model.safetensors`（每 Linear
  `weight_lo_packed / weight_lo_scale / weight_hi_bytes / weight_hi_scale /
  weight_global_scale` + 合并前缀 `column_order`）、`config.json` 的
  `quantization_config{quant_method:"promix", k_high map, reordered_prefixes,
  act_mse, ...}`（`export.py:251-276`）、`t2_reference.pt` 逐 token logprob
  快照（T2 位精确 gate 的参考，`t2_snapshot` `export.py:300-326`）。
- `promix/quantize/weight_pack.py`：离线权重打包。
  - spec 同款量化器（deq 与 fake_quantize_* 位一致、codes/SF 与 kernel 位一致）：
    `quantize_nvfp4_spec` `weight_pack.py:45-92`、`quantize_mxfp8_spec`
    `weight_pack.py:95-134`（两级 g 分解 micro=eff/g `weight_pack.py:77-83,121-128`）。
  - `PackedWeight` 数据类（fused 路径 4 buffer + decode-GEMV 路径 4 buffer +
    g_w + W_deq 参考）：`weight_pack.py:137-163`。
  - `split_k`（K_high=round(hf·K)，断言 K_low%16==0、K_high%32==0）：
    `weight_pack.py:166-175`；主函数 `pack_weight_mixed`
    `weight_pack.py:178-284`（SFB 交织用 kernel 自己的 prepacker
    `cd.prepack_sf_fp4_b/fp8_b`，K_high==0 空 buffer + 单 phase 路径
    `weight_pack.py:249-260`，K_low==0 明确拒绝 `weight_pack.py:212-216`）。

### 3.2 运行时模块：MixedQuantLinear

`promix/quantize/mixed_linear.py:200` — 真 kernel Linear，持有 PackedWeight，
每 forward：切分激活 → 量化+pack（CUDA 融合）→ 融合 GEMM。`forward`
`mixed_linear.py:528-567` 的路由优先级与各路径触发条件：

| 路径 | 触发条件 | 作用 | 代码 |
|---|---|---|---|
| marlin (W4A16) | `_MARLIN_TABLE` 实测窗口内 2≤M≤48，`attach_marlin` 已挂载 | Marlin fp4 TC GEMM（低段）+ MXFP8-marlin high companion / hi_add GEMV，激活不量化 | `marlin_route` `mixed_linear.py:656`，表 `mixed_linear.py:152-181`，`_marlin_forward` `mixed_linear.py:672` |
| decode GEMV | M==1，或 1<M≤min(16,按形状 2/4) 且 w4a16 模式 | 带宽受限 W4A16/W4A4 GEMV，流式读 packed 权重 | `gemv_route` `mixed_linear.py:601`，`_decode_gemv` `mixed_linear.py:356` |
| splitk / streamK | `_SPLITK_TABLE` 窗口（down/gate_up 小 M） | K 维跨 CTA 切分 + FP32 workspace 归约（非位一致，cos≈1） | `_splitk_cfg` `mixed_linear.py:449`，表 `mixed_linear.py:106-110` |
| smallm cfg1 | `_SMALLM_TABLE` 窗口（128×64 tile 赢的 M 区间） | 小 M 翻倍 CTA 数填满 SM（位一致） | `_smallm_cfg` `mixed_linear.py:437`，表 `mixed_linear.py:77-84` |
| bias/castfold | 其余 GEMM 路（含 M≥1024 prefill） | epilogue 内 bf16→fp16 cast + bias 折叠，省一次 [M,N] HBM 往返（位一致） | `_gemm` `mixed_linear.py:462-526`（castfold 分支 `mixed_linear.py:505-517`） |
| single_phase | `K_high==0`（纯 NVFP4 Linear，低预算分配产物） | 单 phase NVFP4 GEMM（Phase-2 编译期裁掉），g 仍走 epilogue alpha | `mixed_linear.py:245,466-476` |

激活 pack：`_pack_activation` `mixed_linear.py:394-435`（qpfm 单 launch 双段 /
qpf 原位切片 / torch 慢路径）；融合生产者入口：`forward_swiglu_pack`
（silu·up→Hadamard→pack 单 kernel，down_proj）`mixed_linear.py:578`、
`forward_had_pack` `mixed_linear.py:719`、`forward_gather_pack`（列重排折进
pack 的 load）`mixed_linear.py:742`、`forward_norm_pack`（residual+RMSNorm+pack）
`mixed_linear.py:569`。接线函数 `wire_real_kernel`
`promix/quantize/full_network.py:356`（替换 wrapper.module、关 fake-quant）。

### 3.3 vLLM 插件

- 注册：`@register_quantization_config("promix")` `promix/vllm_integ/plugin.py:79`，
  经 `vllm.general_plugins` entry point，无 vLLM fork；checkpoint 的
  `quant_method:"promix"` 触发。
- `PromixLinearMethod`（每合并前缀 qkv/o/gate_up/down）：
  - `create_weights`（按 k_high map 注册 packed 参数 + 合并模块 column_order 的
    全拷贝 loader）：`plugin.py:178-223`；
  - `process_weights_after_loading`（JIT 加载 ext、SF prepack 到交织布局、构建
    MixedQuantLinear、派生 decode-GEMV scale、Marlin 低/高段 load-time 重打包、
    注册 graph-safe 自定义算子 handle）：`plugin.py:225-384`；
  - `apply` → `torch.ops.vllm.promix_linear`（CUDA-graph/torch.compile 兼容，
    `PROMIX_EAGER_APPLY=1` 旁路）：`plugin.py:420-424`；eager 体
  `_apply_eager`（down_proj: 融合 Hadamard+pack 或 torch Hadamard；reordered:
  gather 折叠 pack 或 torch gather；最后 `mql(x)`）：`plugin.py:386-418`。
- kernel JIT 装载（act-MSE 编译 flag 与参考实现对齐）：
  `promix/vllm_integ/kernels.py:23-55`；自定义算子注册 `promix/vllm_integ/ops.py`；
  norm→pack 的编译期 fusion pass `promix/vllm_integ/fusion_pass.py`；serving 跑法
  见 `promix/vllm_integ/bench_serve.py` 与 export.py docstring（`export.py:20-27`）。

## 4. 算子侧（kernel，kernels/mixed_gemm_sm120/）

### 4.1 CodesignFusedKernel 结构

- 本体 `codesign_fused_kernel.hpp:60`（`CodesignFusedKernel` 模板）：CUTLASS
  `sm90_gemm_tma_warpspecialized_cooperative.hpp` 的裁剪 fork，驱动 SM120
  block-scaled collective。两个关键改动（头注释 `codesign_fused_kernel.hpp:1-27`）：
  1. **双 mainloop collective 顺序执行、共享 accumulator**：Phase-1 NVFP4
     （ClearAccum=true）跑 K_low，Phase-2 MXFP8（ClearAccum=false）在**同一**
     FP32 fragment 上继续累加 K_high，acc 不出寄存器，一次 epilogue 存 bf16/fp16
     ——load 序列 `codesign_fused_kernel.hpp:484-495`，mma 序列
     `codesign_fused_kernel.hpp:559-569`；两 collective 的 accumulator
     fragment 形状/类型 static_assert 一致 `codesign_fused_kernel.hpp:98-115`；
     SMEM 以 union 复用（Phase-1 排空后 Phase-2 覆写）
     `codesign_fused_kernel.hpp:164`。
  2. `DualPhase` 模板 bool：同一 glue 编译出单格式 NVFP4 kernel
     （`single_nvfp4_gemm_prepacked` 的来源）`codesign_fused_kernel.hpp:23,59`。
- block-scaled mainloop（SM120 SF 交织 MMA collective，CUTLASS 派生保护文件）：
  `codesign_sm120_bs_mma.hpp`。
- castfold/bias epilogue：`codesign_bias_epilogue.hpp`（epilogue 内
  `half(float(bf16(acc))+bias)`，位一致替代 GEMM 后两次 elementwise）。
- split-K/stream-K fork：`codesign_splitk_kernel.hpp`。

### 4.2 入口 .cu 一句话职责

| 文件 | 职责 |
|---|---|
| `codesign_fused_ext.cu` | 生产主入口：双格式融合 GEMM（`fused_gemm_prepacked` :596、`single_nvfp4_gemm_prepacked` :674）+ load-time SF prepack（`prepack_sf_fp4_b` :405） |
| `codesign_fused_bias_ext.cu` | r26 lever A：bias+cast epilogue 变体（`fused_gemm_prepacked_bias`） |
| `cd_smallm_ext.cu` | r2：小 M 128×64 tile 配置变体 |
| `cd_splitk_ext.cu` | r3：split-K/stream-K 变体（decode b8/b16 延迟地板） |
| `codesign_nvfp4_ext.cu` | STAGE-1 去风险：纯 NVFP4 走同 glue 验证 TC duty |
| `quant_pack_float_ext.cu` | 激活 float→packed codes+交织 SFA 单 pass（qpf） |
| `quant_pack_fused_ext.cu` | qpfm：一次 launch 同产 NVFP4+MXFP8 双段（含 gather 变体） |
| `hadamard_pack_ext.cu` / `hadamard28_pack_ext.cu` | down_proj 融合 SwiGLU+Hadamard(+28×512 非 2 幂)+量化 pack |
| `add_rmsnorm_pack_ext.cu` / `vllm_rms_quant_pack_ext.cu` | residual-add+RMSNorm+pack 生产者（promix 参考版 / vLLM 位精确版） |
| `decode_gemv_ext.cu` | M=1/小 M 带宽型 W4A16/W4A4 GEMV（+_sf8、hi_add、批量 mb 变体） |
| `lmhead_mxfp8_ext.cu` | lm_head 纯 MXFP8 W8A8 GEMM（W4 混合在未旋转 lm_head 上 PPL 不过门） |
| `nvfp4_gemm_ext.cu` / `mxfp8_gemm_ext.cu` | 单格式基线（CUTLASS ex.79 配置），2-launch baseline 组件 |
| `fused_gemm_ext.cu` / `mixed_gemm.py` | 早期单 host-launch / 2-launch 混精基线（对照用） |
| `cd_cfg_ext.cu` / `pure_presf_ext.cu` / `tile_probe_ext.cu` 等 | 调优 harness（grid search / probe），非 serving 路径 |

### 4.3 调优史与配置点

- tile/stage 常量与 per-shape 启发式 dispatch：`config_dispatch.hpp`
  （ConfigTag 枚举 NVFP4/MXFP8/CODESIGN `config_dispatch.hpp:55-71`；生产双格式
  配置 = `CdTag::T128x128_NV4MX2`，tile 128×128×128、NV stages=4 / MX stages=2，
  小 M 备选 128×64；SM120 约束：SMEM ~101 KB/CTA、cooperative 要求 Tile-M≥128、
  无 cluster multicast→ClusterShape 1×1×1，`config_dispatch.hpp:20-26`）。
- per-shape 运行时 dispatch 窗口（Python 侧，实测表）：
  `_SMALLM_TABLE` `promix/quantize/mixed_linear.py:77`、`_SPLITK_TABLE`
  `mixed_linear.py:106`、`_MARLIN_TABLE/_MARLIN_HI_TABLE` `mixed_linear.py:152/172`
  ——均为 CUDA-graph replay 下 probe 实测（gemm_probe / splitk_probe /
  marlin_probe，`promix/vllm_integ/`）。
- 平台：以 **SM120（RTX Pro 6000）** 为主；**SM121（DGX Spark GB10）** 已验证
  单 launch 通路（profiler 12 代表点均单 CodesignFusedKernel），KDA 调优进行中
  （不在此下结论）。

## 5. 评测基建（简短）

| 脚本 | 职责一行 |
|---|---|
| `promix/scripts/rd_mmlu.py` | 单点精度评测：ring512+measured 构建 → wiki PPL 复现门 → lm-eval MMLU 5-shot / 0-shot 6 任务 / C4（`rd_mmlu.py:1-31`） |
| `promix/scripts/metricsweep.sh` | 11 点 r∈{0..1} 精度-比例曲线扫（调 rd_mmlu.py `--budget_frac r`），幂等+失败跳过+逐点 commit（`metricsweep.sh:1-45`） |
| `promix/eval/rd_oppareto_ours.py` | kernel 延迟-r 全曲线（走 `MixedQuantLinear._gemm` 真实 serving dispatch，cudaEvent+graph-replay 双计时，`rd_oppareto_ours.py:1-17`）；同族 `rd_oppareto_torch.py`（torch 参考）/`rd_oppareto_agg.py`（汇总） |
| `promix/scripts/rd_scale.py` | 70B offload 安全版精度点（去掉全模型 `.to(DEV)`，逐层 offload eval + offload 版 `collect_stats_offload` `rd_scale.py:119`；小模型须 digit 级复现 acctune 锚点，`rd_scale.py:1-35`） |
| `promix/scripts/rd_validity.py` | A/B/C 分配有效性门（§2.3），全家族构建机器的单一事实来源 |
| `promix/eval/rd_bench_m8192_*.py` / `promix/scripts/rd_bench_m8192_all.sh` | M=8192 GEMM 对 vLLM/MicroMix/SharQ/ARCQuant 的 head-to-head |

数据目录对照：精度曲线 `docs/rd_metricsweep_data/`（metricsweep.sh 写入）；
acctune 锚点 `docs/rd_acctune_data/`；MMLU 报告 `docs/rd_mmlu_report.md`；
调优/实验叙述 `docs/rd_accuracy_tuning.md`、`docs/paper_experiments.md`。

## 6. 论文叙事映射表

| 论文概念 | 代码实体 | 一句话 |
|---|---|---|
| 精度比例 r | `budget_frac` → `allocate_by_budget(target_mean_fp8=r)` → 每 Linear `K_high=|chosen|·ring_size`（`adaptive.py:166`, `export.py:420`） | r 是全局平均 FP8 环占比，不是逐层常数 |
| 分配单位（环） | `ring_size`（实验 64，部署 512）`rd_validity.py:58` / export `--ring_size 512` | 16/32 的公倍数 ⇒ 环内 FP 块对齐自动成立 |
| 分配分数 | `deltaD = mse4 − mse8`（`rd_validity.py:146-148`, `adaptive.py:94`） | 实测升格收益，等成本下全局单阈值即最优 |
| 分配器 | `allocate_by_budget` `adaptive.py:166` + `clamp_chosen` `rd_validity.py:207` | 排序-取 top-n；每 Linear 保 ≥1 NVFP4 环 |
| 代理基线（消融） | `variance_alloc`（能量阈值 tau 二分）`rd_validity.py:165`；`derive_khigh` `adaptive.py:36` | "方差代理 vs 实测 ΔD" 对照 |
| 旋转底座 | `build_uniform_ring_R1` `ring_rotation.py:53` + `build_main_transform` `rotation.py:24` | U@环块对角 Hadamard，离线烘焙、零在线开销 |
| 在线旋转开销 | down_proj Hadamard（`quant_utils.py:486-505`/`hadamard_pack_ext.cu`）+ KV `QKRotationWrapper` `kv_quant.py:218` | 全模型仅此两处运行时旋转 |
| 双格式量化 | `fake_quantize_nvfp4/mxfp8` `quant_utils.py:1075/1025` | kernel 位精确参考实现 |
| scale 精修 | act-MSE `quant_utils.py:1111`、weight-MSE `quant_utils.py:820`、两级 g `quant_utils.py:745` | 三个正交精度 lever，env 可逐一消融 |
| FP GPTQ | frozen-scale 列取整 `quant_utils.py:981` + `gptq.py:52` | 块 scale 冻结使 GPTQ 与块量化可组合 |
| KV4 | `setup_k_quant` `kv_quant.py:333` + `kv_fp_high_length` `quant_utils.py:97` | post-RoPE 旋转 + 块对齐的 per-head 混合 KV |
| 单 launch 融合 GEMM | `CodesignFusedKernel` `codesign_fused_kernel.hpp:60` | 双 phase 共享 FP32 acc，一次 epilogue |
| K_high=0 退化 | `single_nvfp4_gemm_prepacked` `mixed_linear.py:466` | 低预算分配下 Linear 可整段纯 NVFP4 |
| 端到端 serving | `plugin.py:79` + `MixedQuantLinear.forward` `mixed_linear.py:528` | vLLM 插件按 M/形状窗口 dispatch |
| 精度-延迟权衡图 | `metricsweep.sh` × `rd_oppareto_ours.py` | 同一 r 轴上的两条曲线 |

## 附：与 CLAUDE.md 的主要不一致（以代码为准）

1. CLAUDE.md 全文仍写 INT4/INT8 + ResQ/CUTLASS SM80/SM90；实际主线为
   NVFP4+MXFP8 + SM120 `kernels/mixed_gemm_sm120/`（SM80/SM90 目录是历史遗留）。
2. "fuse_norm/rotation 委托给 project-resq"已过时：二者均为 promix 自含实现
   （`fuse_norm.py`、`rotation.py`）；optimize_rotation 也已用 in-repo
   `promix/train/modeling_llama_train.py` + `promix/train/optimizer.py`（SGDG）。
3. 远程环境（H20 容器、tmux 连接方式）过时；现平台为 RTX Pro 6000
   （10.208.130.90）与 DGX Spark GB10（见 memory）。
4. "Pending: 真实推理管线 / SM90 WGMMA" 已被 SM120 真 kernel + vLLM 插件实现取代。
5. 用户提及的 basis "CPU LAPACK 回退" 在本地 checkout 不存在 [待核]
   （`basis.py:68-91` 的 eigh 全程 CUDA；疑在远端 metricsweep 工作区未合并）。
