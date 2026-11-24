# M32 对齐与执行指导 — Motivation §3.2 ↔ M31/主实验基建

日期：2026-11-24 ｜ 任务书：`docs/paper_motivation_3_2_experiments.md`（协议权威）｜ 前置：`docs/motivation_3_1/alignment.md`
问题：固定分配规则与精度 mask 后，精度组内正交变换是否还改善量化质量（Direct vs Rotated，两条件、别无其他）。

## 1. 骑在 M31 上（复用是本任务的主结构）

| 件 | 来源 | 说明 |
|---|---|---|
| PCA 表示 / AV 分数 / 精度 mask | M31 已产（scores/、allocations.jsonl） | M32 冻结复用；Direct 与 Rotated **共用同一 mask**（任务书 §3） |
| Direct 一侧结果 | = M31 AV-within（r=0.1..0.9 子集） | 复用前逐项核验：权重 PTQ 配方、mask、表示、fit 数据、量化布局全同（不是只对 PPL 数）；核验记录进 manifest |
| runner | 扩展 `rd_m31.py`：`--variant rotated --rotation_seed N` | Rotated=每层 H/L 两组各乘确定性随机正交阵（float64 Gaussian→QR→对角符号校正，seed 派生 model/module/ratio/group/rotation_seed），显式参考实现，权重侧 V_g R_g 配对，披露非零开销 |
| 层误差 hook (E_l) | M31 §7.2 同口径 | 直接复用 |
| 新增统计器 | — | §5.1 幅度（p50/90/99/99.9/max + block max/RMS + 组能量比）与 §5.2 激活误差 A（H/L 分开，float64 累计） |

## 2. 任务书 §8 陷阱（执行 gate）

- 禁用 ring 路径（`ring_r1` 优先级高于 identity，必须显式关）；`identity_main_rotation=True` ≠ 关闭 R2/key/value/down 变换 — 每 run 输出 **transform manifest** 逐项列两侧共同开启/被实验改变的变换
- 不复用固定 r=1/8 的旧 R 矩阵扫比例；任意合法配额维度都要能构造 R（QR 通用维度，不用截断 Hadamard）
- 无量化等价性：正交残差、组能量保持、层输出、整网 logits/PPL 桥接 → equivalence.json

## 3. 矩阵与排班

- 主矩阵：2 模型 × 9 比例（0.1..0.9 非退化）× Direct/Rotated，rotation_seed=0 → 36 逻辑配置，Direct 复用后新跑 ~18-27 点
- 附加：Rotated 在 r=0.1/0.5/0.9 补 rotation_seed=1,2（+12）；Direct 不假装三样本
- 排班：与 M31 同批 GPU4-7 共卡 lane（精度实验可共卡）；先一个模型中间比例正确性检查，再放量
- 输出独立 `docs/motivation_3_2/`（任务书 §9 全套文件），绝不写主实验目录

## 4. 跟踪表 M32

| 模型 | Direct 复用核验 | Rotated ×9 (seed0) | seeds1/2 ×3r | A/幅度统计+图A/B |
|---|---|---|---|---|
| Llama-3.2-3B | ⬜ | ⬜ | ⬜ | ⬜ |
| Llama-3-8B | ⬜ | ⬜ | ⬜ | ⬜ |

验收 gate：两侧 mask 与 basis 逐字节一致；变换只在组内；无 ring/allocator 混入；无量化等价过；四类指标可从原始累计数重算；层间 std 与 seed std 分开报；**不以 Rotated 胜出为通过条件**，不利结果照报。

## 5. 执行方式

追加给 M31 同一执行 agent（基建在其手里）；顺序 = M31 3B 四点审查过关 → 放量 M31 同时实现 Rotated 分支 → M32 矩阵。主 loop 每 tick harvest + 盯表。
