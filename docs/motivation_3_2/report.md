# M32 Report — Motivation §3.2 固定精度分配下的组内正交变换

日期: 2026-11-25 ｜ 任务书: `docs/paper_motivation_3_2_experiments.md` ｜ 数据: `docs/motivation_3_2/{runs.csv,layer_metrics.csv,magnitude_stats.csv,equivalence.json,data/}` ｜ 图: `motivation-representation-{distribution-3b,distribution-8b,quality}.{pdf,png}`

## 1. 完成度与 gate

- 主矩阵 2 模型 × 9 比例 × {Direct, Rotated seed0} = 36 逻辑配置全落点（Direct 18 点复用 M31 AV-within）；附加 Rotated rotation_seed=1,2 @ r∈{0.1,0.5,0.9} ×2 模型 = 12 点全落。failures.csv 仅早期共卡 OOM（已重试成功）。
- **Direct 复用核验**: `verify32` 对两模型 × 9 比例逐项核对 mask alloc_hash / representation / PTQ recipe 与 M31 run json 全等（不只对 PPL）→ all_match=True（direct_reuse_verification_{3b,8b}.json，引用于 manifest）。
- **无量化等价性**（equivalence.json, r=0.5 rot0）: 正交残差 ≤9.9e-15（fp64）；组能量比 ∈[0.99999,1.00001]；无量化层输出 E floor ≈8-9e-8（= fp16 底噪）；整网 logits cos 0.99999+；无量化 PPL Δ ≤2.2e-4。
- §8 陷阱: ring_r1 显式 False；未复用任何旧 R1/R2；任意配额维度 QR 构造；每 run json 带 transform_manifest（两侧共同变换 vs 实验改变逐项列出）。变换作用于全部 eligible 投影的 H/L 两组。

## 2. 四类指标分别回答了什么（seed0 配对；层等权 mean±std 见 runs.csv/图）

**5.1 幅度分布**: 旋转使 H 组的幅度尾部收敛（代表层 3B layer14 q_proj H: max 29→11.8, p99 7.6→5.8, block max/RMS p99 3.98→3.31），分布主体抬升（集中化）；L 组与 down_proj（已被在线 Hadamard 混合）几乎不变。组能量比 ≈1（正确性检查通过，不作为指标）。零 block: 0。

**5.2 激活量化误差 A**: 两组均基本不变 — A_H ≈ 7.0e-4、A_L ≈ 7.5e-3，Direct/Rotated 差 <1%（两模型、全部比例、全部 seeds）。解释性观察（非因果声明）: PCA 排序后组内相邻通道方差相近 + NVFP4/MXFP8 的 per-block MSE scale 搜索已吸收分布形状，组内再混合对激活侧 fake-quant 相对误差无可测影响。

**5.3 层输出误差 E**: Rotated 系统性略低于 Direct（r≥0.3；3B r=0.5: 2.50e-3 vs 2.71e-3；r=0.9: 0.97e-3 vs 1.20e-3；8B 同向）。由于 A 不变，收益来自权重侧（GPTQ 在旋转后权重上的误差反馈/冻结 scale 交互），非激活侧。

**5.4 PPL（wiki/C4 同向）**: 交叉行为 —
| wiki | r=0.1 | r=0.3 | r=0.5 | r=0.7 | r=0.9 |
|---|---|---|---|---|---|
| 3B Direct | 8.378 | 8.279 | 8.130 | 8.078 | 8.023 |
| 3B Rotated | 8.490 | 8.254 | 8.122 | 8.029 | **7.941** |
| 8B Direct | 6.620 | 6.563 | 6.420 | 6.355 | 6.300 |
| 8B Rotated | 6.709 | 6.559 | 6.442 | 6.341 | **6.259** |

低比例（r≤0.2）Rotated 更差（+0.09~+0.11 wiki），中段持平，高比例（r≥0.7）Rotated 更好（r=0.9: 3B −0.082, 8B −0.042）。rotation seeds 高度一致（3 seeds range ≤0.05 PPL @0.1, ≤0.02 @0.9）→ 差异非旋转随机性。

## 3. 趋势一致/不一致（如实并列，不强行因果）

- 一致: E 下降与高比例 PPL 改善同向；A 不变与"收益在权重侧"的解释一致；幅度集中化只在 H 组可见。
- 不一致: 低比例处 E 仍是 Rotated 略低（3B r=0.1: 4.79e-3 vs 5.07e-3）但 PPL 更差 — 局部误差下降不自动兑现为模型质量（任务书 §5.4 预期的非单调情形，照报）。
- 回答 §1 的问题: **固定分配与 mask 后，组内正交变换仍能改变量化质量，但方向依赖比例**；在本协议（PCA 表示 + AV-within mask + NVFP4/MXFP8 真格式 + 冻结 GPTQ）下它不是单调改进，不能宣称"旋转一定改善 PPL"。

## 4. 适用范围与未完成项

- 结论范围: 每层 H/L 两逻辑组、PCA(ResQ 式未中心化 basis)表示、AV-within mask、NVFP4(blk16)/MXFP8(blk32) + 两级 scale + GPTQ 冻结配方；Rotated 为显式参考实现（激活侧运行时 fp16 matmul，非零开销，不声称可零成本部署）。
- 未做（任务书边界）: 原始表示对照、rotation-only、across、训练旋转、ring/group allocator、r=0/1 两组混合标注（单格式对照在 M31）。
- 分位数来自固定 reservoir 采样（≤200k/组, seed 320001）+ 精确 max；manifest 记录。
- Qwen3-8B 扩展未启动（可选项，不阻塞）。
