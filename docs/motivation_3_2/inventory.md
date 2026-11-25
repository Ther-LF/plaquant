# M32 Inventory — Motivation §3.2 固定分配下的组内正交变换

日期: 2026-11-25 ｜ 任务书: `docs/paper_motivation_3_2_experiments.md` ｜ 前置: M31（docs/motivation_3_1/，已完成 198/198）

## 环境 / 代码
- 同 M31（inventory.md §1-2）: worktree /home/luofan/plaquant-metricsweep, venv torch 2.13/TF5.17, GPU4-7 共卡。
- Runner: `promix/scripts/rd_m32.py`（run32/stats32/equiv32/verify32/aggregate32），launcher `promix/scripts/m32sweep.sh`。

## 复用（冻结自 M31）
- 表示: PCA（fuse_basis identity_main_rotation=True, **ring_r1=False 显式**；basis = 未中心化 X^T X、1% damping、eigh 升序，hash 见 M31 manifest）。
- 分数/mask: AV-within（scores/3b_pca.pt, 8b_pca.pt + 确定性 top-h + 升序打包）。Direct 与 Rotated **共用同一 mask**；`verify32` 对两模型 9 比例逐项核验 M31 run 的 alloc_hash/representation/recipe 全等（direct_reuse_verification_{3b,8b}.json: all_match=True）。
- PTQ 配方 / fit / held-out / PPL 协议: 与 M31 manifest 完全一致。
- Direct 一侧 E/PPL = M31 `{mk}_av_within_s0_r*` 运行（核验后复用）；Direct 的 A/幅度统计由 `stats32` 在浮点 PCA 孪生上新测（激活统计不需要重新量化）。

## Rotated 实现（算法级参考实现，非零开销，已披露）
- 每 eligible Linear 的打包布局 [L|H] 两组各乘确定性随机正交阵: float64 Gaussian→QR→对角符号校正；seed = md5(m32:{model}:{module}:r{ratio}:{group}:rot{rotation_seed})。**不复用任何旧 R1/R2 矩阵**，任意合法配额维度直接 QR 构造（无截断 Hadamard）。
- 权重侧 V_g R_g: float64 在 GPU 上乘入权重列（打包后分块），存回 fp16。激活侧 Z_g R_g: 运行时 fp16 matmul（共同计算 dtype），由 ActQuantWrapper.forward 的进程内 patch 在 reorder 之后、量化之前施加（含 down_proj 在线 Hadamard 之后）。
- GPTQ 在旋转后权重/激活上按同一冻结配方重跑（候选权重分别生成，不声称逐元素相同）。

## Transform manifest（每 run json 内 `transform_manifest`）
两侧共同: norm fusion、残差流 U（R1 关闭）、per-head R2（仅 v-out/o-in；o_proj 不在 eligible、FP16）、down_proj 在线 Hadamard、共享 mask 列置换。实验改变: 仅组内 blockdiag(R_L,R_H)（Rotated 侧），作用于全部 eligible 投影（q/k/v/gate/up/down）。

## 等价性 gate（equivalence.json）
- 3B r=0.5 rot0: ortho 残差 7.3e-15；组能量比 [0.99999, 1.00001]；无量化层输出 E floor 9.3e-8（=fp16 底噪）；logits cos 0.999992；无量化 PPL Δ=+9.1e-5。
- 8B r=0.5 rot0: ortho 9.9e-15；能量 [0.99999, 1.00001]；floor 8.3e-8；logits cos 0.999994；PPL Δ=-2.2e-4。

## 新统计器
- §5.1 幅度: 每投影×组 |z| 分位数(p50/90/99/99.9/max, 精确 max + 固定 seed=320001 的 ≤200k/组 采样)、scale-block max/RMS 分布（NVFP4 blk16 / MXFP8 blk32，零 block 单计）、组 ENERGY（能量比在聚合侧 Direct/Rotated 对比）。代表层预声明: layers {2, L/2, L-3} × {q_proj, down_proj}（存完整 101 点分位曲线）。
- §5.2 A = SSE/(ENERGY+1e-12)，float64 累计，teacher 表示 Z'（held-out 32 窗），H/L 分开，层等权 mean±std(ddof=0)。

## 缺口 / 注意
- 8B Rotated 单点 ~1.5-2.2h（fp64 权重旋转 + 共卡），矩阵在 lane13/14 上滚动。
- stats32 的分位数来自固定 reservoir 采样（非全量），max 为精确值；manifest 记录样本数与 seed。
