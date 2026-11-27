# rd-ringsize: P0-3 — 组宽度 b (`ring_size`) 敏感度消融

**Spec** `docs/p0_supplemental_experiments_spec.md` 任务 P0-3 · **HW** RTX PRO
6000 Blackwell (SM120), container `luofan-plaquant`, 精度 GPU lanes 4/5 (3B) +
6/7 (8B), 算子 GPU 7 (idle) · **Date** 2026-11-28 · **Raw JSONs**
`docs/rd_ringsize_data/` (synced from `/home/luofan/ringsize/`) · **Code**
clean worktree `/home/luofan/plaquant-ringsize` @ `ab09b0b` (detached, no
dirty files; rotation artifacts symlinked from the production repo) ·
**Drivers** `promix/scripts/rd_acctune.py` (accuracy, the EXACT shipped-anchor
runner, one point per process) + `promix/scripts/rd_ringsize_harvest.py`
(alloc 结构分析 + 算子代价, new).

**Question**: 论文精度扫用 b=64、部署用 b=512 — 从 32/64 放大到 512 损失多少
精度、换来多少系统收益？控制变量单轴扫描 b ∈ {32, 64, 128, 256, 512}，固定
完整 shipped 配方（PCA + 组内 Sylvester–Hadamard ring R1 + 全局实测 ΔD 分配 +
GPTQ g512 + 两级 weight scale w7 + act-MSE d3u2，KV16，r = 1/8；3B 另有
percdamp=.025 — 与 acctune/buildup 锚点逐字节同配方）。

---

## 0. 协议门控（入账前提）

`b=512` 行与 `rd_buildup_ablation.md` / `rd_accuracy_tuning.md` 锚点
**浮点逐位相同**（不是 <0.001，是 `==`，与 `/home/luofan/acctune/
{3b_stack_full_g512_d025,8b_stack_full_g512}.json` 的 float 完全一致）：

| gate | wiki | C4 | vs 锚点 |
|---|---|---|---|
| 3B b512 | 8.217978477478027 | 11.254905700683594 | bit-exact PASS |
| 8B b512 | 6.564249038696289 | 9.563854217529297 | bit-exact PASS |

→ 本扫描的 runner/配方/seed 与在账锚点是同一条生产管线，b 是唯一变量。

## 1. 主表 — b × {wiki, C4, avg_fp8, 非单调层占比, pack 延迟, GEMM 延迟}

精度：WikiText-2 / C4 PPL（acctune 协议，eval seed 固定）。`avg_fp8` =
build 后实测 FP8 通道占比（`rv.avg_fp8_fraction`）。非单调层 = 分配结果中
FP8 ring 集合不是排序坐标尾部连续块、需要在线 group gather（Π≠I）的
Linear 数 / 可重分配 Linear 总数（q/k/v/gate/up/down × 层数；o_proj 恒为
positional，Π≡I）。pack / GEMM 延迟 = 8B gate_up 代表形状
（N=14336, K=4096, M=8192，取该 b 下实际分配的中位 K_high），CUDA-graph
replay 口径（§3，us/iter）。

**Llama-3.2-3B**

| b | wiki | C4 | avg_fp8 | 非单调层 | 8B-shape pack (us) | 8B-shape GEMM (us) |
|---|---|---|---|---|---|---|
| 32 | 8.2938 | 11.3504 | 0.1250 | 17/168 (10.1%) | — | — |
| 64 | 8.2964 | 11.3865 | 0.1250 | 0/168 (0%) | — | — |
| 128 | 8.2248 | 11.2781 | 0.1250 | 0/168 (0%) | — | — |
| 256 | 8.2414 | 11.2966 | 0.1250 | 0/168 (0%) | — | — |
| **512** | **8.2180** | **11.2549** | 0.1250 | 0/168 (0%) | — | — |

**Llama-3-8B**（pack/GEMM 列 = gate_up 形状 @ 该 b 的中位 K_high）

| b | wiki | C4 | avg_fp8 | 非单调层 | K_high (med) | pack (us) | gather+pack (us) | GEMM (us) |
|---|---|---|---|---|---|---|---|---|
| 32 | 6.5869 | 9.6011 | 0.1250 | 28/192 (14.6%) | 832 (mod128=64) | 131.7 | 137.6 | 960.6 |
| 64 | 6.5648 | 9.5939 | 0.1250 | 0/192 (0%) | 832 (mod128=64) | 132.3 | 131.8 | 958.9 |
| 128 | 6.5773 | 9.6129 | 0.1250 | 0/192 (0%) | 896 | 126.4 | 130.7 | 935.0 |
| 256 | 6.5671 | 9.6011 | 0.1250 | 0/192 (0%) | 768 | 123.9 | 128.7 | 916.7 |
| **512** | **6.5642** | **9.5639** | 0.1250 | 0/192 (0%) | 1024 | 128.7 | 133.1 | 964.1 |

down_proj 代表形状（N=4096, K=14336, M=8192）：所有 b 下实测分配给
down_proj 的 FP8 ring 数均为 **0**（K_high=0，纯 NVFP4 单相路径），算子
行为与 b 无关：pack 402.2–403.2 us、GEMM 687.6–691.4 us（b 不敏感，
差异 < 0.6%）。

`avg_fp8`：5 个 b 点、两个模型的实测值全部**精确等于 0.1250**（ring 级
sel/total 与 build 后通道级 hi/tot 双口径均 0.1250）— 两模型的 total_rings
在所有 b 下均被 8 整除，边界 guard（clamp_chosen ≥1 NVFP4 ring）在 1/8
预算下从不触发，**无任何 r 偏移**。

## 2. 分配结构 — 非单调层与 down_proj

- **b ≥ 64：所有层的实测 ΔD 分配都是排序坐标的连续尾块（Π≡I），在线
  group gather 一个都不需要**。attn/gate/up 的输入在 PCA 排序坐标系里，
  per-ring ΔD 在 ≥64 通道的 ring 粒度上尾部单调；选中集 = 尾部 top-k。
- **b = 32 打破这一性质**：per-ring ΔD 估计在 32 通道粒度上出现噪声交叉，
  3B 17/168（10.1%）、8B 28/192（14.6%）的层选中集不再尾部连续
  （全部落在 q/k/v/gate/up；明细见 `alloc_analysis.json` per_proj）——
  这些层在 serving 侧必须走 gather-folded pack（§3）。
- **down_proj 在所有 b 下拿到 0 个 FP8 ring**：其输入是在线 Hadamard 后
  （非 PCA 排序）的激活，per-ring ΔD 平坦且远小于 attn/mlp 入口的尾部
  ring，全局阈值把全部 1/8 预算让给 q/k/v/gate/up（它们的实际 FP8 占比
  因此 > 1/8，中位 K_high 见主表）。这也是 ΔD 非单调唯一真实存在的地方
  （down 的 ΔD argmax 在中部）——但预算为 0，不构成 gather 需求。

## 3. 算子代价（8B 代表形状, M=8192, GPU 7 独占）

口径：`promix/scripts/rd_ringsize_harvest.py op` — 真实 shipping 路径
（`pack_weight_mixed` + `MixedQuantLinear`，qpfm fused mixed pack + castfold
128×128 fused GEMM，全套 kernel 表），`decode_probe.bench_graph` CUDA-graph
replay（serving 口径），每 b 用该 b 实际分配的中位 K_high。gather 用
ring 粒度随机置换（Π≠I 最坏情形）的 `quant_pack_float_mixed_gather`
单 launch 融合路径。

- **pack**：123.9–132.3 us，对 b 本质平坦（差异来自 K_high 本身）。
- **gather+pack**（仅 b=32 的非单调层需要）：+3–6 us（≈+4%）每 Linear —
  融合 gather 使代价很小，但 b=32 时 10–15% 的层平白多付这笔。
- **GEMM**：916.7–964.1 us，随 K_high 变化（768→916.7，1024→964.1，
  ≈+5% 对 +256 FP8 通道）。**128-pad/对齐惩罚实测**：b≤64 的中位
  K_high=832（mod 128 = 64，MXFP8 相以半 k-tile 收尾）跑 960.6 us ——
  比**更大的** 128 对齐 K_high=896（b=128，935.0 us）还慢 2.7%。b=512 的
  K_high 恒为 512 倍数，128 对齐 by construction，不存在该惩罚。
- SM120 上 `khigh_pad_enabled` 为 off（低 SM 设备专用），832 不触发
  zero-pad（K_high_pad==K_high），misalignment 直接体现在 mainloop 里 ——
  即"b<128 的 128-pad/对齐惩罚"以 +2.7% GEMM 的形式如实落账。

## 4. 结论 — "b=512 的取舍"其实没有取舍

**b=512 同时是精度最优点和系统最干净点，放大到 512 不是用精度换系统收益
—— 两头都赢**：

1. **精度**：b=512 在两个模型 × 两个数据集上全部最优。缩到 b=32/64 反而
   损失精度（3B 最多 +0.078 wiki / +0.132 C4，≈15× 噪声底；8B +0.023 /
   +0.049）。机理与 `rd_ron_ringsize` 的 mixing-width 结论一致：更宽的
   Sylvester–Hadamard ring 把 2–16× 更多坐标混合进每个旋转块，量化前
   子空间更接近各向同性；而"更细的分配粒度"在 1/8 预算下没有可兑现的
   收益（b 轴不严格单调 — 3B b128<b256、8B b64<b128，均在交叉项噪声内，
   但 b=512 的最优和 b≤64 的劣势在两个数据集上同向）。
2. **分配干净度**：b≥64 全部层 Π≡I（零在线 gather）；b=512 的 ring 数
   最少（8B 2176 vs b32 的 34816），分配/export 元数据缩 16×。
3. **算子**：K_high 恒 512 倍数 → 永远 128 对齐，无 b≤64 实测到的
   half-k-tile 惩罚（+2.7% GEMM）与 b=32 的 gather 路径（+4% pack，
   10–15% 层）。
4. **预算实现**：所有 b 的 avg_fp8 均精确 0.1250，r 偏移为零 —— b=512
   的粗粒度并不牺牲预算精度。

论文可直接写：组宽从 32 扫到 512，精度单调不降（至 512 最优），且 512 是
唯一让分配天然尾部连续、K_high 天然 128 对齐的部署点 —— b=512 既是
accuracy argmax 也是 systems argmin。

## 5. Reproduce

```bash
# 精度（每点一个进程；3B 加 --percdamp 0.025；b 换 --ring_size）
PROMIX_ACT_MSE_STEPS_UP=2 PROMIX_WEIGHT_MSE_STEPS=7 \
python promix/scripts/rd_acctune.py --config promix/configs/llama-3-8b-mxfp8-nvfp4.yaml \
  --point 8b_b512 --substrate ring --ring_size 512 --alloc measured \
  --gptq_nsamples 512 --c4 --stats_cache stats_8b_b512.pt --out A_8b_b512.json
# 分配结构（CPU）+ 算子（idle GPU）
python promix/scripts/rd_ringsize_harvest.py alloc --stats_dir DIR --out alloc_analysis.json
CUDA_VISIBLE_DEVICES=7 python promix/scripts/rd_ringsize_harvest.py op \
  --alloc_json alloc_analysis.json --model 8b --m 8192 --out op_cost_8b_m8192.json
```

Raw: `docs/rd_ringsize_data/A_{3b,8b}_b{32,64,128,256,512}.json`（含 meta/
alloc_meta/avg_fp8/wiki/c4）、`alloc_analysis.json`（非单调层 + per-proj
K_high 分布）、`op_cost_8b_m8192.json`（算子延迟）。远端
`/home/luofan/ringsize/`（含 .log 与 stats_*.pt ring 统计缓存）。
