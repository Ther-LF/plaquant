# Motivation §3.3 补充微基准任务书（M33-ext）：Naive 2-Launch 混合精度 GEMM 随比例 $r$ 的延迟扫描

- **日期：** 2026-11-27
- **对应论文：** AdaMix（仓库 `promix/`），Section 3.3 Motivation（Fig. 5）
- **前置参考：** `docs/rd_buildup_data/h2h_naive.json`（既有仅测 `r = 1/8 = 0.125` 单点）与 `docs/rd_perfpuzzle_ratio.csv`
- **预计耗时：** **< 1 分钟**（纯单层 GEMM CUDA Event 计时，不加载模型权重、不跑 GPTQ/PPL）
- **隔离原则：** 所有产物写入 `docs/motivation_3_3_ext/`，严禁覆盖 `docs/rd_buildup_data/`。

---

## 1. 实验目标（为什么补做这 30 秒微基准）

既有 `h2h_naive.json` 仅在固定 `r = 0.125 (1/8)` 测了 naive 2-launch（一次 NVFP4 GEMM + 一次 MXFP8 GEMM + 累加）的延迟，审稿人容易质疑单点代表性。

本任务沿高精比例 $r \in [0, 1]$ 完整扫描 **Naive 2-Launch 混合精度 GEMM** 延迟，并与 `Uniform NVFP4 (r=0)`、`Pure MXFP8 (r=1)`、`BF16` 及理想线性插值 $t_{\mathrm{ideal}}(r) = (1-r)\,t_{\mathrm{fp4}} + r\,t_{\mathrm{fp8}}$ 对比，直接检验一个关键系统现象：
- **一旦 $r > 0$（哪怕极小比例如 $r=1/16$ 或 $1/8$），naive 2-launch 是否因双 launch + HBM 中间部分和读写 + 窄 $K$ 分块导致延迟瞬间跃升，在几乎整个中间区间 $r \in (0, 1)$ 拱在理想线性插值乃至纯 MXFP8（$r=1$）之上？**
- 完成后，Section 3 Motivation 的全部三张实测图（Fig. 3 / Fig. 4 / Fig. 5）横轴全部统一为 `FP8 fraction r ∈ [0, 1]`。

### 严格反循环边界（Anti-Circularity）
- Section 3.3 的 Fig. 5 **严禁混入 AdaMix 自己的 fused kernel（`fused_ms`）**；本任务仅测标准/naive 路径（`bf16`、`pure_nvfp4`、`pure_mxfp8`、`naive_2launch` 及其拆项）。

---

## 2. 形状、场景与比例网格

### 2.1 模型与矩阵形状（复用 `h2h_naive.json` 口径）
- **主模型（P0 必跑）：Llama-3-8B**
  - `qkv`：`N=6144, K=4096`（合并 QKV）
  - `o_proj`：`N=4096, K=4096`
  - `gate_up`：`N=28672, K=4096`（合并 Gate/Up）
  - `down`：`N=4096, K=14336`
- **对照模型（P1 顺手跑，多花 ~15 秒）：Llama-3.2-3B**
  - `qkv`：`N=5120, K=3072`；`o_proj`：`N=3072, K=3072`；`gate_up`：`N=16384, K=3072`；`down`：`N=3072, K=8192`

### 2.2 推理场景（两种典型 $M$）
- **Prefill（计算受限）：** `M = 8192`
- **Decode（launch / 访存受限）：** `M = 128`

### 2.3 比例网格 $r \in [0, 1]$（10 点，确保 $K_{\mathrm{high}} \bmod 32 = 0$ 且 $K_{\mathrm{low}} \bmod 16 = 0$）
$$r_{\mathrm{target}} \in \{0.0,\ 0.0625,\ 0.125,\ 0.25,\ 0.375,\ 0.50,\ 0.625,\ 0.75,\ 0.875,\ 1.0\}$$

- 计算规则：`K_high = round(r * K / 32) * 32`（$r=0$ 取 `0`，$r=1$ 取 `K`），`K_low = K - K_high`，断言 `K_high % 32 == 0` 且 `K_low % 16 == 0`，记录真实比例 `r_actual = K_high / K`。
- 在 8B（`K=4096`）上上述 10 点全部精确整除（`K_high = 0, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096`）。

---

## 3. 被测路径与计时协议

复用 `h2h_naive.json` 相同的 SM120（RTX Pro 6000 Blackwell）硬件与 `torch._scaled_mm` / CUTLASS 标准路径（与既有 `torchfp4_ms`、`torchmixed2_ms`、`bf16_ms` 完全同口径）：

1. **`bf16_ms`**：`torch.mm` / cuBLAS BF16 参考（与 $r$ 无关，每个 `(model, shape, M)` 测一次即可）。
2. **`torchfp4_ms`（$r=0$ 单 launch 纯 NVFP4）**：全 $K$ 走标准 block-16 NVFP4 GEMM。
3. **`torchfp8_ms`（$r=1$ 单 launch 纯 MXFP8）**：全 $K$ 走标准 block-32 MXFP8 GEMM。
4. **`torchmixed2_ms`（Naive 2-launch 混合 GEMM，用于所有 $r \in (0, 1)$）**：
   - Launch 1：`Y_lo = GEMM_nvfp4(A_lo, B_lo)`（维度 $M \times N \times K_{\mathrm{low}}$）
   - Launch 2：`Y_hi = GEMM_mxfp8(A_hi, B_hi)`（维度 $M \times N \times K_{\mathrm{high}}$）
   - Merge：`Y = Y_lo + Y_hi`（或 `beta=1` epilogue/elementwise add，严格保持与 `h2h_naive.json` 中 `torchmixed2_ms` 同一实现，并在 `manifest.json` 写明是显式 `add` 还是 `beta=1`）
   - *注*：在端点 $r=0$ 记 `torchmixed2_ms = torchfp4_ms`；在端点 $r=1$ 记 `torchmixed2_ms = torchfp8_ms`（便于画完整连续的 $r \in [0, 1]$ 曲线）。
5. **可选开销拆项（强烈建议顺手记一下，多花 5 秒）**：分别记录 `part_fp4_ms`、`part_fp8_ms`、`part_add_ms`，便于正文精准解释“两段 kernel 自身 + 中间 HBM 部分和写回/相加”各自占了多少。

### 3.2 运行时变换、重排与独立量化打包开销（必测，多花 ~8 秒，与 §4.3 消除的开销严格 1-to-1 对应！）
为了确保 Section 4.3（AdaMix 两级融合）消除的**每一项运行时开销**在 Section 3.3 都有白盒实测支撑，对每个 `(model, shape, M)` 额外测量以下 4 项标准算子延迟：
1. **`online_dense_rot_ms`（跨层切分耦合导致的在线残差流基变换 $X \cdot (T_{\ell-1}^\top T_\ell)$，与 $r$ 无关）**：
   - 在 `qkv`、`o_proj`、`gate_up` 上测一次标准 BF16 `M × K @ K × K` 矩阵乘（例如 8B 上 `M × 4096 @ 4096 × 4096`），直接量化“如果像 ResQ 一样旋转块随每层 $d_{H,\ell}$ 改变而破坏共享离线吸收，每层被迫在运行时做一次稠密正交基转换要付多少 ms”（`down` 维 $K=14336$ 跳过稠密乘，仅测下面 Hadamard）。
   - *对应 §4.3 消除项*：模型级共享分组旋转 $T = U\operatorname{blkdiag}(R_1,\dots,R_K)$ 离线吸收（Proposition 2 → 0 ms）。
2. **`online_gather_ms`（离散逐通道重排/gather 开销，与 $r$ 无关）**：
   - 测一次 `X.index_select(1, perm)`（将离散高精度与低精度通道重排为连续内存块），直接量化 Fig. 1(a) 逐通道混合精度的内存搬运税。
   - *对应 §4.3 消除项*：融合激活准备（Fused Activation Prep，单趟完成重排与打包，零中间 HBM 往返）。
3. **`standalone_quant_pack_ms`（独立两段在线激活量化+打包开销，随 $r$ 变化）**：
   - 对每个 $r \in (0, 1)$，分别独立执行一次 `NVFP4 (M × K_low)` 量化打包与一次 `MXFP8 (M × K_high)` 量化打包（两趟独立 HBM 读/写，不融合 gather/Hadamard），记录总延迟（$r=0$ 仅 NVFP4，$r=1$ 仅 MXFP8）。
   - *对应 §4.3 消除项*：融合激活准备单趟双格式量化打包。
4. **`online_hadamard_ms`（仅 `down_proj`，独立在线 Walsh-Hadamard 变换，与 $r$ 无关）**：
   - 复用仓库现有 `matmul_hadU_cuda` 测 `down_proj`（`K=14336` / `8192`）一次独立在线 Hadamard 延迟（既作与稠密旋转的轻量对照，也量化未与打包融合时的独立变换税）。
   - *对应 §4.3 消除项*：FFN `down_proj` 在线 Hadamard 与激活量化打包融合（Step 4 / Fig. 8a）。

> 这样组合后，Section 3.3 实测的四类系统税与 Section 4.3 消除的四类开销严格 **1-to-1 闭环**：既能画出 `Naive 2-launch` 随 $r \in [0, 1]$ 的延迟曲线，也能在同一张图上叠出 **`+ Online prep (gather + quant/pack)`** 与 **`+ Online dense rotation`（跨层变 $d_{H,\ell}$ 无法离线吸收时的真实总开销）**，一针见血打通 §3.2B、§3.3 与 §4！

### 计时标准
- 空闲 GPU（固定时钟或记录实际 GPU 型号/温度/驱动/PyTorch & CUTLASS 版本）；
- 每个配置先 `warmup = 20` 次，再用 CUDA Event 测 `iters = 50` 次（含 L2 cache flush 或明确声明与 `h2h_naive.json` 一致的缓存口径），记录 **median（`*_ms`）** 与 **min（`*_min_ms`）**。

---

## 4. 验收 Gate 与交付物

1. **一致性校验 Gate**：核对 `r = 0.125` 点上的 `bf16_ms`、`torchfp4_ms`、`torchmixed2_ms` 与既有 `docs/rd_buildup_data/h2h_naive.json` 基本一致（偏差 $\le 5\%$ 属正常温度/频率波动；若超 10% 排查实现差异）。
2. **输出文件**（全部存入 `docs/motivation_3_3_ext/`）：
   - `naive_rsweep.json`（及同名 `.csv`）：包含字段 `model, shape, M, N, K, K_low, K_high, r_target, r_actual, bf16_ms, torchfp4_ms, torchfp8_ms, torchmixed2_ms, torchmixed2_min_ms, part_fp4_ms, part_fp8_ms, part_add_ms, online_dense_rot_ms, online_gather_ms, standalone_quant_pack_ms, online_hadamard_ms, ideal_linear_ms, slowdown_vs_fp4, slowdown_vs_ideal, slowdown_vs_fp8`；
   - `manifest.json`：GPU 型号、PyTorch/CUDA 版本、commit SHA、`torchmixed2` 具体实现方式、warmup/iters 与 L2 flush 口径。

---

## 5. 数据回来后，论文 Fig. 5 的直接落图方式

待 `docs/motivation_3_3_ext/naive_rsweep.json` 返回后，**Fig. 5** 将直接升级为与 Fig. 3 / Fig. 4 完全统一的 **`r ∈ [0, 1]` 曲线双联图**（Llama-3-8B）：
- **(a) Prefill (`M=8192`) latency vs. $r$**：同图对比 `Ideal linear (1−r)t₄ + r t₈`、`Naive 2-launch`（全程拱起）以及 **`Naive 2-launch + Online rotation`**（当旋转与切分耦合、跨层变 $d_{H,\ell}$ 无法离线吸收时的真实总延迟），同时标出 `Pure MXFP8 (r=1)` 参考线；
- **(b) Decode (`M=128`) latency vs. $r$**：展示 `Naive 2-launch` 与 `+ Online rotation` 全程高出 `Uniform NVFP4` 甚至远超 `BF16` 参考线，一图彻底立住**“共享离线吸收（消除在线旋转）+ 单 launch 融合 GEMM（消除双 launch 税）”**的双重系统必要性！
