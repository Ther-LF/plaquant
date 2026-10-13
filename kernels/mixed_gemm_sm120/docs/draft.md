# SM120 NVFP4+MXFP8 融合 GEMM — KDA Draft

> KDA 工作流的 draft 阶段产物。**实现代码在本 draft 转成 executable plan 之前不动手。**
> 参照:NVlabs/KDA agent-flow(契约 → draft → 单候选迭代 → 证据 → 择优)。

---

## 1. 任务契约(Task Contract)

- **目标**:为混合精度 Linear(q/k/v/gate/up/o/down_proj)实现**单次 kernel launch** 的 NVFP4+MXFP8 融合 GEMM,在 SM120(RTX Pro 6000 Blackwell)上,精度对齐 PyTorch fake-quant oracle,延迟优于 2-launch FP 基线和 FP16 cuBLAS。
- **输入**:
  - `A_low` NVFP4 packed(M×K_low,每 2 个 FP4 一字节)+ `SF_A_low`(每 16 元素一个 E4M3 块 scale)
  - `A_high` MXFP8(M×K_high,E4M3)+ `SF_A_high`(每 32 元素一个 E8M0 块 scale)
  - `B_low`/`B_high` 权重(N×K,同格式,列主序 = ColumnMajor(K,N))+ 各自块 scale
  - `s_x`(per-token,M)、`s_w`(per-channel,N)外层 scale
- **输出**:FP16(M×N)。
- **正确性**:对同一组量化输入,输出与 PyTorch oracle 的 `cosine ≥ 0.9999`(bit-exact 为理想)。oracle = `promix/quantize/quant_utils.py` 的 `fake_quantize_nvfp4/mxfp8` 分段 + FP16 matmul 的等价计算。
- **性能目标**:在关键 shape 上,融合 kernel 延迟 < 2-launch 基线(两个单精度 GEMM + epilogue add)且 < FP16 cuBLAS。
- **允许手段**:CUTLASS 3.x CuTe,SM120(`-arch=sm_120a`),nvcc 13.0,PyTorch 扩展(setup.py build_ext),单 kernel launch,寄存器内共享累加器。
- **验证命令**:`python -m pytest tests/test_mixed_gemm_sm120.py`(待建:编译 + 对 oracle 的 cosine 检查)。
- **评估命令**:`python kernels/mixed_gemm_sm120/benchmark.py`(待建:逐 shape 测融合 vs 2-launch vs cuBLAS 延迟)。
- **promotion 标准**:候选同时满足 (a) 全部 shape cosine ≥ 0.9999,(b) 融合延迟 < 2-launch 基线。记录拒绝原因。

### 关键 shape(验证/评估用,high_fraction=1/8)

| 投影 | K_high (MXFP8) | K_low (NVFP4) | 来源 |
|------|---------------|---------------|------|
| q/k/v/gate/up (1B) | 256 | 1792 | hidden 2048 |
| o_proj (1B, 重排后) | 256 | 1792 | 全局 [低\|高] |
| down_proj (1B) | 1024 | 7168 | intermediate 8192(或纯 FP4,见风险) |
| 3B | 384 | 2688 | hidden 3072 |
| 8B | 512 | 3584 | hidden 4096 |

M = batch×seqlen(prefill 大 M;decode M=1)。N = 输出维。

---

## 2. 当前基线与验证路径

- **算法侧已验证(fake quant)**:1B/3B/8B 的 NVFP4+MXFP8 PPL 已跑通(14.92 / 8.65 / 6.83),见 `docs/superpowers/specs/2026-10-13-fp-vs-int-1b3b8b-results.md`。所以**数值正确性有 PyTorch 参照(oracle)**。
- **性能基线目前不存在**:还没有任何真低比特 GEMM;当前全是 fake quant(FP16 乘,零加速)。本任务要建两个基线:
  1. **2-launch FP**:CUTLASS 单精度 NVFP4 GEMM(K_low)+ 单精度 MXFP8 GEMM(K_high)+ epilogue add → 既是"要打败的基线",也是融合 kernel 的正确性锚点。
  2. **FP16 cuBLAS**:同 shape 的 FP16 GEMM 延迟上界参照。
- **验证如何建立**:用 pipeline 产出的真实量化权重/激活(或随机但符合格式的张量),喂 kernel,和 oracle 的分段 fake-quant 结果比 cosine。

---

## 3. 主要风险与未知

| 风险 | 说明 | 缓解 |
|------|------|------|
| **寄存器内共享累加器跨两种 atom** | NVFP4 atom = `SM120_16x8x64_TN_VS`(k64),MXFP8 = `SM120_16x8x32_TN_VS`(k32);需二者 FP32[4] 累加器 CLayout 相同(idea 文档称都是 `SM80_16x8_Row`,**需在 `mma_traits_sm120.hpp` 实证**) | Candidate 0 先各自单精度跑通并打印 CLayout 确认 |
| **CollectiveBuilder 不支持双 phase** | CUTLASS 3.x 的 CollectiveBuilder 是单一 mainloop,可能无法原生表达"先 FP4 后 FP8 进同一累加器" | 退路:手写 CuTe mainloop(不走 Builder);或 EVT-workspace-add 兜底(idea 文档 v5 方案) |
| **FP4 的 `<<2` 位移** | SM120 `ld.matrix b4x16_p64` 后 FP4 需左移 2 位放到 MMA 期望位置,漏掉就错 | 从示例 79 抄取确切处理 |
| **SMEM 99KB 预算** | 两 phase 的 A/B + 块 scale buffer;SM120 只有 99KB(非 SM100 的 228KB) | union 复用(Phase2 覆写 Phase1);限制 tile/stage |
| **块 scale 寄存器压力** | SM80 kernel 已用 90-96 寄存器,block scale operand 再加压 | 监控 ncu occupancy |
| **down_proj 该不该分段** | idea 文档说 down_proj 用 Hadamard + 纯 FP4(不分段);但 Stage 1b 代码分了段。kernel 侧可能只需 NVFP4-only 单 phase | 先按分段做(通用),后续按 PPL 决定是否简化 |
| **GPU 共享争抢** | 别的用户随时占满 GPU(本 session 已多次遇到) | 跑前挑空闲卡 + `expandable_segments:True` |

---

## 4. 候选方向(按价值/风险排序)

- **Candidate 0 — 单精度跑通 + CLayout 实证(最高优先,解锁一切)**
  编译运行 CUTLASS 示例 79(SM120 NVFP4 GEMM)和 87(SM120 FP8 blockwise)原样;从中抽出最小 NVFP4 GEMM 和 MXFP8 GEMM,各自对 oracle 验正确性;打印两个 atom 的累加器 CLayout 确认可共享。**产出:两个能跑的单精度 kernel + CLayout 结论。** 风险低、信息量最大。

- **Candidate 1 — 2-launch 基线**
  用 Candidate 0 的两个单精度 kernel + epilogue add 组装 2-launch 路径。既是正确性锚点(应 cosine≈1),又是要打败的性能基线。**产出:baseline latency 表。**

- **Candidate 2 — 融合单 launch(主交付物)**
  手写 CuTe mainloop:Phase1 NVFP4 主循环(K_low/64)→ SMEM union 复用 → Phase2 MXFP8 主循环(K_high/32),两 phase 累加进同一 FP32[4] 寄存器;epilogue `out = acc·s_x·s_w → FP16`。对 oracle 验正确 + 对 Candidate 1 验加速。**产出:融合 kernel。**

- **Candidate 3+ — 优化**(达成 promotion 后)
  tile 形状(64×64 / 64×128 / 128×64 / 128×128)、stage 数(3/4/5)、cross-phase prefetch(Phase2 加载与 Phase1 尾段计算重叠,idea 文档估 8-13%)、per-shape config 表。每步用 **ncu-report-skill** 取 profiling 证据。

---

## 5. 第一批具体实现步骤(Candidate 0)

1. 在远程容器建最小编译脚手架:`kernels/mixed_gemm_sm120/`,一个 setup.py(`-arch=sm_120a`,include CUTLASS)+ 一个 .cu。
2. 复制 CUTLASS 示例 79,原样编译运行,确认 SM120 NVFP4 GEMM 工具链通(nvcc 13.0 + sm_120a)。
3. 复制示例 87,确认 MXFP8 blockwise 通。
4. 读 `mma_traits_sm120.hpp` 里 `SM120_16x8x64_TN_VS` 和 `SM120_16x8x32_TN_VS` 的 CLayout,实证是否同为 `SM80_16x8_Row`(共享累加器的前提)。
5. 写 oracle 对照:PyTorch 端用 `fake_quantize_nvfp4/mxfp8` 造一组 (A,B,scale),kernel 算同样输入,比 cosine。
6. 记录到 `candidates.jsonl` + `benchmark.csv`。

---

## 6. 验证与评估命令(待建)

```bash
# 远程容器内,挑空闲 GPU + expandable_segments
# 编译
cd kernels/mixed_gemm_sm120 && rm -rf build *.so && python setup.py build_ext --inplace
# 验证(正确性 vs oracle)
python -m pytest tests/test_mixed_gemm_sm120.py -v
# 评估(延迟:融合 vs 2-launch vs cuBLAS,逐 shape)
python kernels/mixed_gemm_sm120/benchmark.py
```

---

## 7. 证据要求(promote / revise / reject)

- `candidates.jsonl`:每个候选的名字、父候选、状态(promoted/revised/rejected)、一句话理由。
- `benchmark.csv`:shape × 配置 × 延迟(融合/2-launch/cuBLAS)。
- `profile/`:ncu-report(用 ncu-report-skill 分析 occupancy / 寄存器 / SMEM / stall)。
- promote 条件:cosine≥0.9999 且 延迟 < 2-launch。reject 必须写原因(不静默丢弃)。

---

## 8. 参照材料

| 来源 | 作用 |
|------|------|
| `~/mixed_tensor/mixed_gemm_include/`(远程,私有库完整版)| SM80 双 phase 共享累加器**架构灵感**:`MixedMma`+`MixedMmaMultistage`(944行 multistage 是核心)。CUTLASS 2.x,**不可机械移植**到 SM120 |
| `kernels/mixed_gemm_l20/mixed_gemm_l20.cu`(本仓库)| 我们简化版 SM80 融合 kernel,SMEM union + 双 phase 模式参照 |
| CUTLASS `examples/79_blackwell_geforce_gemm`(远程)| **SM120 NVFP4 GEMM 代码底座** |
| CUTLASS `examples/87_blackwell_geforce_gemm_blockwise`(远程)| **SM120 FP8 blockwise GEMM 代码底座** |
| CUTLASS `include/cute/arch/mma_sm120.hpp`(远程)| atom:`SM120_16x8x64_TN_VS`(NVFP4)、`SM120_16x8x32_TN_VS`(MXFP8) |
| CUTLASS `include/cute/atom/mma_traits_sm120.hpp` | CLayout 实证 |
| `docs/superpowers/ideas/plaquant-sm120-fp8-nvfp4-fused-gemm.md` | 完整 SM120 设计 + 数据流图 |
| `promix/quantize/quant_utils.py:660-751` | oracle:`fake_quantize_nvfp4/mxfp8` |
| `docs/superpowers/specs/2026-10-13-fp-vs-int-1b3b8b-results.md` | 算法侧已验证的精度数字(正确性参照) |

---

## 9. 工作区布局(KDA)

```
kernels/mixed_gemm_sm120/
  docs/draft.md        ← 本文件
  docs/plan.md         ← 待:draft 转 executable plan
  setup.py / *.cu / *.cuh
  benchmark.py
  candidates.jsonl     ← 候选记录(git-ignored)
  benchmark.csv        ← 延迟记录(git-ignored)
  profile/             ← ncu 报告(git-ignored)
tests/test_mixed_gemm_sm120.py  ← 验证
```
