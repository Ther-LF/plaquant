# KDA 优化闭环 — SM120 融合 kernel

> 用 KernelWiki(找模式/PR 参考)+ ncu-report-skill(profile→诊断→计划)驱动真·单 kernel 融合到极致。每轮:改 → profile → 按证据定下一步。

## 硬目标
**3 个 CUDA kernel(Rung A 的 2 GEMM + combine)必须融成 1 个真·单 kernel**:两段 MMA 共享 FP32 寄存器累加器,中间结果不落显存。Rung A(单 host 调用、3 kernel、bf16 显存往返)不接受为最终形态。

## KernelWiki 提炼的优化方向(SM120 适用)
| 方向 | 来源 | 对应我们的弱项 |
|------|------|---------------|
| Small-Tile-N block-scaled GEMM(SM12x 专门)| flashinfer PR-3152, cutlass PR-3176 | 中等 N 的 LLM shape;小 M 利用率 |
| 小 M NVFP4 GEMM 优化 | sglang PR-10101 | decode(M=1/16)—Rung A 慢 3-7× 的地方 |
| 修 NVFP4 寄存器 spilling | sglang PR-8127 | 融合双 phase → 寄存器压力(C0 已见 SM80 用 90-96 寄存器) |
| MXFP8 block-scaled on Blackwell | tilelang PR-1945 | 高精度段 |
| 注意:B200 的 tcgen05/TMEM/2-SM 不适用 SM120(我们用 mma.sync,无 TMEM) | — | — |

## ncu profiling 流程(每个 kernel candidate)
按 ncu-report-skill:
1. 远程建独立 harness(standalone .cu + `-lineinfo`),一个 run 一个 `profile/<run>/` 目录。
2. 跑两档:`ncu --set full`(总览 + PmSampling)+ `ncu --set source --section SourceCounters`(逐行 stall)。
3. 用 `ncu_report` Python 模块解析(别肉眼看 CLI)。
4. 六维分析:occupancy / balance / stalls / tensor core / timeline / memory。
5. 对诊断手册:signal → 原因 → 修法,按预期收益排序。
6. 写 `profile/<run>/REPORT.md`,证据驱动下一轮改动。
   - 注意:ncu-report-skill 的 metric 名是 sm_100 的,SM120 多数通用,少数要查 `reference/08-b200-metric-names.md` 对照。

## 候选阶梯(真融合)
- B1:最简单单 kernel(朴素搬运,无流水线),两 phase 共享寄存器 acc,中间不落地 → **先正确**
- B2:cp.async 多级流水 + SMEM union + 跨 phase 预取
- B3+:ncu 驱动 —— Small-Tile-N 配置、寄存器压力(spilling)、小 M 路径、per-shape tile/stage config
- gate:每轮 cosine≥0.999 且 延迟单调下降;终极目标在 decode 小 M 和 prefill 大矩阵都 ≤ FP16 cuBLAS。

## 当前
- Rung B subagent 在建真·单 kernel(B0→B1→B2)。
- 它产出正确的单 kernel 后,立即进 ncu 闭环按上面方向逐轮压。
