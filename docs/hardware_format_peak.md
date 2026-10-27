# 硬件格式峰值吞吐 — RTX Pro 6000 Blackwell (SM120, cc 12.0, GB202)

> 2026-10-27。目的:**先排除硬件/格式混淆**。在我们实际的卡上,逐数值格式实测张量核 dense 峰值,
> 回答用户的核心质疑:"我们用 NVFP4、peers 用 INT4,这两个在 Pro 6000 上性能一样么?"
> 实测(dense,M=N=K=8192/16384,cudaEvent median,统一按 2·M·N·K 计 ops/sec)。
> 远程原始数据:`/home/luofan/plaquant/docs/hardware_format_peak.{md,csv}` + commit `adba0e7`(卡在远端未推);本文为本地权威副本。

## 实测峰值表

| 格式 | kernel | @8192 | @16384 | vs FP16 | vs INT8 |
|---|---|---|---|---|---|
| FP16/BF16 | cuBLAS | 406–417 TFLOPS | 405–414 | 1.0× | 0.5× |
| FP8 E4M3 | cuBLASLt | 807 TFLOPS | 791 | ~2.0× | ~0.99× |
| INT8 | CUTLASS IMMA m16n8k32 (tuned) | 814 TOPS | 844 | ~2.0× | 1.0× |
| **INT4 (s4.s4)** | **= INT8 同一指令** `IMMA.16832.S8.S8`(SASS 零 S4 opcode) | ceiling ≤ 814 | ≤ 844 | ≤2.0× | **1.0×** |
| **NVFP4 (block-scaled)** | CUTLASS SM120 ex79b (tuned) | **1469 TFLOPS** | **1260** | ~3.6× | **1.49–1.80×** |

> INT4 行实测 shim 得 91 TOPS,但那不是硬件天花板——s4.s4 MMA 在 SM120 被 ptxas **降级成与 INT8 完全相同的 `IMMA.16832.S8.S8`**(SASS 证据:零 S4 opcode)。所以 INT4 的真实上限 = INT8 ≈ 814 TOPS。即 **在 Blackwell 上 INT4 相对 INT8 零加速**。

## 结论(回答"FP4 vs INT4 性能一样么")

**不一样,但也不是我最初臆断的干净 2×。真实是 1.49–1.80×,且这个优势是 NVFP4 自己的 block-scaled 数据通路专属的。**

1. **精度阶梯不是干净的 1:2:4。** 实测 FP16 : FP8/INT8 : FP4 ≈ **1 : 2 : 3.6**。FP4 相对 FP8 只有 ~1.8×(理论干净应是 2×),说明 GB202(消费/工作站级)确实把 dense FP4 速率**部分 gate** 了——不像 B200。用户"也许差不多"的怀疑方向对,但 gate 得没那么狠,FP4 仍有实打实的 ~1.5–1.8× 领先。

2. **关键:这是 NVFP4 专属,不是"4-bit 通吃"。** 纯 INT4(s4)在 SM120 **编译成和 INT8 一模一样的指令**,相对 INT8 **零加速**。所以 INT4-based 的 W4A4 peers(QuaRot / Atom / ResQ / QServe-W4)在 Blackwell 上最快只能跑到 INT8 速率;NVFP4 则有 ~1.5–1.8× 于此。

3. **诚实的 paper 口径(三层拆分,堵死"你只是赢在格式"):**
   - **格式-硬件红利 ≈ 1.5–1.8×(不算我们算法功劳)**:NVFP4 有专属通路,INT4 没有(= INT8)。任何还用 INT4 的 W4A4 方法在这代硬件上先天拿不到。
   - **精度轴**:同格式(都 NVFP4)下,ProMix 分区 + act-MSE + 两级 scale 的精度优势(待同格式复测 peers 验证)。
   - **性能轴**:同格式下,我们融合 kernel 相对朴素 FP4 kernel 的额外收益(whole-K cross-phase overlap + SFASYNC)。

## 口径注意

- FP8 ≈ INT8(807 vs 814),符合预期(同 m16n8k32 档)。
- spec-sheet dense TOPS 本轮因无网络未复核;NVIDIA 公布的 headline "AI TOPS" 多含 2:4 sparsity,本表全部为 **dense 实测**,勿与之混。
- achieved/theoretical efficiency 待补(需复核官方 dense 规格)。
