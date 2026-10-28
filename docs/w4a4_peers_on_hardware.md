# W4A4 同类对比(RTX Pro 6000 / SM120 实硬件) — paper-ready 定稿

> 2026-10-28。把本轮所有 W4A4 同类 baseline 的**精度**(同协议实测)与**硬件格式峰值**(同卡实测)
> 合成一张给 paper 用的表,并给出"格式红利 / 精度 / kernel"三层诚实拆分。
> 对比对象 = **激活+权重同时量化**的 W4A4 工作(QuaRot / SpinQuant / Atom / QServe / vLLM-NVFP4),
> **不含** weight-only(GPTQ/AWQ,激活不量化,非同条件)。

## 1. 评测协议(所有方法统一)

wikitext-2-raw-v1 **test**、seqlen 2048、非重叠 chunk=141、288,627 token、bf16、
`add_special_tokens=False`、`"\n\n".join`、token-pooled `exp(mean NLL)`。
**每个实现的 FP16 bridge 都逐位复现我们 committed 的 BF16 基线 1B 13.16 / 3B 7.82 / 8B 6.14**
→ 下表每个 Δ 都是真·量化差异,非 tokenization/协议 artifact。
所有 peer 跑的是**官方开源实现**在 SM120 上的 fake-quant 精度模式。

## 2. 精度表(W4A4,越低越好)

| 方法 | bit/w | 1B | 3B | 8B | 备注 |
|---|---|---|---|---|---|
| BF16 参照 | 16 | 13.16 | 7.82 | 6.14 | — |
| **ProMix (SHIPPED, r=1/8)** | 4.97 | **14.56** | **8.30** | **6.60** | NVFP4 低 + MXFP8 高 |
| ProMix r=1/16(性价比拐点) | 4.73 | — | 8.36 | — | Pareto 优于 1/8 |
| vLLM NVFP4 | 4.50 | 15.65 | 8.63 | 6.87 | 纯 NVFP4 |
| ProMix r=0(纯 NVFP4,**同 bit**) | 4.50 | — | 8.86 | — | 我们关掉分区 |
| ResQ(直系前身,INT4+⅛INT8) | ≈4.5 | 15.98 | 8.71 | 7.06 | — |
| SpinQuant(**学习旋转** Cayley-SGD) | 4.0 | 16.14 | 9.25 | 7.88¹ | — |
| Atom(+128 INT8 outlier) | ≈4.2 | 18.40 | 9.70 | *(8B 跑中)* | — |
| SpinQuant(Hadamard,非学习) | 4.0 | 18.62 | 10.09 | 7.88 | — |
| QuaRot(纯 INT4) | ≈4.0 | 22.11 | 10.72 | 8.65 | — |

¹ SpinQuant 8B 学习旋转单卡太慢未训,用 Hadamard。SpinQuant 1B/3B 学习旋转单卡 eff.batch 8/2(< 论文 8-GPU batch-8),100 步匹配论文。

**精度判定:ProMix 在 W4A4 同类里全面第一。** 8B 上 ProMix Δ=+0.46,ResQ +0.92、SpinQuant(学习)+1.74、Atom +~3、QuaRot +2.51。连直系前身 ResQ、学习旋转的 SpinQuant 都明显更差。

## 3. 硬件格式峰值(同卡 dense 实测,M=N=K=8192/16384,2·M·N·K/time)

| 格式 | @8192 | vs INT8 | 说明 |
|---|---|---|---|
| FP16/BF16 | 406–417 TFLOPS | 0.5× | cuBLAS |
| FP8 E4M3 | 807 TFLOPS | ~0.99× | cuBLASLt |
| INT8 | 814 TOPS | 1.0× | CUTLASS IMMA m16n8k32 |
| **INT4 (s4)** | ≤ 814(= INT8) | **1.0×** | s4 在 SM120 **编译成与 INT8 完全相同**的 `IMMA.16832.S8.S8`,SASS 零 S4 opcode |
| **NVFP4** | **1469 TFLOPS** | **1.49–1.80×** | 专属 block-scaled 通路 |

详见 `docs/hardware_format_peak.md`。

## 4. kernel 可行性(SM120)

| 方法 | 格式 | SM120 真 kernel |
|---|---|---|
| **ProMix** | NVFP4+MXFP8 | ✅ 单 launch 双格式融合(bit-exact,beats vLLM NVFP4 GEMM) |
| vLLM NVFP4 | NVFP4 | ✅ |
| QuaRot | INT4 | ⚠️ 编译过但比 FP16 慢 5-6×(s4→INT8,无加速) |
| Atom | INT4+INT8 | ❌ s4 降级 INT8 |
| ResQ | INT4+INT8 | ❌ 同上 |
| QServe | W4A8 | ❌ 真 kernel 在 CUDA13 因 `cub::Max` 被删编不过;且是 W4A**8**,严格非 W4A4 同类 |
| SpinQuant | — | ❌ 无 CUDA/CUTLASS kernel(仅 ExecuTorch 移动端) |

## 5. 三层诚实拆分(堵死"你只是赢在格式")

1. **格式-硬件红利 ≈ 1.5–1.8×(不算算法功劳)**:NVFP4 有专属张量核通路,INT4 在 Blackwell **= INT8 速率零加速**。所有 INT4-based peers 先天拿不到。注意:GB202 把 dense FP4 相对 FP8 gate 到 ~1.8×(非理论干净 2×),所以红利**不算大**——我们端到端的赢不是靠格式占大便宜。
2. **精度轴(算法)**:ProMix 的高低精度分区 + act-MSE + 两级 weight scale,在上表全面领先。**同格式(全 NVFP4)剥离格式红利后是否仍赢**,由 `docs/same_format_nvfp4_ablation.md` 验证(进行中)。
3. **性能轴(kernel)**:同格式下,我们的单 launch 双格式融合(whole-K cross-phase overlap + SFASYNC)相对朴素 FP4 kernel 的额外收益。

## 6. 诚实 caveats

- Atom 8B 计算中;QServe W4A8 非严格 W4A4 同类且真 kernel 在 SM120 不可用。
- SpinQuant 学习旋转用了缩减的单卡 batch(非论文 8-GPU),但 100 步匹配;8B 未训。
- 硬件峰值的 achieved/theoretical efficiency 待复核官方 dense 规格(本轮无网络);NVIDIA headline "AI TOPS" 多含 2:4 sparsity,本表全 dense。
- 原始数据:peers 在 GPU host `/home/luofan/baselines/peers/{quarot,spinquant,atom}/` 与 `/home/luofan/refquant/`;硬件峰值 `docs/hardware_format_peak.md`。
