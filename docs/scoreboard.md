# 统一记分牌 —— 精度轴 × 性能轴(loop 的唯一判据)

> 2026-10-26 建立。**所有优化轮必须以本表为准:不赢 FP16,赢 FP16 不算成绩;要赢的是"精度效率"和"对 FP16 的加速比"。**
> 数据来源:同硬件 RTX Pro 6000 (SM120) 实测。我们的数为真·kernel 路径;related work 为 vLLM 0.32 + llm-compressor(GPTQ/AWQ 已核正配方,NVFP4/FP8 已核原论文)。

## 轴 1:精度效率(同 bit 比 PPL;低 bit 高精度为赢)

| 方法 | bit/weight | 1B PPL | 3B PPL | 8B PPL |
|---|---|---|---|---|
| BF16(FP16 参照) | 16 | 13.16 | 7.82 | 6.14 |
| **vLLM NVFP4 W4A4** | **≈4.25** | 15.65 | 8.63 | 6.87 |
| **ProMix W4A4 — SHIPPED(r=1/8, weight-A2)** | **4.97** | 14.85* | **8.533** | **6.799** |
| **ProMix W4A4 — 精度最大(act-MSE+weight-A2)** | **4.97** | 14.76* | **8.376** | **6.715** |
| ProMix W4A4 r=1/16 精度最大 | 4.73 | 14.83* | **8.463** | — |
| GPTQ W4A16 | 4.0 | 14.22 | 8.20 | 6.52 |
| AWQ W4A16 | 4.0 | 14.53 | ~8.35 | 6.59 |
| FP8 W8A8 | 8.0 | 13.30 | — | — |

\* 1B 用 pipeline PPL 协议,与其余行(vLLM 绝对协议)不同,**未对齐,不能横比**;3B/8B 已同协议。
SHIPPED = round-3 真·kernel,weight-A2 默认(1B 关:+0.08;3B −0.08;8B −0.02)+ hadamard28 + decode-v5;
act-MSE 默认关(它 5.3× 生产端 pack → 8B prefill 1.32→0.78x),精度最大行为其 opt-in 配置(fake-quant,real OOM 时以 fake 代入,round2 已验)。

**判定(round-3):精度最大配置在 W4A4 的绝对 PPL 上已超过 NVFP4(3B 8.38<8.63、8B 6.72<6.87),但多花 0.72 bit,精度/bit 仍不压 NVFP4;weight-A2 是免费小赢(3B/8B),act-MSE 是大赢但代价是 prefill(未 ship)。**

## 轴 2:性能(统一为"对同 stack FP16/BF16 的加速比")

| 方法 | prefill 1B/3B/8B | decode b1 1B/3B/8B | decode b8 1B/3B/8B |
|---|---|---|---|
| vLLM NVFP4 W4A4 | **1.64 / 1.85 / 2.21×** | **1.48 / 1.83 / 1.61×** | **1.51 / 1.64 / 1.45×** |
| vLLM AWQ W4A16 | 0.91 / 0.89 / 0.89 | 1.55 / 1.75 / 2.53 | 1.58 / 1.99 / 2.45 |
| vLLM GPTQ W4A16 | 0.86 / 0.39 / 0.89 | 1.52 / 0.92 / 2.51 | 1.57 / 1.01 / 2.47 |
| vLLM FP8 W8A8 | 1.31 / — / — | 1.23 / — / — | 1.40 / — / — |
| **ProMix SHIPPED (round-3)** | **1.29 / 1.35 / 1.32×**(M8192;M2048 1.22/1.39/1.32) | **0.72 / 0.85 / 1.06×** | **0.71 / 0.83 / 1.03×** |

(对 FP16-graph,idle GPU。round-3 vs 上一轮:8B prefill 1.22→1.32×(hadamard28 融合生产端),8B decode b1 0.81→1.06×、b8 1.01→1.03×(decode-GEMV v5 可调 warps);1B/3B 基本持平——perf 杠杆主要作用于 8B。act-MSE **默认关**:开它 8B prefill 掉到 0.78×。)

**判定(round-3):8B 首次在 prefill(1.32×)与 decode b1/b8(1.06/1.03×)同时赢过 FP16;但三项仍全面落后 NVFP4(2.21× / 1.61× / 1.45×)。1B/3B decode 仍 <1(M=1 launch-bound / GEMV 占比小)。**
**目标:prefill ≥ NVFP4(8B ≥2.2×)、decode 全面 ≥ NVFP4;并把 act-MSE 做成便宜生产端以同时拿精度。**

## loop 规则(KDA 式,controller 串行调度)

1. **每轮只动一条轴的一个杠杆**(算法或性能),先做数值门,再做测量,再提交;
2. **门**:精度轮 = 同协议 PPL(1B 协议对齐后)/ 性能轮 = 同协议加速比 + 其它既有门(decode/bit-exact)不破;
3. **证据驱动**:算法轮先做消融定位(高精度段通道选择、GPTQ 覆盖、旋转作用、r 扫描),性能轮先 profile;
4. **不赢就不算完**:精度轴要在 **同 bit** 上赢 NVFP4;性能轴要在**同协议**上超过或逼近 NVFP4/Marlin;
5. 每轮结束更新本表;数值或协议不一致的结果不允许进表。

## 先验假设(待验证,按价值排序)

- A1:1/8 MXFP8 高精度段按**激活方差**选通道,但**权重误差热点**不在这批通道 → 0.72 bit 花错地方(可测:按权重误差选通道 vs 激活方差);
- A2:MXFP8 段只做 RTN + 冻结 scale,**没有 GPTQ 误差补偿** → 高精度段的潜力没发挥;
- A3:旋转对**权重**的 block-scaled FP4 是否最优(与 QuaRot/SpinQuant 不同,我们是 PCA+随机正交);
- A4:r 不是最优点(扫 r=0/1/16/1/8/1/4/1/2,画 bit-vs-PPL 曲线);
- P1:decode 落后 = 无 KV4 + attention 未量化 + GEMV 未到 HBM roof;
- P2:prefill 落后 NVFP4 = 未开 8B 的 Hadamard 融合 + epilogue/生产端融合还未全覆盖。
