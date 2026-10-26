# 统一记分牌 —— 精度轴 × 性能轴(loop 的唯一判据)

> 2026-10-26 建立。**所有优化轮必须以本表为准:不赢 FP16,赢 FP16 不算成绩;要赢的是"精度效率"和"对 FP16 的加速比"。**
> 数据来源:同硬件 RTX Pro 6000 (SM120) 实测。我们的数为真·kernel 路径;related work 为 vLLM 0.32 + llm-compressor(GPTQ/AWQ 已核正配方,NVFP4/FP8 已核原论文)。

## 轴 1:精度效率(同 bit 比 PPL;低 bit 高精度为赢)

| 方法 | bit/weight | 1B PPL | 3B PPL | 8B PPL |
|---|---|---|---|---|
| BF16(FP16 参照) | 16 | 13.16 | 7.82 | 6.14 |
| **vLLM NVFP4 W4A4** | **≈4.25** | 15.65 | 8.63 | 6.87 |
| **ProMix W4A4 — SHIPPED(r=1/8, act-MSE快+weight-A2)** | **4.97** | 14.76* | **8.376** | **6.713** |
| ProMix W4A4 r=1/16 精度最大 | 4.73 | 14.83* | **8.463** | — |
| GPTQ W4A16 | 4.0 | 14.22 | 8.20 | 6.52 |
| AWQ W4A16 | 4.0 | 14.53 | ~8.35 | 6.59 |
| FP8 W8A8 | 8.0 | 13.30 | — | — |

\* 1B 用 pipeline PPL 协议,与其余行(vLLM 绝对协议)不同,**未对齐,不能横比**;3B/8B 已同协议。
SHIPPED = round-4 真·kernel,**act-MSE 现默认开**(round-4 做成便宜生产端:fused producer 2.1×→**1.10×**、纯 pack 3.3×→**1.06×**,<1.15× 门;PPL 与 round-3 真除法搜索逐位相同)+ weight-A2(1B 关:+0.08;3B/8B 开)+ hadamard28 + decode-v5。
act-MSE off 基线(weight-A2 only):3B 8.545、8B 6.809;act-MSE 再降 3B −0.17、8B −0.10。

**判定(round-4):act-MSE 便宜化后,精度最大即 SHIPPED——在 W4A4 绝对 PPL 上超过 NVFP4(3B 8.38<8.63、8B 6.71<6.87),但多花 0.72 bit,精度/bit 仍不压 NVFP4。**

## 轴 2:性能(统一为"对同 stack FP16/BF16 的加速比")

| 方法 | prefill 1B/3B/8B | decode b1 1B/3B/8B | decode b8 1B/3B/8B |
|---|---|---|---|
| vLLM NVFP4 W4A4 | **1.64 / 1.85 / 2.21×** | **1.48 / 1.83 / 1.61×** | **1.51 / 1.64 / 1.45×** |
| vLLM AWQ W4A16 | 0.91 / 0.89 / 0.89 | 1.55 / 1.75 / 2.53 | 1.58 / 1.99 / 2.45 |
| vLLM GPTQ W4A16 | 0.86 / 0.39 / 0.89 | 1.52 / 0.92 / 2.51 | 1.57 / 1.01 / 2.47 |
| vLLM FP8 W8A8 | 1.31 / — / — | 1.23 / — / — | 1.40 / — / — |
| **ProMix SHIPPED (round-4, act-MSE 开)** | **1.25 / 1.34 / 1.29×**(M8192;M2048 1.17/1.35/1.27) | **0.73 / 0.85 / 1.06×** | **0.70 / 0.82 / 1.01×** |

(prefill/decode 均 graphed-vs-graphed,idle GPU,一卡一任务。round-4:act-MSE 现默认**开**且便宜(生产端 1.06–1.10×)。真·A/B(act-MSE 开 vs 关,同环境):1B/3B prefill+decode **开==关**(免费);8B prefill 仅 −0.03×(graph 1.29 vs 关 1.31),decode 不变——对比 round-3 旧搜索把 8B prefill 砸到 0.78×、decode-b8 到 0.80×。即:**精度最大与性能最大现为同一 operating point**。prefill cos 1.0000005–1.0000014,decode M=1 位精确保持。)

**判定(round-4):act-MSE 大精度赢(3B −0.17、8B −0.10)现以 ~0.03× 的 8B prefill 代价随 SHIPPED 一起发(1B/3B 免费);8B 仍在 prefill/decode 同时赢 FP16,但三项仍落后 NVFP4(2.21×/1.61×/1.45×)。1B/3B decode <1(M=1 launch-bound)。**
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
