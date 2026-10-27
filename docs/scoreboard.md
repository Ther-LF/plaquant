# 统一记分牌 —— 精度轴 × 性能轴(loop 的唯一判据)

> 2026-10-26 建立。**所有优化轮必须以本表为准:不赢 FP16,赢 FP16 不算成绩;要赢的是"精度效率"和"对 FP16 的加速比"。**
> 数据来源:同硬件 RTX Pro 6000 (SM120) 实测。我们的数为真·kernel 路径;related work 为 vLLM 0.32 + llm-compressor(GPTQ/AWQ 已核正配方,NVFP4/FP8 已核原论文)。

## 轴 1:精度效率(同 bit 比 PPL;低 bit 高精度为赢)

> **口径分层(2026-10-27 修正)**:我们的工作是 **W4A4(权重+激活同时量化)**。因此:
> **主对比组 = 同为 W4A4 / 激活也量化的方法**(vLLM NVFP4、以及论文口径下的 QuaRot/SpinQuant/Atom/QServe/ResQ W4A4KV4)。
> **W4A16(GPTQ/AWQ)只作参考**:它们的激活是 FP16、不付激活量化成本,与 W4A4 不构成同条件比较,不应作为"我们精度更差"的依据。
> **bit/weight 已纠正**:NVFP4 含 E4M3 scale 是 **4.50 bits/weight**(不是 4.25);我们 r=1/8 是 4.97(+0.47),r=1/16 是 4.73(+0.23),r=0 与 NVFP4 **完全同 bit(4.50)**。

### 主对比组 —— W4A4(激活也量化)

| 方法 | bit/weight | 1B PPL | 3B PPL | 8B PPL |
|---|---|---|---|---|
| BF16 参照 | 16 | 13.16 | 7.82 | 6.14 |
| **vLLM NVFP4 W4A4** | **4.50** | 15.65 | 8.63 | 6.87 |
| **ProMix W4A4 — SHIPPED(r=1/8)** | **4.97** | **14.563** | **8.296** | **6.599** |
| ProMix W4A4 r=1/16 | 4.73 | — | **8.361** | — |
| ProMix W4A4 r=0(纯 NVFP4,**同 bit**) | **4.50** | — | 8.859 | — |
| *(round-13 同协议实测)* QuaRot W4A4KV4(GPTQ W4+per-token A4+asym KV4) | ≈4.0 | 22.11 | 10.72¹ | 8.65 |
| *(round-13 同协议实测)* ResQ W4A4KV4(INT4+1/8 INT8,hf=0.125) | ≈4.5 | 15.98 | 8.71 | 7.06 |
| *(round-13 同协议实测)* Atom W4A4KV4(128 INT8 keeper+g128+GPTQ+INT4 KV) | ≈4.2 | 18.40 | 9.699 | *(8B running)* |
| *(round-14 同协议实测)* SpinQuant W4A4KV4(Hadamard,**非学习**) | ≈4.0 | 18.623 | 10.092 | 7.882 |
| *(round-14 同协议实测)* SpinQuant W4A4KV4(**学习旋转** Cayley-SGD) | ≈4.0 | **16.143** | **9.246** | *(未训,用 Had 7.882)* |
| *(论文口径,待同协议复测)* QServe W4A8KV4 | 4/8 | — | — | — |

> **round-13 W4A4 主对比组填充(QuaRot + ResQ,同协议实测)**。协议 = `docs/baselines_on_hardware.md` 的 wikitext-2 PPL(bf16、`add_special_tokens=False`、`"\n\n".join`、非重叠 2048 chunk=141、288,627 token、token-pooled exp(mean NLL))。**harness bridge 验证**:同一纯-torch full-forward evaluator 在未量化 bf16 上复现committed 基线 **1B 13.1650 / 3B 7.8199 / 8B 6.1382**(= 13.16/7.82/6.14,逐位吻合)→ 任何 Δ 为真·量化差异。跑的是官方开源实现(QuaRot `spcl/QuaRot` fake_quant、ResQ `project-resq` fake_quant)在 SM120 上的 fake-quant 精度模式(torch 2.7.1+cu128,transformers 4.45.2/4.48.1,fast_hadamard 用 pure-torch FWHT shim 替代——数值 maxerr 1e-14 vs 显式 Hadamard 矩阵)。
> - **关键修复**:(a) Llama-3.2-1B/3B `tie_word_embeddings=True`,旋转前必须 untie+clone lm_head(否则 PPL=8e5 全崩);8B 本就 untied。(b) QuaRot fake_quant 的 o_proj online partial-Hadamard else 分支对 24-head(3B,`get_hadK(24)→K=12`)不做 K-因子分解而崩;用 canonical `matmul_hadU_cuda` 统一实现替换(对 32-head 1B/8B 逐位等价)。¹故 3B QuaRot 带此 24-head 分解修复(1B/8B 走原版 K=1 路径)。
> - **判定**:同协议下 **QuaRot/ResQ 均不如 ProMix 也不如 NVFP4**。QuaRot(纯 INT4,无 micro-scale/无高精段)在难量化的 Llama-3.x 上退化最大(1B +68%/3B +37%/8B +41%)。ResQ(INT4+1/8 INT8,ProMix 的前身)居中:3B 8.71 / 8B 7.06,略逊于 NVFP4(8.63/6.87),明显逊于 ProMix SHIPPED(8.296/6.599)——印证 ProMix 在 ResQ 之上的 act-MSE + 两级 weight scale + FP4 格式增益。ResQ 1B 15.98 ≈ NVFP4 1B 15.65。
> - **未复现(本轮)**:SpinQuant 的"学习旋转"需其 GDrive/gated checkpoint 或自训(对 3.2-1B/3B 无公开旋转),Hadamard-mode 退化为 ≈QuaRot;Atom/QServe 为另一套旧 transformers 环境。原始数据:GPU host `/home/luofan/refquant/{results_*.json,SUMMARY.md}`。
> - **Atom 补充(独立 agent,同协议)**:Atom 自身 bridge 在 FP16 复现 **13.159/7.814/6.136**(= committed 基线)→ 协议匹配。真·recipe(128 INT8 keeper + group-128 + hessian reorder + GPTQ + INT4 KV,≈4.2 b/w)fake-quant:1B **18.40**(Δ+5.24)、3B **9.699**(Δ+1.89)、8B 计算中。**明显逊于 ProMix**。Atom 又一次独立印证 **INT4 在 Blackwell 无真数据通路**:ptxas 把 `s4.s4` MMA 降级成 INT8 `IMMA.16832.S8.S8`,零 S4 SASS(与 QuaRot 的 INT4 kernel 慢 5-6× 结论一致)。原始数据:`/home/luofan/baselines/peers/atom/{RESULTS.md,results.json}`。
> - **SpinQuant 补充(独立 agent,同协议)**:FP16 bridge 复现 **13.158/7.816/6.137**(= 基线)→ 协议匹配。SpinQuant 仅 fake-quant(SM120 无 GPU kernel,无加速)。W4A4KV4 4.0 b/w(GPTQ per-channel + KV4 asym + A4 per-token):**Hadamard 旋转(非学习,= QuaRot 式默认)**:1B **18.623**(Δ+5.47)、3B **10.092**(Δ+2.28)、8B **7.882**(Δ+1.75)。学习旋转(Cayley-SGD 100 步,单卡 batch 8/2 远小于论文 8-GPU batch-8):1B **16.143**(Δ+2.99,较 Hadamard 改善 2.48)、3B **9.246**(Δ+1.43,改善 0.85)、8B 太慢未训(用 Hadamard 7.882)。**即便学习旋转,SpinQuant 4.0b(16.14/9.25/7.88)仍明显逊于 ProMix 4.97b(14.56/8.30/6.60)与 NVFP4 4.50b。** SpinQuant 在 SM120 无 CUDA/CUTLASS kernel(仅 ExecuTorch 移动端路径)→ 无 on-hardware 加速。原始数据:`/home/luofan/baselines/peers/spinquant/`(commit 8f47aa3)。

### 参考组 —— W4A16(**激活不量化**;不可作为同条件对比)

| 方法 | bit/weight | 1B PPL | 3B PPL | 8B PPL |
|---|---|---|---|---|
| GPTQ W4A16 | 4.0 | 14.22 | 8.20 | 6.52 |
| AWQ W4A16 | 4.0 | 14.53 | ~8.35 | 6.59 |
| FP8 W8A8(参考) | 8.0 | 13.30 | — | — |

\* 1B 用 pipeline PPL 协议,与其余行(vLLM 绝对协议)不同,**未对齐,不能横比**;3B/8B 已同协议。**round-7:1B base 模型已恢复,两级 scale A/B 实测 real-kernel 14.7604(单级)→14.5629(两级),Δ−0.198(远超预期 −0.05..−0.1),real-vs-twin 门 |Δ|=0.014 PASS → 1B 默认翻 ON,三档全部两级。**
SHIPPED = round-6 真·kernel:act-MSE(默认开)+ **weight 两级 scale(per-tensor FP32 global g × per-block E4M3/E8M0 micro,NEAREST+micro-MSE)**。g 搭在 fused-GEMM epilogue 的 alpha 上(零 GEMM 代价,epilogue 本就乘 alpha);decode 把 g 折进 FP32 scale。
**round-6:两级 scale 相比单级 weight-A2:3B r=1/8 −0.08(真·kernel 8.2955,fake 8.293;与 round-5 fake 8.297 差 0.0015)、3B r=1/16 −0.10、8B r=1/8 −0.11、3B r=0 −0.18。real-vs-twin gate |Δ|=0.0095≤0.1、per-layer cos 0.999998。**

**判定(round-6):两级 weight scale 现可部署且零运行代价——在 W4A4 绝对 PPL 上 r=1/8 以 8.296(3B)/6.599(8B)明显超过 NVFP4(8.63/6.87,−0.33/−0.27),r=1/16 8.361 亦超;但 matched-bit(r=0 4.5b vs NVFP4 4.25b)3B 8.859>8.63(+0.23),仍多花 0.25 bit,精度/bit 未压 NVFP4。**

## 轴 2:性能(统一为"对同 stack FP16/BF16 的加速比")

| 方法 | prefill 1B/3B/8B | decode b1 1B/3B/8B | decode b8 1B/3B/8B |
|---|---|---|---|
| vLLM NVFP4 W4A4 | **1.64 / 1.85 / 2.21×** | **1.48 / 1.83 / 1.61×** | **1.51 / 1.64 / 1.45×** |
| vLLM AWQ W4A16 | 0.91 / 0.89 / 0.89 | 1.55 / 1.75 / 2.53 | 1.58 / 1.99 / 2.45 |
| vLLM GPTQ W4A16 | 0.86 / 0.39 / 0.89 | 1.52 / 0.92 / 2.51 | 1.57 / 1.01 / 2.47 |
| vLLM FP8 W8A8 | 1.31 / — / — | 1.23 / — / — | 1.40 / — / — |
| **ProMix SHIPPED (round-4, act-MSE 开)** | **1.25 / 1.34 / 1.29×**(M8192;M2048 1.17/1.35/1.27) | **0.73 / 0.85 / 1.06×** | **0.70 / 0.82 / 1.01×** |
| **ProMix + round-7 QKV/gate-up decode MERGE** | (prefill 不变;merge 伤 prefill 0.81–0.92×) | **0.85 / 0.96 / 1.23×** | **0.79 / 0.94 / 1.07×** |
| **ProMix + round-12 W4A16 decode (M=1 weight-only, merged;新 decode 默认)** | (prefill 不变,仍 W4A4) | **1.09 / 1.28 / 1.45×** | 1.39 / 0.98 / 1.16×(b8 不变:M=8 走 GEMM) |

(prefill/decode 均 graphed-vs-graphed,idle GPU,一卡一任务。round-4:act-MSE 现默认**开**且便宜(生产端 1.06–1.10×)。真·A/B(act-MSE 开 vs 关,同环境):1B/3B prefill+decode **开==关**(免费);8B prefill 仅 −0.03×(graph 1.29 vs 关 1.31),decode 不变——对比 round-3 旧搜索把 8B prefill 砸到 0.78×、decode-b8 到 0.80×。即:**精度最大与性能最大现为同一 operating point**。prefill cos 1.0000005–1.0000014,decode M=1 位精确保持。)

**判定(round-4):act-MSE 大精度赢(3B −0.17、8B −0.10)现以 ~0.03× 的 8B prefill 代价随 SHIPPED 一起发(1B/3B 免费);8B 仍在 prefill/decode 同时赢 FP16,但三项仍落后 NVFP4(2.21×/1.61×/1.45×)。1B/3B decode <1(M=1 launch-bound)。**
**目标:prefill ≥ NVFP4(8B ≥2.2×)、decode 全面 ≥ NVFP4;并把 act-MSE 做成便宜生产端以同时拿精度。**

**判定(round-7):decode QKV+gate/up 融合(每层 7→4 launch,共享激活只量化一次)实测 1.16–1.22× decode 加速,b1 位精确(max|Δ|=0.0),b8/prefill 路径 +0.022 PPL。新 decode b1 1B/3B/8B=0.85/0.96/1.23×(8B 过 1.0,3B 近 1.0),b8=0.79/0.94/1.07×。此杠杆只对 decode(launch-bound)有效,prefill(compute-bound)被它拖慢 0.81–0.92× → decode 专用。prefill M=8192 stage-profile:3B 混合 GEMM 对 Linear 已 ~1.36×,NVFP4 2.21× 的差距是 FP16 attention(23%)+RMSNorm(13.6%)+RoPE/lm_head 的 Amdahl + 两段式混合 GEMM 结构性低于单段 NVFP4;8B 最大桶是 down_proj FP32 Hadamard(87ms/21.6%),已建的 bit-identical hpk28 融合 producer 实测无增益(occupancy-bound 1 CTA/SM)→ prefill 本轮无免费赢,加速比不变。**

**判定(round-9,口径修正 + Hadamard 赢):同协议(同权重/同 prompt 长/b1/空闲 GPU)实测四路 TTFT。此前的 prefill "1.14–1.35× vs FP16" 是对我们自己的 HF-FP16 graphed forward,而那个基线只有 vLLM BF16 的 0.80–0.93× 快。归一到同一强基线(vLLM BF16)后:我们 real-kernel prefill @8192 = 0.85/0.85/0.92×(低于 BF16),vLLM NVFP4 = 1.72/1.81/2.19×。与 round-8 "GEMM 胜 1.37×" 不矛盾:全模型差距 = 我们未优化的 FP16 attention/RMSNorm/RoPE/lm_head(Amdahl,~45%)+ W4A4 down-Hadamard 税,GEMM 本身是盈余。**
**Task-3 修复:FP32 online Hadamard 是多余的 `.float()` 上采样——fast-WHT 跑 fp16 数值位同(cos=1.000000)且 ~3× 快。8B down-Hadamard 92.8→31.5ms(−66%),全模型 8B prefill 421→370ms(−12.1%,此时已快过 vLLM BF16 388.8ms),8B prefill ×FP16 1.14→1.29×;3B 位同(PPL 不变 8.6816)36.3→10.7ms,prefill 1.05→1.17×。PPL 门过(隔离 twin Δ=−0.0005 8B / +0.0055 3B,<0.01;3B pow2 位同)。`mixed`(fp32 K-matmul)与纯 fp16 PPL 完全一致 → 纯 fp16 即可,radix-4 fp32 无必要。ship `PROMIX_HAD_MODE=fp16`。**
**Task-2 FP8-KV 接入 graphed decode(e4m3 per-vector,写时量化、读时反量化 FP16 attention):显存 1.94×(1B)/1.97×(3B/8B);PPL Δ +0.058(1B opt-in)/+0.015(3B)/+0.012(8B)。decode 速度 0.90–0.99×(略慢)——FP16 attention 读回 fp16,无带宽收益 → FP8-KV 是显存/容量杠杆,非 decode 加速杠杆(要加速需 fp8-native attention)。**

**判定(round-10,FP16 non-GEMM stack 攻坚):per-op profile(同 shape、同进程、median-of-50)定位到非-GEMM 差距几乎全在 RMSNorm+RoPE,attention 是平手(SDPA≈FlashAttention,8B 1.05×)。8B@M8192 model-level:RMSNorm ours 51.3ms vs vLLM 3.1ms(16.5×)、RoPE 17.3 vs 5.6ms(3.1×)、attn 59.7 vs 56.6ms。两个 Triton 融合 kernel(fused RMSNorm drop-in + fused RoPE)都过 PPL 门:RMSNorm 3B Δ+0.0032/8B −0.0062,norm+rope 3B +0.0062/8B −0.0100。prefill@M8192 ×our-FP16:3B 1.16→1.38、8B 1.28→1.52;×vLLM-BF16(forward-only,保守)3B 0.94→1.11、8B 1.05→1.25——从低于 BF16 翻到明确超过。另发现一个免费 TTFT 杠杆:我们 prefill 算全 [M,V] logits(8B lm_head 21.4ms)而 vLLM TTFT 只算末 token(0.7ms),切片(HF logits_to_keep=1)可省 16–21ms → 投影 8B ≈1.34× / 3B ≈1.23× vLLM-BF16(未接线)。fp16 Hadamard 已设为默认(PROMIX_HAD_MODE=fp16)。残余对 NVFP4 的差距不再是 FP16 stack(已追平/超过 vLLM),而是两段式 mixed-GEMM 结构 + down-Hadamard 税。**

**判定(round-12,DECODE 相位自适应 = W4A16):设计级修复——decode(M=1)带宽/延迟受限,量化激活(W4A4)不占张量核、买不到吞吐,只多一遍 pass 且丢精度。新 `decode_gemv_w4a16`(体同 W4A4 GEMV,但不量化激活、仅在寄存器反量化 4/8-bit 权重,像 Marlin;权重 HBM 字节与 W4A4 完全相同),挂 `PROMIX_DECODE_MODE=w4a16|w4a4`(默认 w4a16),prefill 不动仍 W4A4。门全过:逐 shape kernel cos≥0.9999998(含 8B K=14336 smem-opt-in);graph==eager argmax 100%/cos 0.99999;全模型 W4A16-vs-exact 比 shipped W4A4-vs-自身 quant twin 更忠实(1B 0.978>0.889、3B 0.977>0.955)。精度(fake A/B,GPTQ 冻结 scale,wikitext-2):W4A16 关激活量化免费回收 PPL **1B 14.58→13.66(−0.918)、3B 8.29→7.97(−0.318)、8B 6.59→6.31(−0.284)**,逼近 BF16、超 GPTQ/AWQ 纯 W4A16 基线(但我们 4.97 bit vs 其 4.0,非同 bit)。decode b1 ×FP16-graph(graphed/median/同基线/merged)**1B/3B/8B=1.09/1.28/1.45×**(W4A4 0.85/0.96/1.17;首次 1B/3B b1 过 1.0),加速来自去掉每 token 激活量化那一遍(1B merged 4.22→3.27ms,权重流量不变)。merge 仍保留(量化去重理由消失但 7→4 launch 在 M=1 仍赢:1B 1.024→1.094、3B 1.172→1.276、8B 1.376→1.450)。b8 不变(M=8 走 prefill GEMM,非 M=1 GEMV)。仍落后 Marlin b1(1.55/1.75/2.53):残因 =(1)bit/weight 4.97 vs 4.0 +24% 权重带宽(ResQ 1/8 MXFP8 高精段,结构性);(2)M=1 launch/occupancy floor(1B 仍 ~8× 离权重-BW 天花板);(3)FP16 attention/norm Amdahl(~45% decode);(4)GEMV layout 不如 Marlin 交错-fp16。**

### 轴 2 协议对齐表(round-9,TTFT ms @M8192,b1,空闲 GPU;口径见下)

| 口径 | 1B | 3B | 8B |
|---|---|---|---|
| our FP16(forward-only,graphed,无调度/采样) | 84.95 | 229.14 | 482.83 |
| our real-kernel(forward-only,fp32-had) | 92.98 | 217.63 | 422.11 |
| **our real-kernel(fp16-had,round-9 ship)** | — | **197.5** | **370.3** |
| **our real-kernel + Triton fused RMSNorm(round-10)** | 76.0 | **172.3** | **322.2** |
| **our real-kernel + fused RMSNorm+RoPE(round-10)** | 75.0 | **165.9** | **311.2** |
| vLLM BF16(TTFT=调度+prefill+采样1 token) | 78.6 | 183.9 | 388.8 |
| vLLM NVFP4(同 vLLM TTFT) | 45.7 | 101.5 | 177.9 |

归一到同一强基线 vLLM BF16(×BF16):our real-kernel(fp16-had)= 0.85/0.93/**1.05×**;vLLM NVFP4 = 1.72/1.81/2.19×。**警告:our 为纯 forward(不含 vLLM 的调度+采样),对我们偏乐观,故"低于 BF16"是保守结论。**

**round-10 更新:Triton 融合 RMSNorm+RoPE 后 ×vLLM-BF16 = 1.05/1.11/1.25×(1B/3B/8B,原 fp16-had 0.97/0.94/1.05×),3B/8B 翻到超过 BF16。per-op profile 证明非-GEMM 差距全在 RMSNorm(8B 16.5×)+RoPE(3.1×),attention 平手(1.05×)。另:lm_head 全 [M,V] logits(8B 21.4ms)vs vLLM TTFT 末-token(0.7ms)是免费杠杆(切片省 16–21ms),投影 8B ≈1.34×。两个融合 kernel 过 PPL 门(3B +0.0032/+0.0062,8B −0.0062/−0.0100)。**


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
