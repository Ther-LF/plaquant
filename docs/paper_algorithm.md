# ProMix 算法规格(论文 Method 节的精确蓝本)

> 2026-11-03。写作者据此直接写 Method:记号、公式、三个伪代码算法、精确配置、每步成本。
> 实现对应:`promix/quantize/{basis,ring_rotation,adaptive,quant_utils}.py`、`promix/scripts/rd_ron.py`、`promix/vllm_integ/`。

---

## 0. 记号

| 符号 | 含义 |
|---|---|
| X ∈ R^{T×d} | 某线性层的输入激活(T token,d 通道);W ∈ R^{N×d} 权重 |
| U ∈ R^{d×d} | PCA 基(正交):校准协方差 C=E[xxᵀ] 的特征向量,按特征值**降序** |
| R1 = blockdiag(H_S,…,H_S) | 环旋转:每 S 通道一个归一化 Sylvester-Hadamard 块(**S=512**),正交 |
| T = U·R1 | 完整旋转;**离线 bake**:embedding 行乘 T,各权重 W←WT(residual-stream 全模型共享一个 T,在线零开销) |
| 环(ring) | 旋转后坐标里连续 64 通道一组(分配/量化原子;64 = lcm 约束见 §3) |
| Q_f(·) | 格式 f 的块缩放量化-反量化:f=FP4 → NVFP4(E2M1,块16,E4M3 micro-scale);f=FP8 → MXFP8(E4M3,块32,UE8M0 scale) |
| D_f(j) | 环 j 用格式 f 的**实测失真**:D_f(j)=mean_{t,c∈ring j}(x̃_{t,c}−Q_f(x̃)_{t,c})²,x̃=xT 为旋转后激活 |
| ΔD(j) | 升级收益:ΔD(j)=D_FP4(j)−D_FP8(j) ≥ 0 |
| B | **唯一旋钮**:全模型平均 FP8 环比例(预算);θ = 对应的 ΔD 阈值(由 B 反推,一一对应) |
| K_high^(ℓ), K_low^(ℓ) | 层 ℓ 的高/低精度通道数(= 选中环数×64 / 其余) |

## 1. 问题形式化

给定预算 B,为每个线性层的每个环选格式 f(j)∈{FP4,FP8},最小化总失真:

  min Σ_j D_{f(j)}(j)  s.t.  (#FP8 环)/(#总环) ≤ B,K_high^(ℓ)≡0 (mod 32),K_low^(ℓ)≡0 (mod 64) ∀ℓ

两格式下的拉格朗日解 = **全局单阈值**:f(j)=FP8 ⟺ ΔD(j) ≥ θ(B)。
(若引入每环成本 C_f(j),判据为 ΔD/ΔC ≥ 1/μ;等环成本时退化为按 ΔD 排序 —— 我们发货用这个。)

## 2. Algorithm 1 — 离线量化流水线(一次,分钟~小时级)

```
输入: 预训练模型 M, 校准集 D_cal(wikitext train 512 条×2048 tok, 固定种子), 预算 B
输出: 量化 checkpoint(packed 权重+scales+每层K映射)

1  C_ℓ ← 校准前向收集每个目标线性层输入的协方差            # Step-0, hooks
2  U ← eig(C) 按特征值降序; R1 ← blockdiag(H_512,…); T ← U·R1  # 环旋转基座
   # down_proj 例外: intermediate 维过大, 用在线 Hadamard(摊平), 不参与 T
3  bake: embedding←E·T; 各 W←WT(+逆旋转对应处); lm_head 解 tie   # 在线零开销
4  x̃_ℓ ← 再一遍校准前向, 收集旋转后激活(每层 ≤4096 tok 够)
5  for 每层 ℓ, 每环 j: D_FP4, D_FP8 ← 实测量化 MSE(用部署同款量化器); ΔD(j)←差
6  {f(j)} ← Algorithm 2(ΔD, B)                                  # 分配
7  按 f(j) 把每层通道重排成 [FP4 段 | FP8 段](reorder-before-quant, 折进 W 与后续层)
8  权重量化: GPTQ(nsamples 1024/512/512 按 1B/3B/8B) + 两级 scale
   (per-tensor FP32 g × per-block micro-scale, NEAREST+micro-MSE 搜索, depth7 仅 3B/8B)
9  打包: NVFP4 2×4bit/byte + MXFP8 + 交错 SF 布局 + g; 落 checkpoint(含每层 K_high)
```

**成本**:Step-0 ~16min(3B);ΔD 实测一遍校准前向(分钟);GPTQ 标准成本。全离线。

## 3. Algorithm 2 — 率失真环分配(核心,毫秒级)

```
输入: 所有层所有环的 ΔD 值(共 n 个), 预算 B
输出: 每环格式 f(j)

1  L ← 所有 (ΔD(j), 层ℓ, 环j) 摊平成一个全局列表
2  按 ΔD 降序排序 L
3  k ← round(B·n); 前 k 个 → FP8, 其余 → FP4         # θ = L[k].ΔD 即阈值
4  每层 K_high^(ℓ) ← (该层被选环数)×64                 # 对齐自动满足:
   # 64≡0 (mod 32) ⇒ K_high%32=0; d−K_high≡0 (mod 64) ⇒ K_low%64=0
5  return f, {K_high^(ℓ)}
```

性质(论文里要写):
- **θ↔B 一一对应**(单调):用户可给 B(性能目标)、给可接受 PPL(精度目标,二分 B)、或自动取膝盖;
- **每层 r_ℓ 自然浮动**:outlier 重的层多拿,down_proj 自然拿 **0**(ΔD 荒漠,<1% 总 ΔD);
- **为什么可行(环旋转的作用)**:环块对角 R1 使任何"整环"选择都自洽(环内混合、环间独立)—— 这是"每层不同比例 + 保留旋转"能共存的结构前提(稠密 R1 做不到,见消融:ring512 净赢 dense −0.065@3B);
- 64 的由来:lcm(NVFP4 块 16, MXFP8 块 32, K_low%64 kernel 约束) 的最细可行原子。

## 4. Algorithm 3 — 运行时(serving,每 forward)

```
输入: 激活 x(已在旋转坐标, 免费), 层 ℓ 的 packed 权重与 K_high^(ℓ), batch-M
1  if M == 1:            y ← W4A16-GEMV(x_fp16, packed_W)        # 相位自适应: 免激活量化, 精度反升
2  elif M ≤ 48(按形状窗口): y ← marlin-low(FP4段) + MXFP8-marlin-hi(高段) 累加
3  else(prefill/大M):
4     x_pack ← 单launch量化打包: FP4段(块16, MSE scale 搜索: 下3步+上2步) ‖ FP8段(块32)
5     y ← FusedDualGEMM(x_pack, W_pack): NVFP4相 + MXFP8相 → 共享FP32累加器, epilogue×g, 单launch
   (down_proj 前插在线 Hadamard; 窗口均按形状/每 captured graph 静态选择, CUDA-graph 安全)
```

## 5. 量化器精确定义(公式)

- **NVFP4**:块 16;micro scale s_b = E4M3-round(max|x_b|/6);激活侧对候选 s∈{s_b 的 E4M3 格点 下3/上2 步} 取 argmin‖x−Q(x;s)‖²(**act-MSE 搜索**;上向 2 步是实测新增,−0.032);权重侧两级:x̂ = g·s_b·q,g=per-tensor FP32,s_b 经 micro-MSE 搜索。
- **MXFP8**:块 32;s_b = UE8M0(2 的幂,round-up)。
- 有效比特:NVFP4 = 4+8/16 = **4.5**;MXFP8 = 8+8/32 = **8.5**;ProMix(B) ≈ 4.5+4B(B=1/8 → 4.97)。

## 6. 发货配置(论文 reproducibility 清单)

| 项 | 值 |
|---|---|
| 旋转 | T=U·R1,ring S=512,固定 Hadamard(不训练);down_proj 在线 Hadamard |
| 分配 | Algorithm 2,实测 ΔD,等环成本;B 为旋钮(主报告点 1/8;曲线 1/32..1/4) |
| 校准 | wikitext-train 512 条×2048(基/ΔD/act-scale);GPTQ nsamples 1B:1024, 3B/8B:512 |
| 权重 | GPTQ + 两级 scale(micro-MSE;depth7 仅 3B/8B);act_order 关 |
| 激活 | 块 MSE 搜索 d3u2;decode M=1→W4A16 |
| 例外层 | down_proj 全 FP4(实测最优,显式对照);o_proj 按 per-head 处理+重排;lm_head/embedding FP16(主配置) |
| kernel | 单 launch 双相融合 + 路由(GEMV/marlin/split-K/GEMM 按 M/形状窗口,全部 graph-replay 标定) |

## 7. 与 naive 设计的差异点(写作时逐个强调)

1. **实测 ΔD ≠ 方差代理**:方差代理同预算反而比 uniform 差(8.47>8.39),实测 ΔD 赢(8.31)—— 失真必须量,不能猜(抓 clipping/重尾)。
2. **全局排序 ≠ 每层固定比例**:uniform 浪费 41% 预算;全局调配是增益来源。
3. **环旋转 ≠ 全宽 Hadamard**:PCA 集中 + 局部 64/512 块混合;anti-rotation 批评(ARCQuant/DualQuant)针对的全宽摊开不适用于我们。
4. **单 launch 双格式 ≠ 残差 K 扩展/分段多 GEMM**:ARCQuant 扩 K(+S 列)、MicroMix 3 段 GEMM、SharQ 双 GEMM —— 我们一个 mainloop 两相共享累加器,这是"任意混合比例可部署"的前提。
5. **相位自适应格式**(decode W4A16)——vLLM 生态里无人按 captured-graph 尺寸切换数值格式;顺带精度+0.32。
