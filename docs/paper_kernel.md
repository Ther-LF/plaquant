# ProMix kernel 设计与优化(论文 Systems/Kernel 节的精确蓝本)

> 2026-11-03。单 launch 双格式融合 GEMM 的设计、15+ 轮优化的杠杆清单(含诚实负收益)、与竞品 kernel 的结构对比。
> 实现:`kernels/mixed_gemm_sm120/`(b1_tma_ext 主 kernel、cd_splitk/cd_smallm、decode_gemv、quant_pack、codesign_* 研究线);数据出处 `docs/b1_opt_round*.md`、`docs/vllm_perf_round1-5.md`。

---

## 1. 核心结构:单 launch 双格式融合 GEMM

**问题**:混合精度线性层 y = x_low·W_lowᵀ + x_high·W_highᵀ,低段 NVFP4(块16)、高段 MXFP8(块32)。naive 实现 = 两次 GEMM + 一次 add(两次输出显存往返)。

**我们的设计**:一个 kernel、一个 mainloop,**两相顺序执行、共享同一块 FP32 寄存器累加器**:
```
Phase-1: NVFP4 块缩放 MMA 扫 K_low        ┐ 同一 CTA、同一累加器片段
Phase-2: MXFP8 块缩放 MMA 扫 K_high(不清零)┘ → epilogue ×(g·s_x) → fp16/bf16 输出
```
**可行性的硬件依据**(论文要写):SM120 的两个块缩放 MMA 原子 `16x8x64(NVFP4)` 与 `16x8x32(MXFP8)` 的 **C-fragment 布局相同**(SM80_16x8_Row,FP32×4/线程)→ 累加器可以跨格式复用,这是"单 launch 异构融合"成立的根。零中间显存、零第二次 launch、add 免费(累加线性性)。

**与竞品 kernel 的结构对比**(差异化核心):
| 方案 | 结构 | 代价 |
|---|---|---|
| **ProMix** | 单 launch 双相共享累加器 | 混合比例任意、随 r 连续 |
| ARCQuant | 纯 NVFP4 + 残差通道**拼进 K 维**(K+S),标准 GEMM | 残差列数固定;=隐式承认单格式才能流水线 —— 被我们证伪 |
| MicroMix | **3 段 GEMM**(MXFP4/6/8 各一段) | 多次 launch/写回 |
| SharQ | 稀疏骨干 + 稠密残差**双 GEMM**(≈2× FP4 工作量) | 翻倍计算 |
| FGMP | 同 NVFP4+FP8 配对但 **ASIC-only**(自认软件实现不可行) | 我们在真 GPU 上做到了 |

## 2. 优化杠杆清单(按演进序;全部 bit-exact gate 后发货)

### mainloop 级(15 轮 KDA kernel loop)
1. **多 warp 128×128 CTA + cp.async 双缓冲 + ldmatrix**(基础管线);
2. **KPIPE**:寄存器级 K-block 流水(copy(k+1) 先于 mma(k),CUTLASS 式),wait stall↓,TC +1-4pp;
3. **SFASYNC**:scale-factor 从"consumer 同步 gmem 读"改为 **cp.async 挂同一 mbarrier** 的 producer 预取 —— long_scoreboard 全线下降,store bank-conflict 425k→0,延迟 −2~−10%;
4. **per-shape config dispatch**:3250-配置笛卡尔扫(tile×stages×warp布局×raster swizzle),逐 (N,K,r) 选最优 —— 关键轴是 **warp 布局 W4×2** 与 **group-M rasterization**(非 tile 尺寸);
5. **persistent scheduler**(大 N 形状,gate/up):grid=#SM 持续调度,大 N excess −3~−6pp;
6. **TMA**(Phase-2 MXFP8):SW64 swizzle + mbarrier expect_tx,大 K 形状 mio_throttle −82%;
7. **split-K/stream-K 跨双相**(serving 期新增):SM90 stream-K 调度器驱动 **union [NV|MX] k-tile 域**、一次 FP32 fixup —— decode 中 M 的 latency-bound 区(24-48 CTA 吃不满)8B down 1.48-1.66×;
8. **小 M tile 变体**(128×64 + stage 变体):M=64-384 区 GEMM 1.15-1.65×。

### 路由层(serving,按 M/形状静态窗口,CUDA-graph 安全)
9. **M=1 → W4A16 GEMV**(相位自适应:decode 免激活量化,**精度反升 −0.32 PPL**;带宽 73-79% HBM);
10. **M=2..48 → Marlin 路由**:发现我们的 NVFP4 低段打包布局与 vLLM modelopt **逐字节相同** → 直接复用 vLLM marlin_gemm(fp4)(load 时 15 行 repack),+ 自写 **MXFP8-marlin 高段伴随**(e8m0 marlin + 融合 axpy:g+cast+add 单 kernel)—— decode b4 −25~31%;
11. 窗口全部 **graph-replay 计时标定**(eager 标定会错 15-30% —— 方法学教训,论文可写)。

### 量化/打包生产侧
12. **单 launch 双格式 quant+pack**(qpfm:FP4 段+FP8 段+交错 SF 一个 kernel,84-106× 于 torch 实现;R2 再融合 −1 launch/层);
13. **o_proj 列 gather 折进 pack load**(1.2-2.0×);**GEMM 输出 cast 折进 epilogue**(prefill kernels −12%);
14. SF 交错布局 = vLLM `swizzle_blockscale` **byte-identical**(生态可移植性,集成时白送)。

## 3. TC-平价研究(诚实的 systems 发现,值得单独小节)

问题:手写 kernel 的张量核利用率比 CUTLASS 低 15-25pp,能追平吗?追平有用吗?
- **路径 A(co-design)**:kernel-glue 层拼两个 stock CollectiveMma 共享累加器 → **TC 平价达成**(alu/tc 0.7-1.2≈CUTLASS,ACTIVE 入 68-86% 带)—— 证明平价可达;
- **但延迟反而输 30-42%**:两相在 union smem 下被 phase-barrier **串行化**,丢掉了手写版的**整 K 跨相重叠**;加上 wrapper SF-scatter 开销(后修复后 GEMM-only 其实快 13-18%,端到端仍不赢);
- **结论(论文观点)**:对双格式融合,**跨相重叠 > 单相 issue 密度** —— 手写 kernel 的延迟优势来自 CollectiveMma 结构表达不了的整 K 重叠;我们的发货 kernel 延迟**低于纯格式插值下界**(qkv/down 比"理想两次纯格式 GEMM 之和"还快 8-26%)。
- 根因链条完整可写:ALU 瓶颈 = SF-copy 地址算术(ncu per-pipe 证据)→ 层层假设被测量证伪的全过程(round 8-16)。

## 4. 设计纪律(方法论卖点)

- **bit-exact gate 先于一切性能数**:每根杠杆 torch.equal/cosine=1.0 探针 + levers-off 对照 byte-identical 才准测;
- **graph-replay 计时**(serving 窗口标定的唯一正确方式);
- **诚实负收益记录**(4 项:hadamard-pack 生产融合 0.26-0.97×、1 字节 decode scale 0.92-0.98×、norm-fuse serve +1-4%、co-design 串行化)—— 每项有 ncu/serve 归因;
- 发现并记录 vLLM 地雷:AOT compile-cache key 不含 out-of-tree pass(静默 serve 未融合图)。

## 5. 关键数字(kernel 节的结果句)

- vs fp16 cuBLAS:**2.3-3.1×**(M=8192,GEMM 级);
- vs 同算法朴素 2-launch(torch 原生 FP4+FP8):**1.3×,随 M 升至 1.7×**(= 融合本身的独立价值);
- vs 全部可运行 W4A4 对手 kernel(M=8192):赢 vLLM-NVFP4 1.0-1.53×、SharQ 1.12-1.51×、MicroMix 1.4-1.85×,**ARCQuant 平手**(但其精度低于我们 → Pareto 支配);
- 延迟低于纯格式插值下界(融合不是零成本,是**负成本**:qkv/down −8~−26% vs 理想两次纯 GEMM);
- serving(vLLM,R1-R5 后):decode 对纯 NVFP4 0.85-0.93(大 batch),离分解结构地板 ~4%。

## 6. 建议的 kernel 图(论文)

- **Fig(kernel 架构)**:单 CTA 内两相 mainloop 时间线(NVFP4 段→MXFP8 段→epilogue),共享累加器高亮;旁注 SFASYNC/KPIPE/TMA 的流水;
- **Fig(路由)**:M 轴上的 kernel 选择窗口(GEMV→marlin+hi→splitK→fused 128×128→persistent),每段标实测收益;
- **Tab(kernel 消融)**:杠杆 → GEMM/serve 收益 →(负收益也列);
- 可选 **Fig(TC-平价 vs 延迟)**:co-design 点 vs 手写点的 (TC%, latency) 散点 —— "平价≠更快"的反直觉结论。
