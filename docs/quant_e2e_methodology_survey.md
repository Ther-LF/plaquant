# 量化工作的 e2e 加速方法论调查 — 我们为什么"GEMM 快但 e2e 不涨"

> 2026-10-24。目的:搞清别人报的几倍是什么口径,什么条件下成立,我们缺什么。
> 来源:AWQ/ChatGPTQ-Marlin/ExLlama、QuaRot、SpinQuant、Atom、QServe、LLM-MQ、SmoothQuant、
> TensorRT-LLM NVFP4、llm-compressor/vLLM NVFP4(含 RTX PRO 6000 SM120 实测)。

## 1. 各家数字与口径

| 工作 | 报告加速 + 设置 | **对比基线(最关键)** | 计时路径里量化了什么 | 部署路径里的融合 | 显存 |
|---|---|---|---|---|---|
| **AWQ / TinyChat** | decode 2.7–3.9×,7–13B,**bs=1**,RTX 4090 | **HF FP16 eager**(其自测:仅 kernel fusion 就 +19%) | 仅权重 W4A16;激活 FP16 | on-the-fly dequant GEMM;其它层也手写优化 | 4× |
| **GPTQ + 自写 GEMV** | 3.24×(A100)/4.53×(2×A6000),OPT-175B 3-bit,bs=1 | **FP16 跑在 5×A100 / 8×A6000**(跨卡数;单卡基线不是调优过的 cuBLAS) | 仅权重 3-bit;embedding+输出层保持 FP16 | dequant-in-loop | ~5.5× |
| **ExLlama/Marlin** | ExLlamaV2 205 tok/s(4090,无 FP16 列);Marlin "近理想 4×" | **kernel 级**(matmul 对 matmul),非 e2e | 仅权重 W4A16 | repack + L2-resident | 3.87× |
| **QuaRot** | prefill **1.97–2.16×**(7B)/最多 3.33×(70B),bs≤64,seq 2048,3090;**无 decode 加速报告** | **FP16 CUTLASS,但只测单个 transformer block**(整模型放不下) | 权重+激活 4-bit+KV4;Q 保持 FP16 | **未融合**:"separate quantize kernel → CUTLASS INT4 GEMM";作者自认可改进 | 3.63–3.89× |
| **SpinQuant** | 主 e2e 号:**~3× W4A8 vs W16A16 在 MacBook M1 Pro CPU**;H100 表只有 TTFT/TPOT 没有加速比 | **PyTorch FP16 CPU**;H100 上没给加速号 | 权重+激活(评估含 KV);GPU 吞吐路径其实是 W8A8 | Hadamard 独立 TC kernel + FBGEMM epilogue |
| **Atom** | **2.53× vs FP16**,7–13B,4090,**batch 2–256**(大 batch 点为**模拟**) | 同 harness 的 FP16(Punica,较公平) | 权重+激活 4-bit(**128 outlier 通道保持 INT8**)+KV4 | **有**:量化/reorder 融进已有算子;动态 per-token 激活量化 | KV 4× |
| **QServe** | **1.2–2.4×(A100)/1.47–3.47×(L40S)**,7B–72B,ISL/OSL 1024/512 | **TRT-LLM FP16、W8A8、W4A16** ——最强基线;还指出 Atom/QuaRot 在 A100 上比 TRT-LLM W16A16 还慢 | W4 权重、**A8 激活**、KV4 | **多**:激活量化融进**前一个 LN**(QKV/FFN1)或**前一个激活 kernel**(FFN2);paged attention | 3.6× |
| **SmoothQuant** | prefill **1.51×(vs PyTorch)**、**1.56×(vs FasterTransformer)**;decode 1.19–1.42× | 两种基线分开报;**诚实的是 FT 那个** | W8A8;KV 与 attention 未量化 | **有**:激活量化融进前一个算子 | ~2× |
| **TRT-LLM NVFP4** | Llama-3.3-70B FP4 on B200: 10,994 tok/s → 24,667(TP4);"比 DGX H200 >3×" | **DGX H200(不同代 GPU,带宽 2.7×)**;不是同机 FP16 | W4A4 + **FP8 KV** + **MTP 投机解码** | **有**:fused norm+quant、CUDA graphs、in-flight batching | ~4× 权重 / 2× KV |
| **vLLM NVFP4 (RTX PRO 6000, SM120!)** | Qwen3-32B:**~2× BF16 吞吐**,TTFT 150 vs 340ms @ c=1 | **同 stack BF16** —— 最公平的 NVFP4 对照;2× ≈ FP4:BF16 张量核比,即 **compute-bound** | W4A4(全部 linear),FP8/BF16 KV | **有**:`fuse_norm_quant`、CUDA graphs、paged attention | ~4× |

## 2. 四个最影响"几倍"的方法学差异(相对我们的设置)

1. **基线的强弱决定一切。** AWQ 比的是 HF eager(4090 上 Llama-2-7B 只有 52 tok/s,调优 FP16 应有 2–3×);GPTQ 比 5–8 张卡;QuaRot 只测单 block + 自家 CUTLASS。真正跟**同 stack 强 FP16** 比的只有 **QServe(1.2–2.4×)** 和 **vLLM-NVFP4 vs BF16(~2×)**——它们报的数最小、最可信。**公平口径下,合理预期是 1.2–2×,不是 3–4×。**
2. **他们赢在带宽,我们在 1B bs=1 赢不到。** AWQ/GPTQ/Marlin/ExLlama/LLM-MQ 都是 **W4A16**,7B–70B bs=1 decode 时权重流 14–140GB > HBM 带宽,4× 压缩≈直接 4× 加速。**Llama-1B 权重才 ~0.5GB,0.3ms 就读完,是 launch/latency-bound**;4-bit GEMM 反而多解包开销。**1B bs=1 的 3–4× 不成立;8B 才开始见到。**
3. **他们量化的路径比我们长。** QuaRot/Atom/QServe 都量化 **KV cache + attention**,Atom/QServe 还有 QKV/FFN 融合 + continuous batching;NVFP4 部署再加 CUDA graph/投机解码。**我们只量化 7 个 Linear,attention/KV/lm_head/embedding 全是 FP16** → bs=1 decode 下 Amdahl 上限 ~1.3–1.7×。
4. **不少 headline 不是 e2e。** QuaRot 的 3.33× 是单 block prefill;Atom 大 batch 是模拟;Marlin 4× 是 kernel 级。真·全栈测(QServe、vLLM)就回落到 1.2–2.4×。

## 3. 对我们的直接结论

- **要对标的是 vLLM 在 RTX PRO 6000(SM120)上 NVFP4 ~2× BF16** —— 同硬件、同口径,而且那是 compute-bound 下的 2×,说明**我们的 GEMM 侧还有空间**(我们目前 e2e 1.08× 主要被非量化部分和量化税拖住,不是 GEMM 本身)。
- **补齐四件事才有资格同台比**:
  1. **激活量化融进生产者**(LN/epilogue)——QServe 的做法,消掉我们 pack 的独立 launch + gmem 往返;
  2. **KV cache 量化 + attention 量化**——decode 的 Amdahl 上限由它决定;
  3. **lm_head/embedding 纳入**(W8 已验证安全);
  4. **同 stack 基线 + CUDA graph + 报告 TTFT/TPOT/prefill/decode 分列**。
- **1B 只是个 sanity target;要报有竞争力的数,必须上 8B**(权重带宽、Linear 占比都进入有利区)。
- 注意 SM120 的坑:vLLM 的 NVFP4 MoE backend 在 SM120 部分不可用(issue #36094);独立 SM120 研究显示单流 decode 上 **AWQ INT4 常略胜 NVFP4**——所以我们对标 vLLM BF16 的 2×,不要对标 NVIDIA 的 H200 3×。
