# 单层真·kernel 集成 — 设计 spec

> 2026-10-22。Stage 2 kernel 收口后的首个端到端集成步骤。目标:把一个 Llama-3.2-1B
> decoder layer 的 7 个 Linear 从 fake-quant 换成真·packed 融合 kernel 路径,验证**单层跑通
> + 数值对(cosine)+ prefill 加速**。PPL 回退验证留到后续"整网"步骤。

## 背景与前提

- Stage 2 产物:异格式融合 GEMM(NVFP4 低通道 block-16 + MXFP8 高通道 block-32,共享 FP32
  累加器,单 launch)已到硬件天花板;`config_dispatch.hpp` 可按 shape 自动选最优配置
  (`launch_best_codesign`);`cd_quant_pack`(round-18/24)把激活量化+打包成 packed codes +
  **交错 SF 布局**(collective 需要)。
- 现状:PTQ 前向是 **fake-quant**(反量化成 FP16 算,不调真 kernel),所以测不到真加速。
- kernel 加速集中在**大 M = prefill**(decode M=1 不赢,本步不覆盖 decode)。

## 范围(本步只做这些)

- 取**一个** decoder layer(中间任选一层,如 layer 8),替换其 7 个 Linear:
  q/k/v/o_proj(attn)+ gate/up/down_proj(MLP)。
- 不动 norm / rotation / softmax / RoPE —— 进入 Linear 的激活**已是旋转后**的(ResQ 的
  residual-stream 旋转 + down_proj 的 online Hadamard 在上游完成),本步只接 Linear 这一层。
- 不做整网、不做 PPL、不做 decode。

## 组件

### 1. `MixedQuantLinear`(新模块,替换 `ActQuantWrapper` 的前向)
职责:持有离线打包好的权重 buffer,前向时打包激活、调 dispatcher、返回 fp16 输出。
- **构造**:从一个已量化+旋转的 `nn.Linear`(来自 PTQ pipeline)+ 该层的 ResQ 通道划分
  (K_low / K_high,high_fraction=1/8,reorder 顺序)构造。
- **前向** `y = f(x)`:
  1. `x`(已旋转,fp16,(M,K))→ `cd_quant_pack` → `x_packed`(low:NVFP4 / high:MXFP8)+ `SFA`(交错);
  2. `launch_best_codesign(x_packed, SFA, W_packed, SFB, s_x, s_w, M,N,K,Kh)` → `y`(fp16,(M,N));
  3. 返回 `y`。
- bias:若原 Linear 有 bias,epilogue 后加(或折进 epilogue),与 fake-quant 口径一致。

### 2. 权重打包器 `pack_weight_mixed`(新,主要新工作)
- 输入:量化+旋转后的权重 `W`(N,K)、通道划分(K_low/K_high)、per-group/per-channel scale。
- 输出:`W_packed`(low 段 NVFP4 2×FP4/byte + high 段 MXFP8)、`SFB`(**交错 atom 布局**,
  复用 round-24 `prepack_sf_*_b` 的 once-at-load 路径)。
- 约束:[low|high] 段必须对齐两种格式 block(NVFP4 16 / MXFP8 32)—— reorder-before-quant
  的离线版;SFB 布局必须与 `launch_best_codesign` 消费的完全一致(bit-exact 契约)。

### 3. 接线 `build_single_layer_realkernel`
- 从 PTQ pipeline 产出的量化模型里取目标 layer,把 7 个 `ActQuantWrapper` 换成
  `MixedQuantLinear`(各自 pack 权重)。其余层不动。

### 4. 验证/测量 harness(新脚本)
- **数值**:对同一批(已旋转)激活,比较 `MixedQuantLinear` 单层输出 vs fake-quant 单层输出
  的 cosine(目标 ≈0.999,两边同为 NVFP4+MXFP8 数值);附带 vs FP16 看精度 drift。
- **加速**:该 layer 的 prefill 延迟(大 M = seq×batch,如 M=2048/4096/8192)vs 对应 FP16
  cuBLAS 的 7 个 Linear,以及 vs fake-quant;**拆出 pack vs GEMM 占比**(诚实计入 per-forward pack)。

## 成功判据(promotion gate)

1. 7 个 Linear 全部走真 kernel,单层前向跑通,`MixedQuantLinear` vs fake-quant cosine ≥ 0.999;
2. 权重/激活打包的 SFB/SFA 布局与 kernel 消费 bit-exact(torch.equal 探针);
3. 测到该层 prefill 延迟数(真 kernel vs FP16 vs fake-quant),pack/GEMM 拆分清楚;
4. 不改 PROTECTED kernel 文件与 mainloop 逻辑;`config_dispatch.hpp` 原样调用。

## 非目标 / 风险

- **非目标**:整网、PPL、decode、KV cache、与 project-resq 解耦。
- **风险**:权重打包器要和 kernel 的 SFB 交错契约完全对齐(最易错,靠 torch.equal 探针兜);
  激活 pack 是真实 per-forward 成本(本步如实报告,不藏);某些 Linear 的 K_high 很小
  (q/k/v 的 256)要确认 dispatcher 对这些 shape 也选对配置。

## 下一步

本 spec 批准后 → writing-plans 转成逐任务 plan(TDD:先 cosine 探针、再 pack 器、再模块、再接线、再测速)。
