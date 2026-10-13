# SM120 融合 GEMM — Executable Plan (KDA)

> 由 `draft.md` 转化。三个开放点已拍板:(1) 通用双 phase 分段版优先(down_proj 先带 [低|高] 段,"纯 FP4 单 phase" 作后续 ablation);(2) KDA skills 已装(ncu-report-skill + KernelWiki);(3) 融合用手写 CuTe mainloop。

## 候选执行顺序与每步的 promotion gate

### Candidate 0 — 工具链 + 单精度 + CLayout 实证
**目的**:零风险确认 SM120 FP 工具链可用、两个 atom 累加器可共享。
- [ ] C0.1 在远程容器建 `kernels/mixed_gemm_sm120/` 编译脚手架(setup.py,`-arch=sm_120a`,-I CUTLASS include)。
- [ ] C0.2 编译运行 CUTLASS 示例 79(SM120 NVFP4 GEMM),确认 nvcc 13.0 + sm_120a 通。
- [ ] C0.3 编译运行示例 87(SM120 MXFP8 blockwise),确认通。
- [ ] C0.4 读 `mma_traits_sm120.hpp`,实证 `SM120_16x8x64_TN_VS`(NVFP4)与 `SM120_16x8x32_TN_VS`(MXFP8)的累加器 CLayout 是否同为 `SM80_16x8_Row`。
- [ ] C0.5 PyTorch oracle:`fake_quantize_nvfp4/mxfp8` 造 (A,B,scale),各自单精度 kernel 对 oracle 比 cosine。
- **gate**:两个单精度 kernel cosine≥0.9999 + CLayout 结论明确(共享可行/不可行→决定 Candidate 2 走寄存器共享还是 EVT-workspace 兜底)。

### Candidate 1 — 2-launch 基线
- [ ] C1.1 用 C0 的两个单精度 kernel + epilogue add 组 2-launch 路径。
- [ ] C1.2 对 oracle 验证(应 cosine≈1,混合精度分段相加)。
- [ ] C1.3 benchmark:逐关键 shape 测 2-launch 延迟 + FP16 cuBLAS 延迟 → 写 `benchmark.csv`。
- **gate**:2-launch cosine≥0.9999,基线延迟表成形。

### Candidate 2 — 融合单 launch(主交付物)
- [ ] C2.1 手写 CuTe mainloop:Phase1 NVFP4(K_low/64)→ SMEM union → Phase2 MXFP8(K_high/32),同 FP32[4] 寄存器累加。FP4 `<<2` 位移照示例 79。
- [ ] C2.2 epilogue:`out = acc·s_x·s_w → FP16`。
- [ ] C2.3 对 oracle 验正确 + 对 Candidate 1 验加速。
- **gate(promotion)**:全 shape cosine≥0.9999 且 融合延迟 < 2-launch。

### Candidate 3+ — 优化(达 promotion 后)
tile(64×64/64×128/128×64/128×128)、stage(3/4/5)、cross-phase prefetch、per-shape config、down_proj 纯 FP4 单 phase ablation。每步 ncu-report-skill 取证据,写 `candidates.jsonl`/`profile/`。

## 当前动作:Candidate 0,从 C0.2(编译示例 79)开始——最高信息量、最低风险的工具链探针。

---

## Candidate 0 结果(2026-10-13,已完成)

| 步 | 结果 |
|----|------|
| C0.1 build flags | `-arch=sm_120a -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -I include -I tools/util/include -I examples/common` |
| C0.2 NVFP4 (ex79b) | ✅ Disposition Passed,1024³ 0.107ms,20.1 TFLOPS |
| C0.3 MXFP8 (ex79c) | ✅ Disposition Passed,1024³ 0.0715ms,30.0 TFLOPS |
| C0.4 CLayout 实证 | ✅ 两 atom 累加器 CLayout 均 = `SM80_16x8_Row`,FP32[4]/thread 一致 → **寄存器内共享累加器可行**(Candidate 2 不必退 EVT-workspace) |

**风险消除**:工具链(nvcc 13.0 + sm_120a + CUTLASS NVFP4/MXFP8 atom)实证可编译/运行/数值正确;共享累加器前提成立;FP4 `<<2` 位移 CUTLASS 自带 `fp4_shift_A/B`。

**C0.5**(对我方 PyTorch oracle 比 cosine)并入 Candidate 1/2——CUTLASS 示例已自带 "Disposition Passed" 的参照自验;对我方 oracle 的比对要在用我方数据打包的自建 kernel 上做才有意义。

**下一步:Candidate 1 — 2-launch 基线**(两个单精度 GEMM + epilogue add,作正确性锚点 + 性能基线)。
