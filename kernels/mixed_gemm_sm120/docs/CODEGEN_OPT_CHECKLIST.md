# 低层 codegen 优化 checklist(每轮 kernel agent 必过一遍)

CUTLASS 之所以快,一半靠算法/流水(TC 打满),另一半靠这些贯穿始终的"编译期 + 指令级"技巧。
它们和 TC-no-dip 互补:降指令数 / 降寄存器压力 / 让编译器全展开 → 直接填 TC 空隙。
每条都给了「做什么 / CUTLASS 在哪学 / 怎么验」。动手前先在 CUTLASS 源码里看对应写法。

## A. 模板化 / 编译期特化(compile-time specialization)
- **所有循环界、tile shape、stage 数、K-tile 数做成 template/constexpr**,让编译器完全展开、消掉动态分支和边界判断。
  - CUTLASS:`cute::Int<N>`、`Shape<_64,_64,_128>`、`cute::for_each` + 静态 range、`cutlass::gemm::collective` 里全是 constexpr tile。
  - 我方:kmul / stages / tile 已是宏/模板 —— 检查 K-loop 上界是否 constexpr、是否 `#pragma unroll`。
- **`if constexpr` 剪死分支**:r=0 时整个 Phase-2(MXFP8)应在编译期消失,不是运行时 `if`。按「是否存在高精度段」做模板特化,生成两个 kernel 变体。
- **FastDivmod / 预算倒数**:别在 kernel 里做运行时整数除/取模(rasterization、SF 索引)。CUTLASS `cutlass::FastDivmod`(乘 + 移位替除法)。
- **`__launch_bounds__(maxThreads, minBlocks)`**:显式约束寄存器/占用率,防 spill。CUTLASS kernel 都带。

## B. 向量化(vectorized load / store / copy)
- **128-bit 访存**:global/SMEM 搬运用 16B 粒度(`uint4`/`int4`/`float4`/`cutlass::AlignedArray<T,N>`),cp.async 用 `.cg` 16B。我方 SF 之前的 1-byte scatter 就是反例,已改 uint64/uint16 合并。
  - CUTLASS:`cute::copy` 自动选向量化 atom;`cutlass::Array<T,N>` + `__align__(16)`;`AlignedBuffer`。
  - 验:ncu `l1tex__...per_request`、`long_scoreboard` stall(我方从 4.42→1.45 就是合并访存的功劳)。
- **对齐**:A/B/SMEM 指针 16B 对齐(AlignmentA=32 元素 for NVFP4),否则向量化访存降级。
- **ldmatrix / stmatrix**:SMEM↔寄存器用 `ldmatrix`(我方 B 侧已用 SM75_U32x2_LDSM_N);epilogue 回写考虑 `stmatrix` 或至少向量化 store。
- **向量化 epilogue**:acc→FP16 回写按 128-bit 批量,别逐元素。

## C. 指针 / 缓存 hint
- **`__restrict__`** 所有只读指针 → 编译器放开 alias 假设、复用寄存器。
- **只读路径走 `__ldg` / `ld.global.nc`**:权重 B 是只读,走 read-only cache。CUTLASS 用 `cute::Tensor` + global load atom。

## D. 展开 / 寄存器调度
- **`#pragma unroll`** 内层 K-block 循环;配合 A 的 constexpr 界。
- **最小化活跃区 / 复用寄存器** 防 spilling(双 phase 共享 acc 寄存器压力大,sglang PR-8127 专门修 NVFP4 spilling —— 去查)。
- **寄存器 K-block 乒乓**:ldmatrix(k+1) 与 MMA(k) 交织(本轮主攻,LeetCUDA 有确切写法)。

## E. SMEM
- **swizzle 消 bank conflict**(我方 SW64 已把 conflict 18.9-37.8M→0);swizzle 模式编译期算好。
- **union 复用 SMEM**(双 phase 不同时活);check SMEM 预算 vs 占用率权衡。

## 用法
1. 每轮动手前,挑本轮相关的几条,先去 CUTLASS 源码 / KernelWiki / DeepSeek 看别人怎么写(见 [[RESEARCH_TOOLS]])。
2. 改完在 report 里对每条写「做了/没做/为什么」+ ncu 前后对比(指令数、stall、寄存器数、spill)。
3. 别只堆技巧:每条都要有 ncu 证据证明它真降了指令/stall/寄存器,没证据的回退。
