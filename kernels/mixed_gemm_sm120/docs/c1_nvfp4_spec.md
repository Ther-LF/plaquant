# Candidate 1 实现 spec — SM120 NVFP4 block-scaled GEMM torch 扩展(C1.1 keystone)

> C1.1 是 Candidate 1 的基石:一旦 NVFP4 的 SF 布局 + 打包搞对,MXFP8 和 2-launch 组合都是增量。先只做 NVFP4 单精度 GEMM 扩展 + oracle 验证。

## 目标

在远程 Pro 6000 容器里,构建一个 PyTorch 可调用的 **NVFP4 block-scaled GEMM**:
```
nvfp4_gemm(A_packed, SFA, B_packed, SFB, M, N, K) -> D   # D: (M,N) bf16 或 fp32
```
包装 CUTLASS 示例 79b 的 kernel 配置,正确性对一个 PyTorch 参照(同样的 block-scaled 反量化-matmul)验证 **cosine ≥ 0.999**。

## 数据契约(已从 ex79b 实证)

- **A**:NVFP4(`cutlass::nv_float4_t<float_e2m1_t>`),RowMajor (M,K)。每 2 个 FP4 打包进 1 字节 → `A_packed` uint8 (M, K/2)。AlignmentA=32 元素。
- **B**:NVFP4,ColumnMajor (N,K)(= 我们权重存 (N,K) 的约定)→ `B_packed` uint8 (N, K/2)。
- **SF(scale factor)**:`InputSFVectorSize = 16`(每 16 元素一个块 scale,= 我们 `fake_quantize_nvfp4` block_size)。元素类型 `ElementA::ScaleFactorType`(FP8 e4m3 系)。
  - **交错布局用 CUTLASS 工具,不要手推**:
    ```cpp
    using Cfg = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
    auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
    // SF buffer 大小 = size(filter_zeros(layout_SFA))
    // 填充:make_tensor(sfa_ptr, layout_SFA) 按逻辑 [m, k_block] 索引,CUTLASS 自动交错
    ```
  - Python 侧只产出**逻辑的** SFA (M, K/16) 和 SFB (N, K/16)(e4m3 字节);wrapper 内用上面的 layout 把逻辑 scale 搬进交错 buffer。

## 构建

- 编译 flags(C0 已验证):`nvcc -arch=sm_120a -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -I <cutlass>/include -I <cutlass>/tools/util/include -I <cutlass>/examples/common`
- kernel 配置直接抄 ex79b(`79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu` 的 CollectiveBuilder/Gemm 类型),改成接收外部指针(我们的 A_packed/B_packed/SFA/SFB)而非随机生成。
- PyTorch 扩展:setup.py(torch.utils.cpp_extension,传上述 flags),或 load() JIT。

## 验证(oracle)

Python 参照:对同一组逻辑输入做 block-scaled 反量化 matmul——
```
A_deq[m,k] = fp4_to_float(A_code[m,k]) * SFA[m, k//16]
B_deq[n,k] = fp4_to_float(B_code[n,k]) * SFB[n, k//16]
D_ref = A_deq @ B_deq.T          # (M,N)
```
造一组随机但合法的 NVFP4 输入(FP4 码 + e4m3 scale),喂 `nvfp4_gemm`,比 `cosine(D_kernel, D_ref) ≥ 0.999`。可参考 `promix/quantize/quant_utils.py` 的 `fake_quantize_nvfp4`(它已实现 FP4 格点和块 scale 逻辑,拿来造数据/参照)。

## 远程执行

- `scp` 文件到 `/home/luofan/plaquant/kernels/mixed_gemm_sm120/`,在容器 `luofan-plaquant` 内编译/跑。
- 挑空闲 GPU:`nvidia-smi` 选 memory.used 低的(别人随时占 3-7),`-e CUDA_VISIBLE_DEVICES=<free> -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- CUTLASS 在 `/home/luofan/plaquant/third_party/cutlass`。

## gate

cosine ≥ 0.999 vs oracle,在至少两个 shape(含 1024³ 和一个 LLM shape 如 M=2048,N=2048,K=1792)。产出:`nvfp4_gemm` 扩展 + 验证脚本 + 结果记入 candidates.jsonl。
