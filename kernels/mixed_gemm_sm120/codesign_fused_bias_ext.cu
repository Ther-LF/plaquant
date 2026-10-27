// codesign_fused_bias_ext.cu — round-26 lever A: GEMM epilogue bias + output-cast fold.
//
// Non-protected epilogue-config copy of codesign_fused_ext.cu. It changes ONLY
// the epilogue:
//   * ElementD/ElementC = fp16 (so the fused GEMM stores fp16 directly and the
//     112x per-Linear `D.to(fp16)` kernel disappears), and
//   * the fusion operation becomes codesign::LinCombBiasBFRoundHalf, which
//     performs EXACTLY the reference arithmetic
//          half( float(bf16(acc_fp32)) + float(bias[n]) )
//     (see codesign_bias_epilogue.hpp for why the bf16 round must come first —
//     the naive fp32->fp16 store was the documented numerics failure of
//     codesign_fused_fp16_ext.cu) and adds the per-N bias in the epilogue, so
//     the 112x `D + bias` kernel disappears too.
//
// The PROTECTED mainloop (codesign_fused_kernel.hpp / codesign_sm120_bs_mma.hpp)
// is included verbatim and every GEMM/dequant arithmetic step is unchanged.
//
// Entry point: fused_gemm_prepacked_bias(A..., bias) -> D fp16 [M,N]
//   bias may be an empty/undefined tensor for checkpoints without a bias; the
//   no-bias case then uses a zero bias buffer (exact for every value except the
//   +-0.0 sign, which the decoder's fp32 accumulation does not observe).
//
// Built as its own torch extension module; the validated bf16 path
// (codesign_fused_ext) stays untouched and remains the fallback.
#define CD_OUT_FP16 1
#define CD_EPI_BIAS 1
// The included body's pybind block is suppressed; this module registers ONLY
// the round-26 entry point (the bf16 path keeps its own module/so).
#define CD_NO_PYBIND 1
#include "codesign_fused_ext.cu"

// ---------------------------------------------------------------------------
// Bias-fused prepacked GEMM. Reuses run_dual_pre's machinery with a different
// epilogue Arguments (bias pointer) — the mainloop args, scheduler, workspace
// handling and stream/capture guards are identical to the bf16 path.
// ---------------------------------------------------------------------------
torch::Tensor cd_fused_gemm_prepacked_bias(torch::Tensor A_lo, torch::Tensor physSFA_lo,
                                           torch::Tensor B_lo, torch::Tensor physSFB_lo,
                                           torch::Tensor A_hi, torch::Tensor physSFA_hi,
                                           torch::Tensor B_hi, torch::Tensor physSFB_hi,
                                           c10::optional<torch::Tensor> bias_opt,
                                           int64_t M, int64_t N, int64_t K_lo, int64_t K_hi,
                                           double alpha = 1.0) {
  TORCH_CHECK(K_lo % 16 == 0 && K_hi % 32 == 0, "K_lo%16, K_hi%32");
  A_lo=A_lo.contiguous(); B_lo=B_lo.contiguous(); A_hi=A_hi.contiguous(); B_hi=B_hi.contiguous();
  physSFA_lo=physSFA_lo.contiguous(); physSFB_lo=physSFB_lo.contiguous();
  physSFA_hi=physSFA_hi.contiguous(); physSFB_hi=physSFB_hi.contiguous();
  auto dev = A_lo.device();
  auto stream = at::cuda::getCurrentCUDAStream(dev.index());
  int Mi=(int)M, Ni=(int)N, Klo=(int)K_lo, Khi=(int)K_hi;

  torch::Tensor bias_t;
  torch::Tensor bias = bias_opt.has_value() ? *bias_opt : torch::Tensor();
  if (bias.defined() && bias.numel() > 0) {
    TORCH_CHECK(bias.scalar_type() == torch::kHalf, "bias must be fp16");
    TORCH_CHECK(bias.numel() == N, "bias numel ", bias.numel(), " != N ", N);
    bias_t = bias.contiguous();
  } else {
    bias_t = torch::zeros({N}, torch::TensorOptions().dtype(torch::kHalf).device(dev));
  }

  auto D = torch::empty({Mi, Ni}, torch::TensorOptions().dtype(torch::kHalf).device(dev));
  auto nv_args = make_mainloop_args_pre<ForkNV, BSConfigNV>(A_lo, physSFA_lo, B_lo, physSFB_lo, Mi, Ni, Klo);
  auto mx_args = make_mainloop_args_pre<ForkMX, BSConfigMX>(A_hi, physSFA_hi, B_hi, physSFB_hi, Mi, Ni, Khi);

  typename GemmDual::Arguments arguments;
  arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  arguments.problem_shape = {Mi, Ni, Klo, 1};
  arguments.mainloop_nv = nv_args;
  arguments.mainloop_mx = mx_args;
  arguments.K_high = Khi;
  {
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {Mi, Ni, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {Mi, Ni, 1});
    using ElementC = typename Epilogue::ElementC;
    using ElementD = typename Epilogue::ElementD;
    typename Epilogue::Arguments epi_args;
    epi_args.thread.alpha = (float)alpha;
    epi_args.thread.beta = 0.0f;
    epi_args.thread.ptr_bias = reinterpret_cast<cutlass::half_t const*>(bias_t.data_ptr());
    // per-N bias: stride (0, 1) — same as the reference Sm90LinCombPerColBias
    epi_args.thread.dBias = {_0{}, _1{}, 0};
    epi_args.ptr_C = reinterpret_cast<ElementC const*>(D.data_ptr());
    epi_args.dC = sC;
    epi_args.ptr_D = reinterpret_cast<ElementD*>(D.data_ptr());
    epi_args.dD = sD;
    arguments.epilogue = epi_args;
  }

  static GemmDual gemm;
  static torch::Tensor ws_cache;
  static bool checked = false;
  run_with_cached_ws(gemm, arguments, dev, stream, ws_cache, checked, "fused_bias");
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_gemm_prepacked_bias", &cd_fused_gemm_prepacked_bias,
        "dual-format fused GEMM, fp16 out, bias + bf16-round + cast folded into the epilogue; optional per-tensor epilogue alpha (two-level weight global)",
        pybind11::arg("A_lo"), pybind11::arg("physSFA_lo"), pybind11::arg("B_lo"), pybind11::arg("physSFB_lo"),
        pybind11::arg("A_hi"), pybind11::arg("physSFA_hi"), pybind11::arg("B_hi"), pybind11::arg("physSFB_hi"),
        pybind11::arg("bias"), pybind11::arg("M"), pybind11::arg("N"), pybind11::arg("K_lo"), pybind11::arg("K_hi"),
        pybind11::arg("alpha") = 1.0);
  m.def("compat_report", &cd_compat_report, "build/smem report");
}
