// codesign_fused_fp16_ext.cu — prefill-fusion round: fp16-output epilogue copy.
//
// This is the "NEW non-protected epilogue-config copy" required by the
// prefill-fusion step (target #3: fold the 112x per-Linear bf16->fp16 cast
// `D.to(fp16)` into the fused-GEMM epilogue). It changes ONLY the epilogue
// output element type (bf16 -> fp16) by defining CD_OUT_FP16 and then
// #including the full codesign_fused_ext.cu body verbatim — so the PROTECTED
// mainloop (codesign_fused_kernel.hpp / codesign_sm120_bs_mma.hpp) and all
// GEMM math are byte-for-byte unchanged; the GEMM dequant accumulator (fp32)
// is simply rounded to fp16 at store time instead of bf16.
//
// Built as a separate torch extension module (its own .so + pybind name) so
// the validated bf16 path (codesign_fused_ext) stays available untouched; the
// perf/PPL harnesses load this module and pass it as `cd` to wire_real_kernel,
// which makes MixedQuantLinear's `D.to(fp16)` a no-op (D is already fp16).
#define CD_OUT_FP16 1
#include "codesign_fused_ext.cu"
