// codesign_bias_epilogue.hpp — GEMM-out bias + dtype-cast epilogue fusion (round-26, lever A).
//
// Problem. MixedQuantLinear returned
//     D = gemm_bf16(A, B)            # fused GEMM writes a bf16 [M,N] tile to HBM
//     D = D.to(fp16)                 # 112 separate elementwise kernels per forward
//     D = D + bias.to(fp16)          # 112 more
// i.e. 112 launches + ~8 bytes/element of extra HBM traffic (8.8 ms @ M=8192,
// docs/prefill_stage_profile_report.md §2). The prefill-fusion round already tried
// to fold the CAST alone by emitting fp16 from the epilogue
// (codesign_fused_fp16_ext.cu) and it FAILED the numerics gate
// (logit cos 0.983, PPL +0.0645): rounding the fp32 accumulator straight to fp16
// is NOT the reference arithmetic.
//
// Reference arithmetic (what the validated model is accurate at):
//   with bias:     ref = a_half( bf16(acc_f32) + bias_fp16 )
//   without bias:  ref = a_half( bf16(acc_f32) )        (exact: every bf16 is fp16)
// PyTorch evaluates the half+half add in fp32 (opmath_t = float) and rounds once,
// so the exact chain is
//     half( float(bf16(acc)) + float(bias) )
// i.e. a bf16 round FIRST (the store the validated path performs), then the bias
// add at fp32 precision, then one fp16 round. Verified on-device: this reproduces
// the reference bit-for-bit on 100k random (acc, bias) pairs, while rounding acc
// directly to fp16 mismatches 81% of them.
//
// Implementation: a *leaf* fusion callback (the protocol CUTLASS's
// Sm90VisitorImpl leaves use). Its children are Sm90AccFetch and the per-N bias
// broadcast; visit() receives both fragments and performs the whole chain, so
// the register fragment never makes a second compute-node round trip.
//
// Used by codesign_fused_bias_ext.cu (a non-protected epilogue-config copy); the
// protected mainloop (codesign_fused_kernel.hpp) and the validated bf16 path are
// untouched.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

// D = half( float(bf16(acc)) + float(bias[n]) )
template <class ElementOutput, class ElementBias, int AlignmentBias>
struct LinCombBiasBFRoundHalf : FusionOperation {
  using ElementOutputType = ElementOutput;
  using ElementBiasType = ElementBias;
  static constexpr int AlignmentBiasValue = AlignmentBias;

  static constexpr bool IsSourceSupported = false;
  static constexpr bool IsPerRowBiasSupported = false;
  static constexpr bool IsPerColBiasSupported = true;
  static constexpr bool IsPerColBiasRoundToBf16 = true;
  static constexpr bool IsDePerColBiasSupported = false;
  static constexpr bool IsEltActSupported = false;
  static constexpr bool IsDeEltActSupported = false;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr bool IsAuxInSupported = false;
  static constexpr bool IsAbsMaxSupported = false;
  static constexpr bool IsBlockScaleSupported = false;

  struct Arguments {
    ElementBias const* ptr_bias = nullptr;
    using StrideBias = Stride<_0, _1, int64_t>;
    StrideBias dBias = {_0{}, _1{}, 0};
  };

  template <class ProblemShape>
  static constexpr Arguments
  to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
};

// ---------------------------------------------------------------------------
// Leaf callback: children deliver (acc, bias[n]); the visit computes and the
// epilogue store writes fp16 with the reference's bf16-then-add-then-round chain.
// ---------------------------------------------------------------------------
template <class ElementOutput, class ElementBias, int AlignmentBias, class CtaTileShapeMNK>
struct BiasBFRoundHalfCallbacks : Sm90VisitorImpl<> {
  struct SharedStorage { };
  struct Arguments {
    float alpha = 1.0f;
  };
  struct Params {
    float alpha = 1.0f;
  };
  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return {args.alpha};
  }

  // Custom Arguments/Params replace the empty-base ones, so the workspace hooks
  // the EVT framework calls must be provided here too (this node owns no device
  // workspace -> zero bytes, nothing to initialize).
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&,
                                              void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  float alpha_ = 1.0f;
  CUTLASS_HOST_DEVICE
  BiasBFRoundHalfCallbacks() {}
  CUTLASS_HOST_DEVICE
  BiasBFRoundHalfCallbacks(Params const& p, SharedStorage const&) : alpha_(p.alpha) {}

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    float alpha;
    CUTLASS_DEVICE ConsumerStoreCallbacks(float a) : alpha(a) {}

    template <typename ElementAccumulator, typename... ElementInputs, int FragmentSize>
    CUTLASS_DEVICE Array<ElementOutput, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int, int, int,
          Array<ElementInputs, FragmentSize> const&... frg_inputs) {
      // Tree protocol: inputs are (frg_acc re-passed, bias, ...). The bias is the
      // LAST child (the accumulator is child 0 and is re-passed first).
      auto const children = cute::make_tuple(frg_inputs...);
      constexpr int kChildren = sizeof...(ElementInputs);
      Array<ElementOutput, FragmentSize> out;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        float b = 0.0f;
        if constexpr (kChildren > 1) {
          b = float(cute::get<kChildren - 1>(children)[i]);
        }
        // round-6 two-level weight global: the per-tensor FP32 scalar `alpha`
        // (a power of 2 carried from the weight packer) multiplies the FP32
        // accumulator BEFORE the reference bf16 round, exactly reproducing the
        // two-level fake-quant dequant g*(A@B). alpha=1 is the stock path.
        out[i] = ElementOutput(
            static_cast<float>(static_cast<cutlass::bfloat16_t>(alpha * float(frg_acc[i]))) + b);
      }
      return out;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const&) {
    return ConsumerStoreCallbacks{alpha_};
  }
};

// Fusion callbacks selected by the epilogue builder for our operation tag.
template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementBias,
  int AlignmentBias,
  class CtaTileShapeMNK,
  class EpilogueTile
>
struct FusionCallbacks<
    epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
    LinCombBiasBFRoundHalf<ElementOutput, ElementBias, AlignmentBias>,
    CtaTileShapeMNK,
    EpilogueTile>
  : Sm90EVT<
      BiasBFRoundHalfCallbacks<ElementOutput, ElementBias, AlignmentBias, CtaTileShapeMNK>,
      Sm90AccFetch,
      // EXACTLY the reference Sm90LinCombPerColBias bias source: a per-N
      // broadcast with the N stride (CUTLASS's "per-column bias").
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, float,
                       Stride<_0, _1, int64_t>, AlignmentBias>> {

  using Operation = LinCombBiasBFRoundHalf<ElementOutput, ElementBias, AlignmentBias>;
  using Impl = Sm90EVT<
      BiasBFRoundHalfCallbacks<ElementOutput, ElementBias, AlignmentBias, CtaTileShapeMNK>,
      Sm90AccFetch,
      Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, float,
                       Stride<_0, _1, int64_t>, AlignmentBias>>;

  struct Arguments {
    float alpha = 1.0f;          // round-6: per-tensor two-level weight global (pow2)
    float beta = 0.0f;
    ElementBias const* ptr_bias = nullptr;
    using StrideBias = Stride<_0, _1, int64_t>;
    StrideBias dBias = {_0{}, _1{}, 0};
    operator typename Impl::Arguments() const {
      // Sm90VisitorImpl's tuple is (children..., node): acc, bias, node.
      return {
        {},                                       // Sm90AccFetch
        {ptr_bias, ElementBias(0), dBias},        // Sm90RowBroadcast (per-N)
        {alpha}                                   // BiasBFRoundHalfCallbacks (alpha)
      };
    }
  };

  using Impl::Impl;
};

// NOTE on bias-less checkpoints. Llama-3.2-1B's 7 target Linears carry no bias
// (checked), and Llama checkpoints never do. The host entry still accepts an
// empty bias and substitutes a zero buffer: +0.0f leaves a nonzero
// bf16-rounded accumulator unchanged, and the only differing case is acc ==
// -0.0 (the reference skips the add and stores -0.0) — the decoder consumes
// these values in fp32 accumulation, where -0.0 == +0.0.

}  // namespace cutlass::epilogue::fusion
