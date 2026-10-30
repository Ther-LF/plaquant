"""Out-of-tree vLLM quantization plugin for ProMix.

Registers the "promix" quantization method via vLLM's
@register_quantization_config — NO vLLM fork. The exported checkpoint's
config.json carries `quantization_config = {"quant_method": "promix", ...}`;
the method resolves either via the `vllm.general_plugins` entry point
(pyproject.toml — spawned workers / `vllm serve`) or a plain
`import promix.vllm_integ.plugin` with VLLM_ENABLE_V1_MULTIPROCESSING=0.

Phase 2: the per-layer apply path is wrapped in the opaque custom op
`torch.ops.vllm.promix_linear` (see ops.py) so torch.compile tracing and
piecewise/full CUDA-graph capture work with enforce_eager=False. Set
PROMIX_EAGER_APPLY=1 to bypass the op (Phase-1 behavior, needs eager).

Per quantized Linear (vLLM merged prefixes qkv_proj / o_proj / gate_up_proj /
down_proj):
  create_weights                 registers the packed-checkpoint params
                                 (N-major, K-packed; linear-layout SF)
  process_weights_after_loading  prepacks the weight SF into the kernel's
                                 interleaved atom layout (== vLLM's
                                 swizzle_blockscale, probe-proven byte-identical)
                                 and builds a promix MixedQuantLinear around the
                                 loaded buffers (same class the promix reference
                                 runs — maximizes bit-exactness by construction)
  apply                          [down_proj: online Hadamard (fp16, folded
                                 1/sqrt(n))] -> [o_proj: column reorder] ->
                                 quant_pack_float -> fused dual-format GEMM
                                 (epilogue alpha = shared per-tensor g)
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from promix.vllm_integ import ops as promix_ops

logger = init_logger(__name__)

KINDS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")

# Escape hatch: PROMIX_EAGER_APPLY=1 bypasses the custom op and calls the
# kernels directly from apply() (Phase-1 behavior; requires enforce_eager).
_EAGER_APPLY = os.environ.get("PROMIX_EAGER_APPLY", "0") == "1"

# Register torch.ops.vllm.promix_linear (+ the r3 producer-fusion ops) in
# every process that imports the plugin (entry point loads run before model
# construction / compilation), and install the norm->quant_pack fusion pass
# out-of-tree (PROMIX_NORM_FUSE=0 disables).
promix_ops.ensure_registered()
from promix.vllm_integ.fusion_pass import install_promix_fusion_pass  # noqa: E402
install_promix_fusion_pass()


@register_quantization_config("promix")
class PromixConfig(QuantizationConfig):
    """ProMix dual-format (NVFP4 low + MXFP8 high) fused-GEMM quantization."""

    def __init__(self, k_high: dict, hadamard_down_proj: bool = True,
                 fold_had_scale: bool = True, o_proj_reorder: bool = True,
                 had_mode: str = "fp16", full_cfg: dict | None = None):
        super().__init__()
        # {"layers.{i}.self_attn.qkv_proj": K_high, ...} (vLLM merged prefixes)
        self.k_high = {str(k): int(v) for k, v in k_high.items()}
        self.hadamard_down_proj = bool(hadamard_down_proj)
        self.fold_had_scale = bool(fold_had_scale)
        self.o_proj_reorder = bool(o_proj_reorder)
        self.had_mode = had_mode
        self.full_cfg = full_cfg or {}
        if self.had_mode != "fp16":
            raise NotImplementedError(
                f"promix plugin phase 1 implements had_mode=fp16 only, got {had_mode}")

    @classmethod
    def get_name(cls) -> str:
        return "promix"

    def get_supported_act_dtypes(self):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict) -> "PromixConfig":
        return cls(
            k_high=cls.get_from_keys(config, ["k_high"]),
            hadamard_down_proj=cls.get_from_keys_or(
                config, ["hadamard_down_proj"], True),
            fold_had_scale=cls.get_from_keys_or(config, ["fold_had_scale"], True),
            o_proj_reorder=cls.get_from_keys_or(config, ["o_proj_reorder"], True),
            had_mode=cls.get_from_keys_or(config, ["had_mode"], "fp16"),
            full_cfg=config,
        )

    def _match_key(self, prefix: str):
        for key in self.k_high:
            if prefix == key or prefix.endswith("." + key):
                return key
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            key = self._match_key(prefix)
            if key is not None:
                return PromixLinearMethod(self, key)
            return UnquantizedLinearMethod()   # lm_head etc.
        return None                            # embeddings / attention / rest


@register_weight_loader_v2_supported_method
class PromixLinearMethod(LinearMethodBase):
    def __init__(self, cfg: PromixConfig, key: str):
        self.cfg = cfg
        self.key = key
        self.kind = key.split(".")[-1]
        assert self.kind in KINDS, self.kind
        self.k_high = cfg.k_high[key]

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        assert input_size == input_size_per_partition and \
            output_size == sum(output_partition_sizes), \
            "promix plugin phase 1 supports TP=1 only"
        weight_loader = extra_weight_attrs.get("weight_loader")
        K = input_size_per_partition
        N = sum(output_partition_sizes)
        K_high = self.k_high
        K_low = K - K_high
        assert K_low > 0 and K_low % 16 == 0 and K_high % 32 == 0, (self.key, K, K_high)
        layer.logical_widths = output_partition_sizes
        layer.promix_N, layer.promix_K = N, K
        layer.promix_K_low, layer.promix_K_high = K_low, K_high

        layer.register_parameter("weight_lo_packed", ModelWeightParameter(
            data=torch.empty(N, K_low // 2, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader))
        layer.register_parameter("weight_lo_scale", GroupQuantScaleParameter(
            data=torch.empty(N, K_low // 16, dtype=torch.uint8),
            input_dim=1, output_dim=0, weight_loader=weight_loader))
        if K_high > 0:
            layer.register_parameter("weight_hi_bytes", ModelWeightParameter(
                data=torch.empty(N, K_high, dtype=torch.uint8),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("weight_hi_scale", GroupQuantScaleParameter(
                data=torch.empty(N, K_high // 32, dtype=torch.uint8),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
        layer.register_parameter("weight_global_scale", PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader))
        if self.kind == "o_proj" and self.cfg.o_proj_reorder:
            layer.register_parameter("column_order", BasevLLMParameter(
                data=torch.empty(K, dtype=torch.int32),
                weight_loader=weight_loader))

    def process_weights_after_loading(self, layer):
        from promix.quantize.mixed_linear import MixedQuantLinear
        from promix.quantize.weight_pack import PackedWeight
        from promix.vllm_integ.kernels import (load_kernels, load_decode_gemv,
                                               load_smallm, load_qpfm, load_had_pack,
                                               load_splitk, load_norm_pack)
        cd, qpf = load_kernels()
        dg = load_decode_gemv()   # None when PROMIX_DECODE_GEMV=0
        # vllm-perf r2: small-M GEMM configs (lever 1), single-launch mixed pack
        # (lever 2c), fused Hadamard+pack down_proj producer (lever 2a).
        cd_sm = load_smallm()     # None when PROMIX_SMALLM=0
        qpfm = load_qpfm()        # None when PROMIX_QPFM=0
        # vllm-perf r3: split-K/stream-K small-M GEMM (lever 1), fused
        # vLLM-RMSNorm + quant/pack producer for qkv/gate_up (lever 2).
        cd_sk = load_splitk()     # None when PROMIX_SPLITK=0
        vrq = None
        if self.kind in ("qkv_proj", "gate_up_proj"):
            vrq = load_norm_pack()   # None when PROMIX_NORM_FUSE=0
        hpk, hpk28 = (None, None)
        if self.kind == "down_proj" and self.cfg.hadamard_down_proj:
            hpk, hpk28 = load_had_pack()   # (None, None) when PROMIX_HAD_PACK=0

        N, K = layer.promix_N, layer.promix_K
        K_low, K_high = layer.promix_K_low, layer.promix_K_high
        dev = layer.weight_lo_packed.device

        gs = layer.weight_global_scale.data
        if torch.unique(gs).numel() != 1:
            raise ValueError(
                f"promix {self.key}: merged shards carry different per-tensor g "
                f"({gs.tolist()}) — checkpoint must be shared-g re-quantized")
        g = float(gs[0])

        physSFB_lo = cd.prepack_sf_fp4_b(
            layer.weight_lo_scale.data.contiguous(), N, K_low)
        if K_high > 0:
            W_hi = layer.weight_hi_bytes.data.contiguous()
            physSFB_hi = cd.prepack_sf_fp8_b(
                layer.weight_hi_scale.data.contiguous(), N, K_high)
        else:
            W_hi = torch.empty((N, 0), dtype=torch.uint8, device=dev)
            physSFB_hi = torch.empty((0,), dtype=torch.uint8, device=dev)

        packed = PackedWeight(
            N=N, K=K, K_low=K_low, K_high=K_high,
            W_lo_packed=layer.weight_lo_packed.data.contiguous(),
            physSFB_lo=physSFB_lo,
            W_hi_bytes=W_hi, physSFB_hi=physSFB_hi,
            W_deq=None, bias=None, g_w=g)
        # vllm-perf r1: decode-GEMV buffers, derived at load time (no exporter /
        # checkpoint change). The GEMV streams the SAME packed code bytes the
        # fused GEMM uses (checkpoint weight_lo_packed == nv.pack_fp4 == the
        # GEMV's even-k-low-nibble layout; weight_hi_bytes is raw FP8), plus raw
        # FP32 effective block scales with the per-tensor g folded in — exactly
        # pack_weight_mixed's W_*_dec/S_*_dec recipe.
        if dg is not None and K_high > 0:
            packed.W_lo_dec = packed.W_lo_packed                     # shared bytes
            packed.W_hi_dec = packed.W_hi_bytes                      # shared bytes
            if (os.environ.get("PROMIX_DECODE_SF8", "0") == "1"
                    and os.environ.get("PROMIX_DECODE_MODE", "w4a16") != "w4a4"
                    and hasattr(dg, "decode_gemv_w4a16_sf8")):
                # vllm-perf r2 (secondary lever): stream the checkpoint's RAW
                # 1-byte block scales (E4M3 low / UE8M0 high); the _sf8 GEMVs
                # fold g in-kernel. BIT-EXACT (sf8_probe torch.equal, all shapes
                # M=1..16) but MEASURED SLOWER under graph replay (0.92-0.98x,
                # r2_sf8_probe.json): the in-kernel E4M3 cvt / ldexpf outweighs
                # the ~18% low-segment byte saving — r1's +8-12% estimate is
                # REFUTED. DEFAULT-OFF; kept for experiments.
                packed.S_lo_dec = layer.weight_lo_scale.data.contiguous()
                packed.S_hi_dec = layer.weight_hi_scale.data.contiguous()
            else:
                packed.S_lo_dec = (layer.weight_lo_scale.data
                                   .view(torch.float8_e4m3fn).float() * g).contiguous()
                packed.S_hi_dec = (torch.exp2(
                    layer.weight_hi_scale.data.float() - 127.0) * g).contiguous()
        # down_proj online-Hadamard setup (needed before mql construction: the
        # fused Hadamard+pack producer takes the CPU 28x28 had_K for K=14336).
        had_K, Kfac = None, 1
        if self.kind == "down_proj" and self.cfg.hadamard_down_proj:
            from promix.quantize.hadamard import get_hadK
            had_K, Kfac = get_hadK(K)
            layer.promix_had_K = None if had_K is None else had_K.to(dev)
            layer.promix_had_Kfac = Kfac
        mql = MixedQuantLinear(packed, nv=None, mx=None, cd=cd,
                               out_dtype=torch.float16, qpf=qpf, dg=dg,
                               qpfm=qpfm, cd_sm=cd_sm, cd_sk=cd_sk,
                               hpk=hpk, hpk28=hpk28,
                               had28=None if had_K is None else had_K.cpu())
        layer.promix_mql = mql.to(dev)

        # vllm-perf r4 lever 1: Marlin W4A16 small-M route — repack the low
        # segment into vLLM's Marlin fp4 layout ONCE at load (tile-interleaved
        # qweight + S0E5M3 scales + folded global scale). Honest memory cost: a
        # second copy of the low-segment weight bytes (the fused prefill GEMM
        # keeps the original layout). PROMIX_MARLIN=0 disables (and skips the
        # memory). Routed by the measured _MARLIN_TABLE windows (M=8..32).
        from promix.vllm_integ.marlin_w4a16 import (marlin_enabled,
                                                    marlin_hi_enabled,
                                                    prepack_marlin_high,
                                                    prepack_marlin_low)
        if marlin_enabled() and dg is not None and K_high > 0:
            try:
                ml = prepack_marlin_low(packed.W_lo_packed,
                                        layer.weight_lo_scale.data.contiguous(),
                                        g, N, K_low)
                # vllm-perf r5 lever 1: MXFP8-marlin high companion — the r4
                # hi_add GEMV scales ~linearly in M (7x the marlin low's
                # per-byte cost at M=16, closing every window by M=16-24);
                # marlin's native MXFP8 path (e8m0 group-32 scales, exactly our
                # checkpoint layout) runs the high segment on tensor cores
                # near-flat in M. g rides the companion's add alpha. Memory:
                # a second copy of the high-segment bytes (~K_high/K of the
                # weight). PROMIX_MARLIN_HI=0 restores the r4 hi_add-only route.
                mh = None
                if marlin_hi_enabled():
                    mh = prepack_marlin_high(packed.W_hi_bytes,
                                             layer.weight_hi_scale.data.contiguous(),
                                             g, N, K_high)
                # hi_add companion rows/block: large N saturates with 32
                # warps, small N stays launch-bound at 16 (r4_marlin_probe
                # t_hi_add sweep; same heuristic family as _dec_warps).
                if not layer.promix_mql.attach_marlin(
                        ml, hi_warps=32 if N >= 4096 else 16, marlin_high=mh):
                    logger.info("promix %s: marlin route unavailable (sf8/dg)", self.key)
            except Exception as e:
                logger.warning("promix %s: marlin prepack failed (%r) — "
                               "W4A4 GEMM route kept", self.key, e)

        if self.kind == "o_proj" and self.cfg.o_proj_reorder:
            layer.promix_col_order = layer.column_order.data.to(dev, torch.long)
            # vllm-perf r3 lever 3: int32 copy for the gather-folded qpfm pack
            # (PROMIX_GATHER_PACK=0 keeps the separate torch gather everywhere).
            if os.environ.get("PROMIX_GATHER_PACK", "1") != "0":
                layer.promix_col_order_i32 = layer.promix_col_order.to(torch.int32).contiguous()
            else:
                layer.promix_col_order_i32 = None
            del layer.column_order
        # free the linear-layout params (data lives on inside mql / was prepacked)
        for pname in ("weight_lo_packed", "weight_lo_scale", "weight_hi_bytes",
                      "weight_hi_scale", "weight_global_scale"):
            if hasattr(layer, pname):
                delattr(layer, pname)
        # Phase 2: register the graph-safe custom-op runner for this layer.
        # mql/vrq ride along for the r3 norm->quant_pack fusion pass (qkv/gate_up
        # only — the norm-fed sites); handle_can_norm_fuse gates on them.
        layer.promix_handle = promix_ops.register_layer(
            lambda x, layer=layer: self._apply_eager(layer, x), N,
            mql=layer.promix_mql if vrq is not None else None, vrq=vrq)

    def _apply_eager(self, layer, x):
        """The Phase-1 eager apply body (runs inside the custom op)."""
        mql = layer.promix_mql
        if self.kind == "down_proj" and self.cfg.hadamard_down_proj:
            # vllm-perf r2 lever 2: on the GEMM route, run the fused
            # Hadamard+quant/pack producer (one kernel, no fp16 [M,K] HBM
            # round-trip). GEMV routes (decode small M) consume an fp16
            # activation, so they keep the torch Hadamard. M is static per
            # captured CUDA graph (r1), so this branch is capture-safe.
            M = x.numel() // x.shape[-1]
            if mql.can_had_pack() and not mql.fp16_route(M):
                return mql.forward_had_pack(x)
            from promix.quantize.hadamard import matmul_hadU_cuda
            x_dtype = x.dtype
            x = matmul_hadU_cuda(x, layer.promix_had_K, layer.promix_had_Kfac,
                                 fold_scale=self.cfg.fold_had_scale).to(x_dtype)
        elif self.kind == "o_proj" and self.cfg.o_proj_reorder:
            # vllm-perf r3 lever 3: on the GEMM route, fold the static column
            # reorder into the qpfm pack's load (one kernel, no fp16 [M,K]
            # gather round trip). GEMV routes consume an fp16 activation, so
            # they keep the torch gather. M is static per captured CUDA graph
            # (r1), so this branch is capture-safe.
            M = x.numel() // x.shape[-1]
            if (getattr(layer, "promix_col_order_i32", None) is not None
                    and mql.can_gather_pack() and not mql.fp16_route(M)):
                return mql.forward_gather_pack(x, layer.promix_col_order_i32)
            x = x[..., layer.promix_col_order]
        return mql(x)

    def apply(self, layer, x, bias=None):
        assert bias is None, "promix phase 1: bias-free linears only (llama)"
        if _EAGER_APPLY:
            return self._apply_eager(layer, x)
        return torch.ops.vllm.promix_linear(x, layer.promix_handle)


def register():
    """vllm.general_plugins entry-point hook (import side effect registers)."""
    logger.info("promix quantization plugin registered")
