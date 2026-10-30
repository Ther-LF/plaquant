"""vllm-perf r3 lever 2: the promix norm->quant_pack producer-fusion pass.

Rewrites, in the post-grad fx graph (where vLLM's own fusion passes run,
BEFORE vllm_ir lowering):

    norm_out, resid_out = torch.ops.vllm_ir.fused_add_rms_norm(x, res, w, eps)
    d = torch.ops.vllm.promix_linear(norm_out, handle)

into

    code_lo, sfa_lo, code_hi, sfa_hi, xn, resid_out' =
        torch.ops.vllm.promix_rms_quant_pack(x, res, w, eps, handle)
    d = torch.ops.vllm.promix_gemm_packed(code_lo, sfa_lo, code_hi, sfa_hi,
                                          xn, handle)

for the qkv_proj / gate_up_proj input sites (the only promix linears fed
directly by a norm). This is the promix analog of vLLM-NVFP4's
RMSNormQuantFusionPass (rms_quant_fusion.py): the activation quantize+pack
rides the producing RMSNorm, so no standalone act-quant launch and no separate
fp16 [M,K] read between norm and pack. The fused kernel replicates vLLM's
fused_add_rms_norm bit-exactly (see vllm_rms_quant_pack_ext.cu), so the
rewrite is BIT-INVISIBLE end-to-end (gated by norm_pack_probe torch.equal +
the T2 byte-identity gates).

A manual fx walk is used instead of torch's pattern matcher because the layer
handle is an int constant per site (the pattern-matcher tracing API has no
clean wildcard for it); the structure match is trivial (one op feeding one op).

Installed out-of-tree by install_promix_fusion_pass(): wraps
PostGradPassManager.configure to append this pass after vLLM's own passes
(still before vllm_ir lowering / cleanup / fix_functionalization, which the
manager runs after the pass list). PROMIX_NORM_FUSE=0 disables installation.
"""

import operator
import os

import torch
from torch import fx

from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.logger import init_logger

from promix.vllm_integ import ops as promix_ops

logger = init_logger(__name__)


class PromixNormPackFusionPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        FA = torch.ops.vllm_ir.fused_add_rms_norm.default
        PL = torch.ops.vllm.promix_linear.default
        RQP = torch.ops.vllm.promix_rms_quant_pack.default
        GP = torch.ops.vllm.promix_gemm_packed.default
        matched = 0
        for node in list(graph.nodes):
            if node.op != "call_function" or node.target != FA:
                continue
            # vllm_ir.fused_add_rms_norm(x, x_residual, weight, epsilon,
            #                            variance_size=None)
            args = list(node.args) + [None] * (5 - len(node.args))
            kw = dict(node.kwargs)
            x = args[0] if args[0] is not None else kw.get("x")
            res = args[1] if args[1] is not None else kw.get("x_residual")
            w = args[2] if args[2] is not None else kw.get("weight")
            eps = args[3] if args[3] is not None else kw.get("epsilon")
            vsz = args[4] if args[4] is not None else kw.get("variance_size")
            if x is None or res is None or w is None or eps is None or vsz is not None:
                continue
            # exactly two getitem users (norm out 0, residual out 1)
            gets = {}
            ok = True
            for u in node.users:
                if u.op == "call_function" and u.target == operator.getitem:
                    gets[u.args[1]] = u
                else:
                    ok = False
            if not ok or set(gets) != {0, 1}:
                continue
            out0, out1 = gets[0], gets[1]
            # the norm output must feed exactly one promix_linear and nothing else
            if len(out0.users) != 1:
                continue
            (lin,) = out0.users
            if lin.op != "call_function" or lin.target != PL:
                continue
            handle = lin.args[1]
            if not isinstance(handle, int) or not promix_ops.handle_can_norm_fuse(handle):
                continue

            fake_mode = None
            vals = []
            for n in (x, res, w):
                v = n.meta.get("val") if isinstance(n, fx.Node) else None
                vals.append(v)
            try:
                from torch._guards import detect_fake_mode
                fake_mode = detect_fake_mode(tuple(v for v in vals if v is not None))
            except Exception:
                fake_mode = None

            with graph.inserting_before(node):
                fused = graph.call_function(RQP, (x, res, w, eps, handle))
                parts = [graph.call_function(operator.getitem, (fused, i))
                         for i in range(6)]
            with graph.inserting_before(lin):
                gemm = graph.call_function(
                    GP, (parts[0], parts[1], parts[2], parts[3], parts[4], handle))

            # meta propagation (inductor sizing needs node.meta["val"])
            if fake_mode is not None and all(v is not None for v in vals):
                with fake_mode:
                    fv = RQP(vals[0], vals[1], vals[2], eps, handle)
                    gv = GP(fv[0], fv[1], fv[2], fv[3], fv[4], handle)
                fused.meta["val"] = fv
                for i, p in enumerate(parts):
                    p.meta["val"] = fv[i]
                gemm.meta["val"] = gv

            lin.replace_all_uses_with(gemm)
            out1.replace_all_uses_with(parts[5])
            graph.erase_node(lin)
            graph.erase_node(out0)
            graph.erase_node(out1)
            graph.erase_node(node)
            matched += 1
        if matched:
            logger.info("promix norm->quant_pack fusion: %d sites rewritten", matched)
        self.matched_count = matched


_installed = False


def install_promix_fusion_pass():
    """Out-of-tree pass registration: wrap PostGradPassManager.configure to
    append the promix fusion pass after vLLM's own passes. Idempotent;
    PROMIX_NORM_FUSE=0 disables (levers-off control path)."""
    global _installed
    if _installed or os.environ.get("PROMIX_NORM_FUSE", "1") == "0":
        return
    from vllm.compilation.passes.pass_manager import PostGradPassManager
    if getattr(PostGradPassManager, "_promix_fusion_patched", False):
        _installed = True
        return
    orig_configure = PostGradPassManager.configure

    def configure(self, config):
        orig_configure(self, config)
        self.passes.append(PromixNormPackFusionPass(config))
        logger.info("promix norm->quant_pack fusion pass installed")

    PostGradPassManager.configure = configure
    PostGradPassManager._promix_fusion_patched = True
    _installed = True
