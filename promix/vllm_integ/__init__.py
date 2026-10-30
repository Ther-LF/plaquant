"""ProMix -> vLLM integration (Phase 1: exporter + out-of-tree quant plugin + gates).

See docs/vllm_integ_phase1.md. Layout:

  shared_g.py   shared-g re-quantization for vLLM's merged QKV / gate_up GEMMs
  export.py     offline exporter: prepared ProMix model -> safetensors checkpoint
                (+ shared-g re-referenced PPL + T2 reference snapshot)
  kernels.py    JIT loader for the SM120 exts inside the vLLM venv
  plugin.py     PromixConfig(QuantizationConfig) + PromixLinearMethod
  gates/        T0 (packed-buffer torch.equal), T1 (per-layer bit-exact),
                T2 (logits top-1 + wikitext PPL) harnesses
"""
