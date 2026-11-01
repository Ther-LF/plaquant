# Peer W4A4 Methods on RTX Pro 6000 (SM120) — Status Log

Host: ICT90, 8x RTX Pro 6000 Blackwell Server (cc 12.0, 96GB), CUDA 13.0, container luofan-plaquant (pytorch 2.11 cu130 image).
Serving stack: /home/luofan/baselines/venv (Python 3.12, vllm 0.32.0, torch 2.13.0+cu130, transformers 5.17).
GPU used: 0 (occasionally 4). All logs under /home/luofan/vllm_integ/bench_final/peers/.

## Peer 1: SharQ (arXiv 2606.26587) — IN PROGRESS
- Shipped kernel build (kernels/build_cmake_sm120a/sharq_ops.so, built vs container torch 2.11): imports, but ABI-INCOMPATIBLE with venv torch 2.13 ("W must be contiguous" bogus error from garbage TensorImpl reads).
  - Evidence: smoke_test.py passes under container python/torch 2.11 (cosine 0.9937 vs bf16 ref), fails under venv torch 2.13.
- DEVIATION 1 (build glue only): rebuilt identical kernel sources against venv torch 2.13 into kernels/build_cmake_sm120a_venv (log: sharq/kbuild_venv.log, BUILD_EXIT=0). Same smoke-test cosine 0.9937 => numerically identical.
- DEVIATION 2 (glue only): out-of-tree registration wrapper sharq/sharq_register.py imports their benchmarks/vllm/sharq.py AS-SHIPPED and registers via vllm register_quantization_config("sharq"). No edits to their files or to the vllm package.
- Note: their sharq.py prefers module `sharq_ops_vllm` with fused sharq_linear_forward; repo only ships `sharq_ops` => fallback 3-kernel path (quantize_x + sparse_matmul + matmul_accum) is used, as shipped.
- Quantization of unsloth/Llama-3.2-3B running: sharq/quant_3b.log -> ckpt at sharq/ckpt-llama32-3b.

## Peer 2: fp_quant / QuTLASS (Quartet) — IN PROGRESS
- qutlass cloned from github (IST-DASLab/qutlass v0.2.0). Building from source vs venv torch 2.13 / CUDA 13.0 / sm_120a: qutlass/build.log.
  - Hiccup 1: container lacked git (cmake FindGit fail) -> apt-get install git.
  - Hiccup 2: git dubious-ownership in container -> git config safe.directory "*".
- vLLM 0.32 has native fp_quant method wired to torch.ops._qutlass_C. Needs an FP-Quant format checkpoint (quant_method fp_quant); HF offline => must produce locally via IST-DASLab/FP-Quant repo.

## Peer 3: MicroMix (arXiv 2508.02343) — PENDING
- Kernel mixedgemm.so already built for sm_120a (kbuild_sm120.log EXIT=0, built vs container torch 2.11).
- Plan: run their own e2e path (model/, test.sh) under container python (torch 2.11) to match kernel ABI.

## Progress update (2026-11-01 ~09:40 UTC)

### SharQ
- Quantized ckpt for unsloth/Llama-3.2-3B: /home/luofan/vllm_integ/bench_final/peers/sharq/ckpt-llama32-3b (log sharq/quant_3b.log EXIT=0).
  - DEVIATION 3: Llama-3.2-3B ties lm_head<->embed_tokens; their sharq_quant.py save_file() crashes on shared tensors. Used minimally-adapted copy sharq/sharq_quant_tied.py that drops lm_head.weight when tie_word_embeddings (vLLM re-ties). No quantization-method change.
- SERVES in vLLM 0.32 (our venv, torch 2.13, out-of-tree registration): smoke run isl128/osl16/b1 OK, full torch.compile + FULL_AND_PIECEWISE CUDA graphs (sharq/serve_smoke.log, sharq3b_smoke.json). VERDICT so far: RAN.
- Accuracy via THEIR OWN pipeline (model/main.py, real SM120 kernels, wikitext2, seqlen 2048, Llama-3.2-3B):
  - SHARQ:  PPL 8.385  (sharq/ppl_sharq_3b.log)
  - NVFP4 (their dense baseline): PPL 8.609 (sharq/ppl_nvfp4_3b.log)
  - DEVIATION 4 (eval env): their pin transformers==4.44.0 cannot parse the modern tokenizer.json (tokenizers<0.20); used transformers==4.47.1 + tokenizers 0.21.4 in venv_eval (overlay on container torch 2.11, shipped kernel .so).
- Full serving protocol run PENDING a quiet GPU window (foreign tenants at 40-100% util on GPUs 0 and 4 contaminate latency).

### fp_quant / QuTLASS
- KEY FINDING: our vllm 0.32.0 cu130 wheel BUNDLES qutlass kernels (torch.ops._qutlass_C with matmul_mxf4_bf16_tn, fusedQuantizeMxQuest/AbsMax, fusedQuantizeNvAbsMax) built at /workspace/.deps/qutlass-src. No source build needed for serving.
- We also built qutlass v0.2.0 from source vs torch 2.13/CUDA 13.0/sm_120a successfully (qutlass/build.log EXIT=0) but UNINSTALLED the pip package: importing it alongside vllm duplicates the _qutlass_C op registration (hard crash).
- FP-Quant checkpoint (NVFP4 W4A4, RTN + hadamard-128 transform, fuse_global_scale) being produced from IST-DASLab/FP-Quant @ main with venv_fpq (torch 2.11 system + transformers 4.57.1 + fast-hadamard-transform built from github source): log qutlass/quant_3b_nvfp4_rtn.log.
  - DEVIATION: offline-c4 calibration loader (qutlass/fpq_quant_driver.py monkeypatches get_data to read cached arrow shards of the same c4 file; sampling logic reproduced verbatim).

### MicroMix
- ICLR-2026 branch (micromix) cloned separately: /home/luofan/baselines/peers/micromix/MicroMix-iclr (main branch lacks the e2e benchmark script its prof_micromix.sh references).
- ICLR mgemm kernels built for SM120 vs container torch 2.11 (micromix_kbuild_iclr.log EXIT=0).
- Generating ICLR-format act stats for llama-3-8b (reorder_indices.py --act_sort_metric mean; mm_reorder_8b.log). Plan: e2e prefill/decode benchmark benchmark_e2e_micromix.py (synthetic weights, real kernels; their paper protocol) + accuracy PPL via main-branch model/main.py with existing saved/ stats.

## Progress update 2 (2026-11-01 ~10:15 UTC)

### fp_quant / QuTLASS — SERVES on SM120 (ran-with-caveat)
- BLOCKER (as-shipped wheel): vllm 0.32.0 cu130 wheel bundles vllm/_qutlass_C.abi3.so with ONLY sm_100 cubins (verified via cuobjdump --list-elf => all sm_100). On RTX Pro 6000 every fp_quant quantize kernel fails: "RuntimeError: runGemmNv, fused_quantize_nv_sm100.cu:187, Error Internal" / "fused_quantize_mx.cu:99, Error Internal" (qutlass/serve_smoke*.log of earlier attempts).
- FIX (documented substitution, same qutlass project): copied venv /home/luofan/baselines/venv_fpq_serve = baselines venv copy, removed the sm100-only bundled .so (its import is contextlib.suppress(ImportError)-guarded), installed OUR qutlass v0.2.0 built from source for sm_120a+sm_100a vs torch 2.13/CUDA 13.0. vllm fp_quant then finds torch.ops._qutlass_C from the pip package (import via driver qutlass/bench_serve_fpquant.py).
- Checkpoints (Llama-3.2-3B, FP-Quant repo, RTN + online hadamard, fuse_global_scale, c4-128x2048 calib):
  - NVFP4 W4A4 (hadamard 128): qutlass/ckpt-llama32-3b-nvfp4-rtn (their-pipeline wikitext2 PPL 9.62, 141 chunks, quant_3b_nvfp4_rtn.log)
  - MXFP4 W4A4 (hadamard 32):  qutlass/ckpt-llama32-3b-mxfp4-rtn (their-pipeline wikitext2 PPL 10.25, 141 chunks, quant_3b_mxfp4_rtn.log)
  - "-vllm" variants have QAT-only backward_hadamard_matrix tensors stripped (vLLM loader rejects unknown params; strip_bwd_hadamard.py).
- vLLM 0.32 marks fp_quant deprecated => requires allow_deprecated_quantization=true (passed via extra kwargs).
- SMOKE SERVE OK on SM120 for BOTH formats: MXFP4 (fpquant3b_smoke.json: b1 isl128 TTFT 8.3ms) and NVFP4 (fpquant3b_nvfp4_smoke.json: TTFT 14.9ms) — numbers from contended GPU, indicative only.
- Generation sanity: coherent completions (qutlass/gen_check.log).

### MicroMix paper numbers (arXiv 2508.02343v2) for cross-paper table
- RTX PRO 6000 (same GPU as ours), Figure 7: prefill latency ~85% lower than Atom and QuaRot; decode throughput 1.82-3.02x vs Atom. End-to-end integrated into Transformer: 1.98-2.02x vs FP16.
- RTX 5090 kernel-level: 2.29-3.38x vs TensorRT-FP16, up to 1.74x vs TRT FP8. RTX 5090 e2e prefill (Llama2-7B/Llama3.1-8B, bs {8,12}, seq 2048): ~2.0x latency reduction vs HF FP16 at bs8.
- Accuracy (Llama3.1-8B wikitext2 PPL): FP16 6.24; MicroMix 6.72 (Table 11) / 6.97 at avg 5.51 bits (main table).

### Serving-bench orchestration
- Foreign tenants keep 40-100% util on GPUs 0/4; latency runs require a free window. peer_bench_orchestrator.sh polls both GPUs and, when one is fully free (>=88GiB free, <=5% util), runs sequentially at --gpu-mem-util 0.90: SharQ full protocol -> SharQ qserve-style -> fp_quant MXFP4 full protocol -> MicroMix their-protocol e2e (bs 1/32, prefill 2048, decode 128) -> MicroMix FP16 baseline. Status: peers/orchestrator.status.

## CLEAN SERVING NUMBERS (free GPU 0, no co-tenant; window 10:11-10:45 UTC 2026-11-01)

Protocol = our bench_serve.py: vLLM 0.32.0, torch 2.13.0+cu130, RTX Pro 6000 Blackwell Server, dtype bf16, gpu-mem-util 0.90, max-model-len 4096, max_num_seqs 512, prefix caching off, compile+CUDA graphs ON, greedy, median of 3.

### SharQ (Llama-3.2-3B W4A4 sparse+residual NVFP4, out-of-tree vllm integration) — sharq/sharq3b_main.json
isl1024/osl128: b1 TTFT 33.0ms TPOT 3.51ms 268 tok/s | b16 399ms/4.88ms/2011 | b32 790ms/6.72ms/2493 | b64 1856ms/11.04ms/2514 | b128 3915ms/16.76ms/2711 | b256 7511ms/33.94ms/2772
isl128/osl1024: b1 TTFT 7.1ms TPOT 3.61ms 277 tok/s | b16 57ms/5.15ms/3075 | b32 118ms/6.91ms/4557 | b64 238ms/8.33ms/7484 | b128 480ms/12.90ms/9582 | b256 1018ms/21.40ms/11442
prefill TTFT L2048: b1 63.7ms, b16 1020ms, b32 1903ms, b64 3817ms, b128 7621ms, b256 15657ms
qserve-style (1024:512): b64 TPOT 12.12ms gen 4075 tok/s | b128 20.80ms/4571 | b256 37.31ms/4946 (sharq/sharq3b_qserve.json)

### fp_quant MXFP4 W4A4 RTN+hadamard32 (vLLM-native method, sm120 qutlass source build) — qutlass/fpquant3b_mxfp4_main.json
isl1024/osl128: b1 TTFT 51.7ms TPOT 5.59ms 168 tok/s | b16 164ms/6.91ms/1967 | b32 319ms/11.82ms/2249 | b64 632ms/11.74ms/3859 | b128 1255ms/18.51ms/4544 | b256 2504ms/34.62ms/4748
isl128/osl1024: b1 TTFT 7.8ms TPOT 5.62ms 178 tok/s | b16 55ms/6.39ms/2484 | b32 58ms/7.37ms/4312 | b64 100ms/9.32ms/6800 | b128 174ms/13.07ms/9678 | b256 290ms/17.59ms/14338
prefill TTFT L2048: b1 52.8ms, b16 291ms, b32 577ms, b64 1155ms, b128 2282ms, b256 4584ms

### SharQ zero-shot lm-eval (their pipeline, real kernels, Llama-3.2-3B, 0-shot; sharq/lmeval_sharq_3b.log)
arc_challenge 0.4104 (norm 0.4403) | arc_easy 0.7281 (0.7075) | boolq 0.6832 | hellaswag 0.5392 (0.7226) | lambada_openai 0.6918 | piqa 0.7617 (0.7612) | winogrande 0.6803

## FINAL RESULTS (all timing on a verified-free GPU 0 window, 2026-11-01 10:11-12:29 UTC)

### fp_quant NVFP4 W4A4 RTN+hadamard128 (sm120 qutlass source build) — qutlass/fpquant3b_nvfp4_main.json
isl1024/osl128: b1 TTFT 56.9ms TPOT 5.67ms 165 tok/s | b16 143ms/6.61ms/2084 | b32 277ms/8.10ms/3135 | b64 548ms/10.60ms/4325 | b128 1088ms/16.16ms/5218 | b256 2165ms/28.71ms/5639
isl128/osl1024: b1 TTFT 7.4ms TPOT 5.34ms 187 tok/s | b16 64ms/6.09ms/2603 | b32 68ms/6.86ms/4624 | b64 76ms/8.38ms/7580 | b128 154ms/11.54ms/10960 | b256 295ms/18.13ms/13916
prefill TTFT L2048: b1 56.6ms, b16 296ms, b32 586ms, b64 1167ms, b128 2332ms, b256 4916ms
NOTE: NVFP4 quantize kernels DO run on SM120 when qutlass v0.2.0 is built from source with sm_120a SASS; only the vllm-wheel bundled sm_100-only binary was broken.

### MicroMix e2e (THEIR shipped benchmark: benchmarks/benchmark_e2e_micromix.py, micromix/ICLR branch, llama-3.1-8b config with synthetic weights + real SM120 kernels, channel stats calibrated from llama-3-8b [identical dims]; GPU 0 free) — mm_e2e_micromix.log / mm_e2e_fp16.log
- Prefill 2048, bs1:  MicroMix  92.4 ms  | HF-FP16(real llama-3-8b) 103.0 ms  => 1.11x
- Prefill 2048, bs32: MicroMix 1982.5 ms | HF-FP16 3955.2 ms                  => 2.00x (matches paper 1.98-2.02x e2e vs FP16)
- Prefill+128-step decode, bs1:  MicroMix 11298.7 ms (≈87.5 ms/step => 11.4 tok/s) | FP16 2619.8 ms (≈19.7 ms/step => 50.9 tok/s) => MicroMix decode 4.4x SLOWER as shipped
- Prefill+128-step decode, bs32: MicroMix 13252.8 ms (≈88.0 ms/step => 363 tok/s)  | FP16 15559.4 ms (≈90.7 ms/step => 353 tok/s) => ~1.03x
- Their decode step time is batch-size-independent (87.5 vs 88.0 ms) => decode path is Python/eager overhead-bound as shipped. Paper decode claims are vs Atom, not FP16 — consistent with what we see.
- MicroMix accuracy via THEIR pipeline (main branch, model/main.py --do_quant, real kernels, existing saved/ stats): Llama-3.2-3B wikitext2 PPL 9.045 (mm_ppl_3b.log).
- Caveats: kernels built with CUDA 13.0 nvcc (authors require 12.8; unvalidated by authors on CUDA 13); e2e benchmark only exists on the micromix (ICLR-2026) branch; per-branch kernels rebuilt (micromix_kbuild_iclr.log EXIT=0); llama-3.1-8b stats symlinked from llama-3-8b calibration (reorder_indices.py --act_sort_metric mean, mm_reorder_8b.log, avg bits 5.35).

## FINAL VERDICTS
- SharQ: RAN (end-to-end vLLM 0.32 serving on SM120 + their-pipeline accuracy). Deviations: kernel rebuilt vs torch 2.13 (ABI), tied-embeddings checkpoint fix, out-of-tree registration wrapper, eval env transformers 4.47.1 (their 4.44 pin cannot parse modern tokenizer.json).
- fp_quant/QuTLASS (Quartet): RAN-WITH-CAVEAT. vLLM-native method serves MXFP4 and NVFP4 W4A4 checkpoints on SM120, but ONLY after replacing the vllm-wheel bundled sm_100-only qutlass binary with a source build including sm_120a (venv copy; no shared-venv modification). As literally shipped in the vllm 0.32.0 cu130 wheel: BLOCKED on SM120 (cuobjdump: sm_100 cubins only; runtime "Error Internal" in fused_quantize kernels). Plus: deprecated-method flag required; QAT-only backward_hadamard_matrix tensors must be stripped for the vLLM loader.
- MicroMix: RAN-WITH-CAVEAT (their own benchmark + accuracy pipeline, real SM120 kernels, CUDA 13 build unvalidated by authors; e2e decode slow-as-shipped).

## Shared-venv integrity note
/home/luofan/baselines/venv was left functionally untouched for the main campaign: bundled vllm/_qutlass_C.abi3.so intact, our temporary qutlass pip install was removed the same hour. One inert addition remains: pybind11==3.1.0 (needed as a CMake config package for the SharQ kernel rebuild; it has no import-time side effects on vllm/torch).
All fp_quant serving used the separate copy /home/luofan/baselines/venv_fpq_serve.
