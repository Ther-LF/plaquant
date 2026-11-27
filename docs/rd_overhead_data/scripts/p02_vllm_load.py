"""P0-2 sub-task B: load a model in vLLM (no benchmark) to read weight memory."""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dtype", default="auto")
ap.add_argument("--tag", required=True)
ap.add_argument("--json_out", required=True)
args = ap.parse_args()

import torch
from vllm import LLM

t0 = time.perf_counter()
llm = LLM(model=args.model, dtype=args.dtype, max_model_len=4096,
          gpu_memory_utilization=0.90, enable_prefix_caching=False,
          enforce_eager=True, disable_log_stats=True)
load_s = time.perf_counter() - t0
meta = {"tag": args.tag, "model": args.model, "load_s": load_s,
        "alloc_after_init_bytes": torch.cuda.memory_allocated(),
        "env_decode_gemv": os.environ.get("PROMIX_DECODE_GEMV"),
        "env_marlin": os.environ.get("PROMIX_MARLIN")}
try:
    mc = llm.llm_engine.model_config
    meta["quantization_resolved"] = mc.quantization
    meta["dtype_resolved"] = str(mc.dtype)
except Exception as e:
    meta["introspect_err"] = repr(e)
with open(args.json_out, "w") as f:
    json.dump(meta, f, indent=2)
print("P02_LOAD_OK " + json.dumps(meta))
