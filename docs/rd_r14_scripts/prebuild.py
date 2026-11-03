import time
from promix.vllm_integ import kernels as k
t0 = time.time(); k.load_kernels(); print("cd+qpf", round(time.time()-t0, 1), flush=True)
for name in ("load_decode_gemv", "load_smallm", "load_qpfm", "load_splitk",
             "load_norm_pack", "load_cd_bias", "load_had_pack"):
    t = time.time(); getattr(k, name)(); print(name, round(time.time()-t, 1), flush=True)
print("PREBUILD_DONE", flush=True)
