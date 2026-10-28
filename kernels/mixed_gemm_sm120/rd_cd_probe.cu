// rd_cd_probe.cu — RD perf diagnostic: expose BOTH co-design tile configs explicitly
// so we can measure per-K_high latency for each tile (the config_dispatch auto path
// only ever picks 128x128 at M>=2048). Includes the SAME protected config_dispatch.hpp.
#include <torch/extension.h>
#include "config_dispatch.hpp"

namespace pd = promix_dispatch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // (A_lo,SFA_lo,B_lo,SFB_lo, A_hi,SFA_hi,B_hi,SFB_hi, M,N,Klo,Khi) -> bf16
  m.def("run_t128x128", &pd::CdG_128x128::run);
  m.def("run_t128x64",  &pd::CdG_128x64::run);
  m.def("tag", [](int64_t M, int64_t N, int64_t K, int64_t Kh){
      return pd::tag_codesign((int)M,(int)N,(int)K,(int)Kh); });
}
