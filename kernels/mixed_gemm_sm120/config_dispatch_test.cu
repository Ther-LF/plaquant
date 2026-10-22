// config_dispatch_test.cu — exercises config_dispatch.hpp.
//
// Built once PER KIND (separate module, one DISPATCH_ENABLE_* each) so no single TU
// instantiates more than one kernel family's winning configs -> bounded compile. Exposes:
//   tag_<kind>(M,N,K[,Kh]) -> std::string   (the selected ConfigTag name; host, cheap)
//   run_<kind>(...)         -> bf16 tensor   (launch_best_<kind> -> chosen instantiation)
// The python test (config_dispatch_test.py) checks tag == measured-best (from the round-24
// CSVs) AND cosine == 1.0 for every (shape, M).

#include <torch/extension.h>
#include "config_dispatch.hpp"

namespace pd = promix_dispatch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(DISPATCH_ENABLE_NVFP4)
  m.def("tag", [](int64_t M, int64_t N, int64_t K){ return pd::tag_nvfp4((int)M,(int)N,(int)K); });
  m.def("run", &pd::launch_best_nvfp4);   // (A,SFA,B, dummy, SFB, M,N,K)
#elif defined(DISPATCH_ENABLE_MXFP8)
  m.def("tag", [](int64_t M, int64_t N, int64_t K){ return pd::tag_mxfp8((int)M,(int)N,(int)K); });
  m.def("run", &pd::launch_best_mxfp8);
#elif defined(DISPATCH_ENABLE_CODESIGN)
  m.def("tag", [](int64_t M, int64_t N, int64_t K, int64_t Kh){ return pd::tag_codesign((int)M,(int)N,(int)K,(int)Kh); });
  m.def("run", &pd::launch_best_codesign);  // (A_lo,SFA_lo,B_lo,SFB_lo, A_hi,SFA_hi,B_hi,SFB_hi, M,N,Klo,Khi)
#endif
}
