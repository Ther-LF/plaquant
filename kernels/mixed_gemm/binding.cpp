#include <torch/extension.h>

// Forward declarations
torch::Tensor mixed_gemm_forward(
    torch::Tensor x_main,      // (M, K_main) INT4 quantized activations
    torch::Tensor x_high,      // (M, K_high) INT8 quantized activations
    torch::Tensor w_main,      // (N, K_main) INT4 quantized weights
    torch::Tensor w_high,      // (N, K_high) INT8 quantized weights
    torch::Tensor scale_x_main,// (M, 1) per-token activation scale
    torch::Tensor scale_x_high,// (M, 1) per-token activation scale
    torch::Tensor scale_w_main,// (N, 1) per-channel weight scale
    torch::Tensor scale_w_high,// (N, 1) per-channel weight scale
    torch::Tensor zero_x_main, // (M, 1) per-token activation zero point
    torch::Tensor zero_x_high  // (M, 1) per-token activation zero point
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mixed_gemm_forward, "Mixed-precision GEMM (INT4+INT8)");
}
