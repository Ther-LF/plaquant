/*
 * PyTorch binding for Mixed-Precision FlashAttention.
 *
 * API:
 *   int8_flash_attn(Q_int8, K_int8, V_fp16, scale_q, scale_k, scale, causal)
 *     -> O_fp16  (B, H, Lq, d_head)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// From mixed_flash_attn.cu
torch::Tensor int8_flash_attn(
    torch::Tensor Q_int8,    // (B, H, Lq, D) INT8
    torch::Tensor K_int8,    // (B, H, Lkv, D) INT8
    torch::Tensor V_fp16,    // (B, H, Lkv, D) FP16
    float scale_q,
    float scale_k,
    float scale_s,           // softmax scale (1/sqrt(D))
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Mixed-Precision FlashAttention — CUTLASS 3.x Hopper";
    m.def("int8_flash_attn", &int8_flash_attn,
          "INT8 FlashAttention forward",
          py::arg("Q_int8"), py::arg("K_int8"), py::arg("V_fp16"),
          py::arg("scale_q"), py::arg("scale_k"), py::arg("scale_s"),
          py::arg("causal") = false);
}
