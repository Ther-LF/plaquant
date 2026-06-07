"""QuantizeLinear — Linear layer that applies rotation R1/R2 dynamically in forward.

During rotation optimization training, R1/R2 are learnable parameters.
This layer applies them to the weight on-the-fly so gradients flow through R.
"""

import torch
import torch.nn as nn


class QuantizeLinear(nn.Linear):
    """Linear layer that optionally applies R1/R2 rotation to weight during forward."""

    def forward(self, input, R1=None, R2=None, transpose=False):
        if R1 is not None:
            dtype = self.weight.dtype
            bias = self.bias
            if not transpose:
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(dtype)
            else:
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(dtype)
                if bias is not None:
                    bias = torch.matmul(R1.T.to(torch.float64), bias.to(torch.float64)).to(bias.dtype)

            if R2 is not None:
                had_dim = R2.shape[0]
                if transpose:
                    init_shape = weight.shape
                    temp = weight.reshape(-1, init_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    weight = temp.reshape(init_shape)
                else:
                    W_ = weight.t()
                    transposed_shape = W_.shape
                    temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
                    temp = temp.to(torch.float64) @ R2.to(torch.float64)
                    if bias is not None:
                        bias_shape = bias.shape
                        temp_bias = bias.reshape(transposed_shape[-1] // had_dim, had_dim)
                        temp_bias = (temp_bias.to(torch.float64) @ R2.to(torch.float64)).to(bias.dtype)
                        bias = temp_bias.reshape(bias_shape)
                    weight = temp.reshape(transposed_shape).t()
            weight = weight.to(dtype)
        else:
            weight = self.weight
            bias = self.bias

        return nn.functional.linear(input, weight, bias)
