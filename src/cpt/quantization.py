"""Linear quantizer with learnable clamp values for cyclic precision training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from typing import Optional, Tuple


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class RoundPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LinearQuantizer(nn.Module):
    """Linear quantizer with learnable clamp values for model weights."""

    def __init__(self):
        """Initialize the linear quantizer."""
        super().__init__()

        self.steps = nn.ParameterDict()

    def forward(self, weight: torch.Tensor, bit_width: Optional[int]) -> torch.Tensor:
        """Quantize weights using linear quantization with learnable clamp values.

        Args:
            weight: Input weight tensor of shape [out_features, in_features] or [features]
            bit_width: Bit width for quantization (default: None, means no quantization)

        Returns:
            Dequantized weight tensor after quantization
        """
        # If no quantization requested, return original weights
        if bit_width is None:
            return weight

        qmin = -2 ** (bit_width - 1)
        qmax = 2 ** (bit_width - 1) - 1

        # Get quantization step size
        if str(bit_width) not in self.steps:
            # Initialize step size for this bit width
            if weight.dim() == 1:
                step = weight.detach().abs().mean() * 2 * (qmax ** -0.5)
            elif weight.dim() == 2:
                step = weight.detach().abs().mean(dim=1, keepdim=True) * 2 * (qmax ** -0.5)
            else:
                raise NotImplementedError("Unsupported weight dimension for quantization")

            self.steps[str(bit_width)] = nn.Parameter(step, requires_grad=True)

        # Quantize weights
        step = self.steps[str(bit_width)]
        step_grad_scale = (weight.numel() * qmax) ** -0.5
        step = GradScale.apply(step, step_grad_scale)
        quantized = torch.clamp(weight / step, qmin, qmax)
        quantized = RoundPass.apply(quantized)
        dequantized = quantized * step

        return dequantized


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""

    def __init__(self, module: nn.Module):
        """Initialize quantized linear layer.

        Args:
            module: Original linear layer to quantize
        """
        super().__init__()

        # Initialize weights and bias
        if isinstance(module, nn.Linear):
            self.weight = nn.Parameter(module.weight)
            self.bias = module.bias
        elif isinstance(module, Conv1D):
            self.weight = nn.Parameter(module.weight.t())
            self.bias = module.bias

        # Initialize quantizer
        self.quantizer = LinearQuantizer()
        if self.bias is not None:
            self.bias_quantizer = LinearQuantizer()

        self.bit_width = None

    def set_quant_config(self, bit_width: Optional[int] = None):
        """Set quantization configuration.

        Args:
            bit_width: Bit width for quantization (default: None, means no quantization)
        """
        self.bit_width = bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """

        quantized_weight = self.quantizer(self.weight, self.bit_width)
        if self.bias is not None:
            quantized_bias = self.bias_quantizer(self.bias, self.bit_width)
        else:
            quantized_bias = None

        return F.linear(x, quantized_weight, quantized_bias)
