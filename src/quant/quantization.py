"""Symmetric MinMax quantization implementation for model weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from typing import Optional


class SymmetricMinMaxQuantizer:
    """Symmetric MinMax quantizer for model weights."""

    def __init__(self, weight: torch.Tensor):
        """Initialize the quantizer.

        Args:
            weight: Input weight tensor of shape [out_features, in_features] or [features]
        """
        self.weight = weight

        if weight.dim() == 1:
            self.max_val = torch.max(torch.abs(weight))
        elif weight.dim() == 2:
            # For 2D weights, we use per-channel quantization
            self.max_val = torch.max(torch.abs(weight), dim=1, keepdim=True).values
        else:
            raise NotImplementedError("Only 1D and 2D tensors are supported for quantization.")
        # Avoid division by zero
        self.max_val = torch.where(self.max_val == 0, torch.ones_like(self.max_val), self.max_val)

    def quantize_weights(self, bit_width: Optional[int]) -> torch.Tensor:
        """Quantize weights using per-channel symmetric MinMax quantization.

        Args:
            weights: Input weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weights, scale) where scale has shape [out_features, 1]
        """
        if bit_width is None:
            # No quantization, return original weights
            return self.weight

        # Calculate scale factors for quantization
        qmin = -2 ** (bit_width - 1)
        qmax = 2 ** (bit_width - 1) - 1
        scale = self.max_val / qmax

        # Quantize weights per channel
        quantized = torch.round(self.weight / scale).clamp(qmin, qmax)

        # Convert back to floating point
        dequantized = quantized * scale

        return dequantized.detach()


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
        self.quantizer = SymmetricMinMaxQuantizer(self.weight)
        if self.bias is not None:
            self.bias_quantizer = SymmetricMinMaxQuantizer(self.bias)

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

        quantized_weight = self.quantizer.quantize_weights(self.bit_width)
        if self.bias is not None:
            quantized_bias = self.bias_quantizer.quantize_weights(self.bit_width)
        else:
            quantized_bias = None

        return F.linear(x, quantized_weight, quantized_bias)
