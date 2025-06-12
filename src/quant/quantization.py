"""Symmetric MinMax quantization implementation for model weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class SymmetricMinMaxQuantizer:
    """Symmetric MinMax quantizer for model weights."""

    def __init__(self, bit_width: int = 8):
        """Initialize the quantizer.

        Args:
            bit_width: Number of bits for quantization (default: 8)
        """
        self.bit_width = bit_width
        self.qmin = -(2 ** (bit_width - 1))
        self.qmax = 2 ** (bit_width - 1) - 1

    def quantize_weights(self, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights using per-channel symmetric MinMax quantization.

        Args:
            weights: Input weight tensor of shape [out_features, in_features]

        Returns:
            Tuple of (quantized_weights, scale) where scale has shape [out_features, 1]
        """
        max_vals = torch.max(torch.abs(weights), dim=1, keepdim=True)[0]  # [out_features, 1]

        # Calculate scale factor per channel
        scales = max_vals / self.qmax
        # Avoid division by zero
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)

        # Quantize weights per channel
        quantized = torch.round(weights / scales).clamp(self.qmin, self.qmax)

        return quantized, scales

    def dequantize_weights(self, quantized_weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Dequantize weights back to floating point.

        Args:
            quantized_weights: Quantized weight tensor
            scales: Per-channel scale factors used during quantization

        Returns:
            Dequantized weight tensor
        """
        return quantized_weights * scales


class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, bit_width: int = 8):
        """Initialize quantized linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
            bit_width: Quantization bit width
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize quantizer
        self.quantizer = SymmetricMinMaxQuantizer(bit_width)

        # Store quantized weights and per-channel scales
        self.register_buffer('quantized_weight', None)
        self.register_buffer('weight_scales', None)
        self.quantized = False

    def quantize(self):
        """Quantize the weights."""
        with torch.no_grad():
            quantized_weight, scales = self.quantizer.quantize_weights(self.weight)
            self.quantized_weight = quantized_weight
            self.weight_scales = scales
            self.quantized = True

    def dequantize(self):
        """Dequantize the weights."""
        if self.quantized and self.quantized_weight is not None:
            with torch.no_grad():
                dequantized = self.quantizer.dequantize_weights(self.quantized_weight, self.weight_scales)
                self.weight.data = dequantized
        self.quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.quantized and self.quantized_weight is not None:
            # Use quantized weights
            weight = self.quantizer.dequantize_weights(self.quantized_weight, self.weight_scales)
        else:
            # Use original weights
            weight = self.weight

        return F.linear(x, weight, self.bias)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, bit_width: int = 8) -> 'QuantizedLinear':
        """Create a QuantizedLinear from an existing Linear layer.

        Args:
            linear_layer: Existing linear layer
            bit_width: Quantization bit width

        Returns:
            QuantizedLinear layer with copied weights
        """
        quantized_layer = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            linear_layer.bias is not None,
            bit_width
        )

        # Copy weights and bias
        with torch.no_grad():
            quantized_layer.weight.copy_(linear_layer.weight)
            if linear_layer.bias is not None:
                quantized_layer.bias.copy_(linear_layer.bias)

        return quantized_layer


class QuantizedConv1D(nn.Module):
    """Conv1D layer with quantized weights (for GPT-2 compatibility)."""

    def __init__(self, nf: int, nx: int, bit_width: int = 8):
        """Initialize quantized Conv1D layer.

        Args:
            nf: Number of output features
            nx: Number of input features
            bit_width: Quantization bit width
        """
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.bit_width = bit_width

        # Initialize weights and bias (following Conv1D convention)
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.randn(nf))

        # Initialize quantizer
        self.quantizer = SymmetricMinMaxQuantizer(bit_width)

        # Store quantized weights and per-channel scales
        self.register_buffer('quantized_weight', None)
        self.register_buffer('weight_scales', None)
        self.quantized = False

    def quantize(self):
        """Quantize the weights."""
        with torch.no_grad():
            weight_transposed = self.weight.t()
            quantized_weight, scales = self.quantizer.quantize_weights(weight_transposed)
            self.quantized_weight = quantized_weight.t()
            self.weight_scales = scales
            self.quantized = True

    def dequantize(self):
        """Dequantize the weights."""
        if self.quantized and self.quantized_weight is not None:
            with torch.no_grad():
                # Transpose for dequantization
                quantized_transposed = self.quantized_weight.t()
                dequantized = self.quantizer.dequantize_weights(quantized_transposed, self.weight_scales)
                self.weight.data = dequantized.t()
        self.quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.quantized and self.quantized_weight is not None:
            # Use quantized weights
            quantized_transposed = self.quantized_weight.t()
            weight_transposed = self.quantizer.dequantize_weights(quantized_transposed, self.weight_scales)
            weight = weight_transposed.t()
        else:
            # Use original weights
            weight = self.weight

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), weight)
        x = x.view(size_out)
        return x

    @classmethod
    def from_conv1d(cls, conv1d_layer: Conv1D, bit_width: int = 8) -> 'QuantizedConv1D':
        """Create a QuantizedConv1D from an existing Conv1D layer.

        Args:
            conv1d_layer: Existing Conv1D layer
            bit_width: Quantization bit width

        Returns:
            QuantizedConv1D layer with copied weights
        """
        quantized_layer = cls(conv1d_layer.nf, conv1d_layer.nx, bit_width)

        # Copy weights and bias
        with torch.no_grad():
            quantized_layer.weight.copy_(conv1d_layer.weight)
            quantized_layer.bias.copy_(conv1d_layer.bias)

        return quantized_layer
