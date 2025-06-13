"""Switchable precision linear layer with multi-LoRA support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from typing import Optional, Dict

from ..quant.quantization import SymmetricMinMaxQuantizer
from .config import LoRAConfig


class SwitchablePrecisionLinear(nn.Module):
    """Linear layer with switchable precision quantization and multi-LoRA support."""

    def __init__(self, module: nn.Module, lora_configs: Dict[str, LoRAConfig]):
        """Initialize switchable precision linear layer.

        Args:
            module: Original linear layer to quantize
        """
        super().__init__()

        # Initialize weights and bias
        if isinstance(module, nn.Linear):
            self.weight = nn.Parameter(module.weight, requires_grad=False)
            self.bias = module.bias
        elif isinstance(module, Conv1D):
            self.weight = nn.Parameter(module.weight.t(), requires_grad=False)
            self.bias = module.bias

        # Initialize quantizer
        self.quantizer = SymmetricMinMaxQuantizer(self.weight)
        if self.bias is not None:
            self.bias_quantizer = SymmetricMinMaxQuantizer(self.bias)

        # Initialize LoRA configurations
        self.lora_configs = lora_configs
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.lora_dropout = nn.ModuleDict()

        # Initialize LoRA layers
        # https://github.com/microsoft/LoRA/blob/c4593f060e6a368d7bb5af5273b8e42810cdef90/loralib/layers.py#L124
        out_features, in_features = self.weight.shape
        for name, config in self.lora_configs.items():
            self.lora_a[name] = nn.Parameter(self.weight.new_empty((config.r, in_features)))
            nn.init.kaiming_uniform_(self.lora_a[name], a=5**0.5)
            self.lora_b[name] = nn.Parameter(self.weight.new_empty((out_features, config.r)))
            nn.init.zeros_(self.lora_b[name])
            self.lora_dropout[name] = nn.Dropout(config.lora_dropout)

        self.bit_width = None
        self.activate_lora = None

    def set_quant_config(self, bit_width: Optional[int] = None):
        """Set quantization configuration.

        Args:
            bit_width: Bit width for quantization (default: None, means no quantization)
        """
        self.bit_width = bit_width

    def set_lora_config(self, activate_lora: Optional[str] = None):
        """Set LoRA configuration.

        Args:
            activate_lora: Name of the LoRA to activate (default: None, means no LoRA)
        """
        self.activate_lora = activate_lora


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

        original_output = F.linear(x, quantized_weight, quantized_bias)

        # If no LoRA is activated, return the original output
        if self.activate_lora is None:
            return original_output

        # Load LoRA configuration and parameters
        lora_config = self.lora_configs[self.activate_lora]
        lora_a = self.lora_a[self.activate_lora]
        lora_b = self.lora_b[self.activate_lora]
        scale = lora_config.lora_alpha / lora_config.r
        lora_dropout = self.lora_dropout[self.activate_lora]

        # Apply LoRA
        lora_output = lora_dropout(x @ lora_a.t()) @ lora_b.t() * scale
        return original_output + lora_output
