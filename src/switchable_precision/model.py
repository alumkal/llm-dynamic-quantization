"""GPT-2 model with switchable precision layers."""

import copy
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from typing import Optional, Dict, List

from .layers import SwitchablePrecisionLinear
from .config import SwitchablePrecisionConfig


class SwitchablePrecisionGPT2Model(nn.Module):
    """Wrapper for GPT-2 model with quantization support."""

    def __init__(self, model: GPT2LMHeadModel, switchable_precision_config: SwitchablePrecisionConfig):
        """Initialize quantized GPT-2 model.

        Args:
            model_name: Name of the GPT-2 model to load
            quantization_config: Configuration for quantization
        """
        super().__init__()

        # Load the original model and freeze its parameters
        self.model = copy.deepcopy(model)
        self.model.eval()  # Set to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False

        # Track quantized layers
        self.quantized_layers: Dict[str, SwitchablePrecisionLinear] = {}

        # Setup quantization
        self.switchable_precision_config = switchable_precision_config
        self._setup_quantization()

    def _setup_quantization(self):
        """Setup quantization by replacing layers."""
        # Get LoRA configurations from the switchable precision config
        lora_configs = {
            name: config[1] for name, config in self.switchable_precision_config.settings.items()
        }

        # Identify linear and Conv1D layers to quantize
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, Conv1D):
                # Create quantized layer
                quantized_layer = SwitchablePrecisionLinear(module, lora_configs)

                # Replace the original layer with the quantized version
                self.model.set_submodule(name, quantized_layer, strict=True)
                self.quantized_layers[name] = quantized_layer

    def set_precision(self, setting_name: str):
        """Set a new quantization configuration.

        Args:
            quantization_config: New quantization configuration to apply
        """
        if setting_name not in self.switchable_precision_config.settings:
            raise ValueError(f"Setting '{setting_name}' not found in switchable precision config.")
        quant_config, _ = self.switchable_precision_config.settings[setting_name]

        # Reapply quantization with the new config
        for layer_name, layer in self.quantized_layers.items():
            bit_width = quant_config.get_bit_width_for_layer(layer_name)
            layer.set_quant_config(bit_width)
            layer.set_lora_config(setting_name)

    def get_quantizable_layer_names(self) -> List[str]:
        """Get all quantizable layer names from a GPT-2 model.

        Args:
            None

        Returns:
            List of quantizable layer names
        """
        return list(self.quantized_layers.keys())

    def calculate_compression_ratio(self) -> float:
        """Estimate the compression ratio of the quantized model.

        Returns:
            Compression ratio as a float, where 1.0 means no savings
        """
        saved_bits = 0

        for layer in self.quantized_layers.values():
            params = sum(param.numel() for param in layer.parameters())
            if layer.bit_width is not None:
                saved_bits += params * (32 - layer.bit_width)

        total_params = sum(param.numel() for param in self.model.parameters())
        total_bits = total_params * 32  # Assuming 32 bits per float

        return (total_bits - saved_bits) / total_bits

    def forward(self, *args, **kwargs):
        """Forward pass through the quantized model.

        Args:
            *args: Arguments passed to the model
            **kwargs: Keyword arguments passed to the model

        Returns:
            Model output
        """
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set the model to training mode.

        Args:
            training: Whether to set the model to training mode (default: True)
        """
        self.model.train(mode)

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
