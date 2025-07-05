"""Quantized GPT-2 model with cyclic precision training support."""

import copy
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from typing import Optional, Dict, List

from .quantization import QuantizedLinear
from .config import QuantizationConfig, CyclicPrecisionScheduler


class QuantizedGPT2Model(nn.Module):
    """Wrapper for GPT-2 model with cyclic precision training support."""

    def __init__(self, model: GPT2LMHeadModel, quantization_config: Optional[QuantizationConfig] = None):
        """Initialize quantized GPT-2 model.

        Args:
            model: GPT-2 model to quantize
            quantization_config: Configuration for quantization
        """
        super().__init__()

        # Load the original model
        self.model = copy.deepcopy(model)

        # Track quantized layers
        self.quantized_layers: Dict[str, QuantizedLinear] = {}

        # Setup quantization
        self._setup_quantization()

        # Set quantization configuration
        if quantization_config is None:
            quantization_config = QuantizationConfig()
        self.set_quantization_config(quantization_config)

        # Initialize cyclic precision scheduler
        self.cpt_scheduler = None
        self.current_step = 0

    def _setup_quantization(self):
        """Setup quantization by replacing layers."""
        # Identify linear and Conv1D layers to quantize
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, Conv1D):
                # Create quantized layer
                quantized_layer = QuantizedLinear(module)

                # Replace the original layer with the quantized version
                self.model.set_submodule(name, quantized_layer, strict=True)
                self.quantized_layers[name] = quantized_layer

    def set_quantization_config(self, quantization_config: QuantizationConfig):
        """Set a new quantization configuration.

        Args:
            quantization_config: New quantization configuration to apply
        """
        self.quantization_config = quantization_config

        # Reapply quantization with the new config
        for layer_name, layer in self.quantized_layers.items():
            bit_width = self.quantization_config.get_bit_width_for_layer(layer_name)
            layer.set_quant_config(bit_width)

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

    def setup_cyclic_precision_training(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        num_cycles: int = 1,
        total_steps: int = 1000,
    ):
        """Setup cyclic precision training.

        Args:
            min_bits: Minimum bit width to use during training
            max_bits: Maximum bit width to use during training
            num_cycles: Number of cycles to complete during training
            total_steps: Total number of training steps
        """
        self.cpt_scheduler = CyclicPrecisionScheduler(
            min_bits=min_bits,
            max_bits=max_bits,
            num_cycles=num_cycles,
            total_steps=total_steps,
        )

    def step_cyclic_precision(self):
        """Update bit width based on the current training step."""
        if self.cpt_scheduler is None:
            return

        # Get the bit width for the current step
        bit_width = self.cpt_scheduler.get_bit_width(self.current_step)

        # Update all layers with the new bit width
        for layer in self.quantized_layers.values():
            layer.set_quant_config(bit_width)

        # Increment step counter
        self.current_step += 1

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
        """Set the model to training mode."""
        self.model.train(mode)

    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()

    @property
    def device(self):
        """Get the device of the model."""
        return self.model.device
