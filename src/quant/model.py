"""Quantized GPT-2 model implementation."""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from typing import Optional, Dict, List

from .quantization import QuantizedLinear, QuantizedConv1D
from .config import QuantizationConfig


class QuantizedGPT2Model:
    """Wrapper for GPT-2 model with quantization support."""

    def __init__(self, model_name: str = "gpt2", quantization_config: Optional[QuantizationConfig] = None):
        """Initialize quantized GPT-2 model.

        Args:
            model_name: Name of the GPT-2 model to load
            quantization_config: Configuration for quantization
        """
        self.model_name = model_name
        self.quantization_config = quantization_config or QuantizationConfig()

        # Load the original model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.model.config

        # Track quantized layers
        self.quantized_layers = {}
        self.original_layers = {}

        # Apply quantization
        self._apply_quantization()

    def _apply_quantization(self):
        """Apply quantization to the model layers."""
        # Quantize transformer blocks
        for i, block in enumerate(self.model.transformer.h):
            self._quantize_transformer_block(block, i)

        # Quantize output layer (lm_head)
        layer_name = "lm_head"
        if self.quantization_config.should_quantize_layer(layer_name):
            self._quantize_linear_layer(self.model, "lm_head", layer_name)

    def _quantize_transformer_block(self, block, block_index: int):
        """Quantize a single transformer block.

        Args:
            block: Transformer block to quantize
            block_index: Index of the block
        """
        # Quantize attention layers
        attn_c_attn_name = f"transformer.h.{block_index}.attn.c_attn"
        attn_c_proj_name = f"transformer.h.{block_index}.attn.c_proj"

        if self.quantization_config.should_quantize_layer(attn_c_attn_name):
            self._quantize_linear_layer(block.attn, "c_attn", attn_c_attn_name)

        if self.quantization_config.should_quantize_layer(attn_c_proj_name):
            self._quantize_linear_layer(block.attn, "c_proj", attn_c_proj_name)

        # Quantize MLP layers
        mlp_c_fc_name = f"transformer.h.{block_index}.mlp.c_fc"
        mlp_c_proj_name = f"transformer.h.{block_index}.mlp.c_proj"

        if self.quantization_config.should_quantize_layer(mlp_c_fc_name):
            self._quantize_linear_layer(block.mlp, "c_fc", mlp_c_fc_name)

        if self.quantization_config.should_quantize_layer(mlp_c_proj_name):
            self._quantize_linear_layer(block.mlp, "c_proj", mlp_c_proj_name)

    def _quantize_linear_layer(self, parent_module, layer_name: str, full_layer_name: str):
        """Replace a linear layer with a quantized version.

        Args:
            parent_module: Parent module containing the layer
            layer_name: Name of the layer to quantize
            bit_width: Bit width for quantization
            full_layer_name: Full name of the layer for tracking
        """
        original_layer = getattr(parent_module, layer_name)
        quantized_layer = None

        bit_width = self.quantization_config.get_bit_width_for_layer(full_layer_name)
        if isinstance(original_layer, nn.Linear):
            # Create quantized linear layer
            quantized_layer = QuantizedLinear.from_linear(original_layer, bit_width)
        elif isinstance(original_layer, Conv1D):
            # Create quantized Conv1D layer
            quantized_layer = QuantizedConv1D.from_conv1d(original_layer, bit_width)

        if quantized_layer is not None:
            # Replace the layer
            setattr(parent_module, layer_name, quantized_layer)

            # Store references using the full layer name
            self.quantized_layers[full_layer_name] = quantized_layer
            self.original_layers[full_layer_name] = original_layer

    def _get_module_path(self, module):
        """Get the path of a module within the model."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return module.__class__.__name__

    @classmethod
    def get_quantizable_layer_names(cls, model_name: str = "gpt2") -> List[str]:
        """Get all quantizable layer names from a GPT-2 model.

        Args:
            model_name: Name of the GPT-2 model

        Returns:
            List of quantizable layer names
        """
        # Load model temporarily to get layer names
        model = GPT2LMHeadModel.from_pretrained(model_name)
        layer_names = []

        # Get transformer block layers
        for i in range(len(model.transformer.h)):
            layer_names.extend([
                f"transformer.h.{i}.attn.c_attn",
                f"transformer.h.{i}.attn.c_proj",
                f"transformer.h.{i}.mlp.c_fc",
                f"transformer.h.{i}.mlp.c_proj"
            ])

        # Add output layer
        layer_names.append("lm_head")

        return layer_names

    def quantize_weights(self):
        """Quantize all weights in the model."""
        for layer in self.quantized_layers.values():
            layer.quantize()

    def dequantize_weights(self):
        """Dequantize all weights in the model."""
        for layer in self.quantized_layers.values():
            layer.dequantize()

    def estimate_memory_savings(self) -> Dict[str, float]:
        """Estimate memory savings from quantization.

        Returns:
            Dictionary with memory savings information
        """
        quantized_params = 0
        saved_bits = 0

        for layer in self.quantized_layers.values():
            params = layer.weight.numel()
            quantized_params += params
            saved_bits += params * (32 - layer.bit_width)

        total_params = sum(param.numel() for param in self.model.parameters())

        return {
            "total_parameters": total_params,
            "quantized_parameters": quantized_params,
            "compression_ratio": (total_params * 32 - saved_bits) / (total_params * 32),
        }
