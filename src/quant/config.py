"""Configuration for quantization settings."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class QuantizationConfig:
    """Configuration for per-layer quantization settings.

    The main configuration is a dictionary mapping layer names to bit-widths:
    {'transformer.h.11.mlp.c_proj': 8, 'transformer.h.0.attn.c_attn': 4, ...}
    """

    # Per-layer bit width configuration
    # Key: exact layer name, Value: bit width
    layer_bit_widths: Dict[str, int] = field(default_factory=dict)

    # Default bit width for layers not specified in layer_bit_widths
    default_bit_width: Optional[int] = None

    def get_bit_width_for_layer(self, layer_name: str) -> Optional[int]:
        """Get bit width for a specific layer.

        Args:
            layer_name: Exact name of the layer (e.g., 'transformer.h.0.mlp.c_proj')

        Returns:
            Bit width for the layer
        """
        return self.layer_bit_widths.get(layer_name, self.default_bit_width)

    def should_quantize_layer(self, layer_name: str) -> bool:
        """Check if a layer should be quantized.

        Args:
            layer_name: Name of the layer

        Returns:
            Whether the layer should be quantized (True if specified in config or using default)
        """
        return self.get_bit_width_for_layer(layer_name) is not None

    @classmethod
    def create_uniform_config(cls, bit_width: int) -> 'QuantizationConfig':
        """Create a config with uniform bit width for all layers.

        Args:
            bit_width: Bit width to use for all layers

        Returns:
            QuantizationConfig with uniform bit width
        """
        return cls(default_bit_width=bit_width)

    @classmethod
    def from_dict(cls, layer_dict: Dict[str, int], default_bit_width: int = 32) -> 'QuantizationConfig':
        """Create a config from a dictionary of layer names to bit widths.

        Args:
            layer_dict: Dictionary mapping layer names to bit widths
            default_bit_width: Default bit width for unlisted layers

        Returns:
            QuantizationConfig with specified layer configurations
        """
        return cls(layer_bit_widths=layer_dict.copy(), default_bit_width=default_bit_width)

    @classmethod
    def create_mixed_precision_config(cls,
                                    attention_bits: Optional[int] = None,
                                    mlp_bits: Optional[int] = None,
                                    lm_head_bits: Optional[int] = None,
                                    default_bit_width: Optional[int] = None) -> 'QuantizationConfig':
        """Create a mixed precision configuration for specific model layers.

        Args:
            model_layers: List of all quantizable layer names in the model
            attention_bits: Bit width for attention layers
            mlp_bits: Bit width for MLP layers

        Returns:
            QuantizationConfig with mixed precision settings
        """
        layer_bit_widths = {}
        for layer_idx in range(12):
            layer_name = f'transformer.h.{layer_idx}'
            layer_bit_widths[f'{layer_name}.attn.c_attn'] = attention_bits
            layer_bit_widths[f'{layer_name}.attn.c_proj'] = attention_bits
            layer_bit_widths[f'{layer_name}.mlp.c_fc'] = mlp_bits
            layer_bit_widths[f'{layer_name}.mlp.c_proj'] = mlp_bits
        layer_bit_widths['lm_head'] = lm_head_bits

        return cls(layer_bit_widths=layer_bit_widths, default_bit_width=default_bit_width)
