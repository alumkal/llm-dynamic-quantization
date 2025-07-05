"""Configuration for cyclic precision training."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


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

    @classmethod
    def create_layerwise_config(cls,
                                attention_bits: List[Optional[int]],
                                mlp_bits: List[Optional[int]],
                                lm_head_bits: Optional[int],
                                default_bit_width: Optional[int] = None) -> 'QuantizationConfig':
        """Create a layer-wise configuration for specific model layers.
        Args:
            attention_bits: List of bit widths for attention layers
            mlp_bits: List of bit widths for MLP layers
            lm_head_bits: Bit width for the language model head
            default_bit_width: Default bit width for unlisted layers
        Returns:
            QuantizationConfig with layer-wise settings
        """
        def get_bit_width(lst: List[Optional[int]], idx: int) -> Optional[int]:
            return lst[idx // (12 // len(lst))]

        layer_bit_widths = {}
        for layer_idx in range(12):
            layer_name = f'transformer.h.{layer_idx}'
            layer_bit_widths[f'{layer_name}.attn.c_attn'] = get_bit_width(attention_bits, layer_idx)
            layer_bit_widths[f'{layer_name}.attn.c_proj'] = get_bit_width(attention_bits, layer_idx)
            layer_bit_widths[f'{layer_name}.mlp.c_fc'] = get_bit_width(mlp_bits, layer_idx)
            layer_bit_widths[f'{layer_name}.mlp.c_proj'] = get_bit_width(mlp_bits, layer_idx)
        layer_bit_widths['lm_head'] = lm_head_bits
        return cls(layer_bit_widths=layer_bit_widths, default_bit_width=default_bit_width)


class CyclicPrecisionScheduler:
    """Scheduler for cyclic precision training.
    
    Implements a cosine scheduler that cycles between min and max bit widths
    during training as described in the CPT paper.
    """
    
    def __init__(
        self,
        min_bits: int = 2,
        max_bits: int = 8,
        num_cycles: int = 1,
        total_steps: int = 1000,
    ):
        """Initialize the cyclic precision scheduler.
        
        Args:
            min_bits: Minimum bit width to use during training
            max_bits: Maximum bit width to use during training
            num_cycles: Number of cycles to complete during training
            total_steps: Total number of training steps
        """
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.num_cycles = num_cycles
        self.total_steps = total_steps
        
    def get_bit_width(self, step: int) -> int:
        """Get the bit width for the current training step.
        
        Args:
            step: Current training step
            
        Returns:
            Bit width to use for the current step
        """
        # Calculate the position within the cycle (0 to 1)
        cycle_length = self.total_steps / self.num_cycles
        cycle_position = (step % cycle_length) / cycle_length
        
        # Use cosine schedule to determine bit width
        # Cosine goes from 1 to -1, so we adjust to go from min_bits to max_bits
        cosine_value = math.cos(cycle_position * math.pi)
        bit_range = self.max_bits - self.min_bits
        
        # Scale cosine from [-1, 1] to [0, 1] and then to [min_bits, max_bits]
        bit_width = self.min_bits + bit_range * (cosine_value + 1) / 2
        
        # Round to nearest integer
        return round(bit_width)
