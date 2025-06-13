"""Configuration for switchable precision quantization."""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from ..quant.config import QuantizationConfig


@dataclass
class LoRAConfig:
    """Configuration for LoRA settings."""

    r: int = 8  # Rank of the LoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.1

@dataclass
class SwitchablePrecisionConfig:
    """Configuration for switchable precision quantization."""

    # Dictionary mapping layer names to (quantization config, LoRA config)
    settings: Dict[str, Tuple[QuantizationConfig, LoRAConfig]] = field(default_factory=dict)
