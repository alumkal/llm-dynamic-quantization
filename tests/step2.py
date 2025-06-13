# filepath: /home/alumkal/storage/llm-dynamic-quantization/tests/step2.py
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.quant import QuantizationConfig
from src.switchable_precision import SwitchablePrecisionConfig, LoRAConfig, SwitchablePrecisionGPT2Model


def test_switchable_precision():
    """Test switchable precision quantization with different configurations."""

    # Load original model
    original_model = GPT2LMHeadModel.from_pretrained("gpt2")
    original_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Test input
    test_text = "Artificial intelligence (AI) is the capability of computational systems"
    test_input = tokenizer(test_text, return_tensors="pt").input_ids

    with torch.no_grad():
        original_output = original_model(test_input).logits

    # Configure different precision settings
    precision_settings = {
        "16": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=16,
                mlp_bits=16,
                lm_head_bits=16,
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "8": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=8,
                mlp_bits=8,
                lm_head_bits=16,
            ),
            LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        ),
        "4": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4,
                mlp_bits=4,
                lm_head_bits=8,
            ),
            LoRAConfig(r=32, lora_alpha=64, lora_dropout=0.1)
        ),
    }
    switchable_config = SwitchablePrecisionConfig(precision_settings)

    # Create switchable precision model
    model = SwitchablePrecisionGPT2Model(original_model, switchable_config)
    model.eval()

    # Print model summary
    active_layer_count, active_param_count = 0, 0
    for param in model.model.parameters():
        if param.requires_grad:
            active_layer_count += 1
            active_param_count += param.numel()
    print(f"Active layers with trainable parameters: {active_layer_count}")
    print(f"Total trainable parameters: {active_param_count}")

    # Test each precision setting
    for setting_name, (quant_config, lora_config) in precision_settings.items():
        print(f"\nTesting {setting_name}:")
        model.set_precision(setting_name)

        with torch.no_grad():
            quantized_output = model.forward(test_input).logits

        # Calculate KL divergence
        kl = F.kl_div(
            F.log_softmax(quantized_output, dim=-1),
            F.log_softmax(original_output, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        # Get compression ratio
        compression = model.calculate_compression_ratio()

        print(f"Precision = {setting_name}: Size = {compression:.2f}x, KL Divergence = {kl.item():.6f}")


if __name__ == "__main__":
    test_switchable_precision()
