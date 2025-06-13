from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent / ".." / "src"))
from quant import QuantizationConfig, QuantizedGPT2Model


def test_different_bit_widths():
    """Test different bit widths and their effects."""

    # Load original model
    original_model = GPT2LMHeadModel.from_pretrained("gpt2")
    original_model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Test input
    test_text = "Artificial intelligence (AI) is the capability of computational systems"
    test_input = tokenizer(test_text, return_tensors="pt").input_ids

    with torch.no_grad():
        original_output = original_model(test_input).logits

    # Create quantized model
    model = QuantizedGPT2Model(original_model)

    # Test different bit widths
    for bit_width in [16, 8, 6, 4, 2]:
        # Set quantization config
        quantization_config = QuantizationConfig.create_mixed_precision_config(
            attention_bits=bit_width,
            mlp_bits=bit_width,
            lm_head_bits=16,
        )
        model.set_quantization_config(quantization_config)

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

        print(f"{bit_width}-bit: Size = {compression:.2f}x, KL Divergence = {kl.item():.6f}")

if __name__ == "__main__":
    test_different_bit_widths()
