"""Quick test of 4-configuration cascade training."""

from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.quant import QuantizationConfig
from src.switchable_precision import SwitchablePrecisionConfig, LoRAConfig, SwitchablePrecisionGPT2Model
from src.training import SwitchablePrecisionTrainer


def main():
    """Quick test of 4-configuration cascade training."""
    print("Quick Test: 4-Configuration Cascade Training")
    print("Configurations: 16-bit, 8-bit, 6-bit, 4-bit (sorted from highest to lowest)")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Configure 4 precision settings with higher bits for LM Head
    precision_settings = {
        "16bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=16, mlp_bits=16, lm_head_bits=16
            ),
            LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.1)
        ),
        "8bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=8, mlp_bits=8, lm_head_bits=16  # Higher bits for LM head
            ),
            LoRAConfig(r=6, lora_alpha=12, lora_dropout=0.1)
        ),
        "6bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=6, mlp_bits=6, lm_head_bits=12  # Higher bits for LM head
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "4bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4, mlp_bits=4, lm_head_bits=8   # Higher bits for LM head
            ),
            LoRAConfig(r=12, lora_alpha=24, lora_dropout=0.1)
        ),
    }
    
    # Create model with ordered settings (highest to lowest precision)
    ordered_settings = ["16bit", "8bit", "6bit", "4bit"]
    switchable_config = SwitchablePrecisionConfig({
        name: precision_settings[name] for name in ordered_settings
    })
    
    sp_model = SwitchablePrecisionGPT2Model(model, switchable_config)
    
    # Print configuration details
    print(f"\nModel Configuration:")
    print(f"Total parameters: {sum(p.numel() for p in sp_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in sp_model.parameters() if p.requires_grad):,}")
    
    print(f"\nPrecision Settings:")
    for setting in ordered_settings:
        sp_model.set_precision(setting)
        compression = sp_model.calculate_compression_ratio()
        print(f"  {setting}: compression={compression:.3f}x")
    
    # Create trainer
    trainer = SwitchablePrecisionTrainer(
        model=sp_model,
        tokenizer=tokenizer,
        precision_settings=ordered_settings,  # Pass ordered list
        learning_rate=1e-4,
        batch_size=1,
        max_length=128
    )
    
    # Quick training test (just 2 iterations)
    print(f"\nTesting cascade training (2 iterations)...")
    avg_loss = trainer.train(num_iterations=2, use_cascade=True)
    
    print(f"\nCascade training completed with average loss: {avg_loss:.4f}")
    
    # Test inference with different precisions
    print(f"\nTesting inference with different precisions:")
    test_text = "The quick brown fox jumps over"
    test_input = tokenizer(test_text, return_tensors="pt")
    
    sp_model.eval()
    for setting in ordered_settings:
        sp_model.set_precision(setting)
        with torch.no_grad():
            outputs = sp_model(**test_input)
            loss = outputs.loss if hasattr(outputs, 'loss') else 0.0
            compression = sp_model.calculate_compression_ratio()
        print(f"  {setting}: compression={compression:.3f}x")
    
    print(f"\n✅ 4-Configuration cascade training test completed successfully!")
    
    print(f"\nKey Features Demonstrated:")
    print(f"- 4 different quantization configurations (16, 8, 6, 4 bits)")
    print(f"- Higher bit allocation for LM head components")
    print(f"- Cascade distillation training without hard-coded configuration names")
    print(f"- Dynamic progress bar showing all configuration losses")
    print(f"- Ordered precision settings (highest to lowest)")


if __name__ == "__main__":
    main()