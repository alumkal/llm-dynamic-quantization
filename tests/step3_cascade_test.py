"""Quick test of cascade distillation training."""

from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.quant import QuantizationConfig
from src.switchable_precision import SwitchablePrecisionConfig, LoRAConfig, SwitchablePrecisionGPT2Model
from src.training import SwitchablePrecisionTrainer


def main():
    """Quick test of cascade training."""
    print("Testing Cascade Distillation Training")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Configure precision settings
    precision_settings = {
        "high": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=16, mlp_bits=16, lm_head_bits=16
            ),
            LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.1)  # Smaller LoRA for faster training
        ),
        "low": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4, mlp_bits=4, lm_head_bits=8
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
    }
    
    # Create model
    switchable_config = SwitchablePrecisionConfig(precision_settings)
    sp_model = SwitchablePrecisionGPT2Model(model, switchable_config)
    
    # Create trainer
    trainer = SwitchablePrecisionTrainer(
        model=sp_model,
        tokenizer=tokenizer,
        precision_settings=list(precision_settings.keys()),
        learning_rate=1e-4,
        batch_size=1,  # Very small batch for quick test
        max_length=128
    )
    
    print(f"Trainable parameters: {sum(p.numel() for p in sp_model.parameters() if p.requires_grad):,}")
    
    # Test cascade training for just 3 iterations
    print("\nTesting cascade training...")
    avg_loss = trainer.train(num_iterations=3, use_cascade=True)
    
    print(f"\nCascade training completed with average loss: {avg_loss:.4f}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    sp_model.eval()
    test_text = "The quick brown fox"
    test_input = tokenizer(test_text, return_tensors="pt")
    
    for setting in precision_settings.keys():
        sp_model.set_precision(setting)
        with torch.no_grad():
            outputs = sp_model(**test_input)
            compression = sp_model.calculate_compression_ratio()
        print(f"{setting}: compression={compression:.3f}x")
    
    print("\nCascade training test completed successfully!")


if __name__ == "__main__":
    main()