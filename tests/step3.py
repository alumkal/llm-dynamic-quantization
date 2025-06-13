"""Step 3: Train GPT-2 with switchable precision on SQuAD dataset."""

from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.quant import QuantizationConfig
from src.switchable_precision import SwitchablePrecisionConfig, LoRAConfig, SwitchablePrecisionGPT2Model
from src.training import SwitchablePrecisionTrainer
from src.evaluation import SQuADEvaluator


def main():
    """Main training function for step 3."""
    print("Step 3: Training GPT-2 with switchable precision quantization on SQuAD")
    
    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Configure different precision settings for switchable training
    precision_settings = {
        "high": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=16,
                mlp_bits=16,
                lm_head_bits=16,
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "medium": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=8,
                mlp_bits=8,
                lm_head_bits=16,
            ),
            LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        ),
        "low": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4,
                mlp_bits=4,
                lm_head_bits=8,
            ),
            LoRAConfig(r=32, lora_alpha=64, lora_dropout=0.1)
        ),
    }
    
    # Create switchable precision config
    switchable_config = SwitchablePrecisionConfig(precision_settings)
    
    # Create switchable precision model
    print("Creating switchable precision model...")
    sp_model = SwitchablePrecisionGPT2Model(model, switchable_config)
    
    # Print model info
    trainable_params = sum(p.numel() for p in sp_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sp_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.4f}")
    
    # Create trainer
    print("Setting up trainer...")
    trainer = SwitchablePrecisionTrainer(
        model=sp_model,
        tokenizer=tokenizer,
        precision_settings=list(precision_settings.keys()),
        learning_rate=5e-5,
        batch_size=2,  # Small batch size for CPU training
        max_length=256  # Shorter sequences for faster training
    )
    
    # Train the model with cascade distillation
    print("Starting training with cascade distillation...")
    avg_loss_cascade = trainer.train(num_iterations=10, use_cascade=True)
    
    print(f"\nCascade training completed with average loss: {avg_loss_cascade:.4f}")
    
    # Compare with original random precision training
    print("\nComparing with original random precision training...")
    
    # Reset model to original state for fair comparison
    sp_model_original = SwitchablePrecisionGPT2Model(model, switchable_config)
    trainer_original = SwitchablePrecisionTrainer(
        model=sp_model_original,
        tokenizer=tokenizer,
        precision_settings=list(precision_settings.keys()),
        learning_rate=5e-5,
        batch_size=2,
        max_length=256
    )
    
    avg_loss_original = trainer_original.train(num_iterations=10, use_cascade=False)
    print(f"Original training completed with average loss: {avg_loss_original:.4f}")
    
    print(f"\nCascade vs Original: {avg_loss_cascade:.4f} vs {avg_loss_original:.4f}")
    print(f"Improvement: {((avg_loss_original - avg_loss_cascade) / avg_loss_original * 100):.2f}%")
    
    # Evaluate different precision settings after training
    print("\nEvaluating different precision settings after training:")
    evaluator = SQuADEvaluator(tokenizer, max_length=256)
    results = evaluator.evaluate_all_settings(sp_model, list(precision_settings.keys()))
    
    print("\nEvaluation Results:")
    print("=" * 60)
    print(f"{'Setting':<10} {'Loss':<8} {'Perplexity':<12} {'Compression':<12}")
    print("=" * 60)
    
    for setting_name, metrics in results.items():
        print(f"{setting_name:<10} {metrics['loss']:<8.4f} {metrics['perplexity']:<12.2f} {metrics['compression']:<12.3f}")
    
    print("=" * 60)
    print("\nStep 3 completed successfully!")
    print("\nSummary:")
    print("- Successfully implemented switchable precision training")
    print("- Trained with multiple quantization configurations simultaneously")
    print("- Different LoRA modules activated for different bit-width configurations")
    print("- Lower bit-widths achieve higher compression but may have higher perplexity")


if __name__ == "__main__":
    main()