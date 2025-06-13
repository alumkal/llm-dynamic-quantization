"""Step 3: Train GPT-2 with 4 switchable precision configurations (16, 8, 6, 4 bits)."""

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
    """Main training function for step 3 with 4 configurations."""
    print("Step 3: Training GPT-2 with 4 switchable precision configurations")
    print("Configurations: 16-bit, 8-bit, 6-bit, 4-bit (sorted from highest to lowest)")
    
    # Load model and tokenizer
    print("Loading GPT-2 model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Configure 4 precision settings (sorted from highest to lowest quality)
    # Give more bits for LM Head as requested
    precision_settings = {
        "16bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=16,
                mlp_bits=16,
                lm_head_bits=16,
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "8bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=8,
                mlp_bits=8,
                lm_head_bits=16,  # Higher bits for LM head
            ),
            LoRAConfig(r=12, lora_alpha=24, lora_dropout=0.1)
        ),
        "6bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=6,
                mlp_bits=6,
                lm_head_bits=12,  # Higher bits for LM head
            ),
            LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        ),
        "4bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4,
                mlp_bits=4,
                lm_head_bits=8,   # Higher bits for LM head
            ),
            LoRAConfig(r=24, lora_alpha=48, lora_dropout=0.1)
        ),
    }
    
    # Create switchable precision config with ordered settings
    # Order matters for cascade training (highest to lowest precision)
    ordered_settings = ["16bit", "8bit", "6bit", "4bit"]
    switchable_config = SwitchablePrecisionConfig({
        name: precision_settings[name] for name in ordered_settings
    })
    
    # Create switchable precision model
    print("Creating switchable precision model...")
    sp_model = SwitchablePrecisionGPT2Model(model, switchable_config)
    
    # Print model info
    trainable_params = sum(p.numel() for p in sp_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sp_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.4f}")
    
    # Create trainer with ordered precision settings
    print("Setting up trainer...")
    trainer = SwitchablePrecisionTrainer(
        model=sp_model,
        tokenizer=tokenizer,
        precision_settings=ordered_settings,  # Pass ordered list
        learning_rate=5e-5,
        batch_size=2,  # Small batch size for CPU training
        max_length=256  # Shorter sequences for faster training
    )
    
    # Train the model with cascade distillation
    print("Starting training with cascade distillation...")
    avg_loss_cascade = trainer.train(num_iterations=10, use_cascade=True)
    
    print(f"\nCascade training completed with average loss: {avg_loss_cascade:.4f}")
    
    # Evaluate different precision settings after training
    print("\nEvaluating different precision settings after training:")
    evaluator = SQuADEvaluator(tokenizer, max_length=256)
    
    results = []
    for setting_name in ordered_settings:
        # Evaluate on small subset
        eval_results = evaluator.evaluate_precision_setting(sp_model, setting_name)
        
        results.append({
            'setting': setting_name,
            'loss': eval_results['loss'],
            'perplexity': eval_results['perplexity'],
            'compression': eval_results['compression']
        })
        
        print(f"{setting_name}: loss={eval_results['loss']:.4f}, "
              f"perplexity={eval_results['perplexity']:.2f}, "
              f"compression={eval_results['compression']:.3f}x")
    
    # Print summary table
    print("\nEvaluation Results:")
    print("=" * 60)
    print(f"{'Setting':<10} {'Loss':<8} {'Perplexity':<12} {'Compression'}")
    print("=" * 60)
    for result in results:
        print(f"{result['setting']:<10} {result['loss']:<8.4f} "
              f"{result['perplexity']:<12.2f} {result['compression']:.3f}")
    print("=" * 60)
    
    print("\nStep 3 completed successfully!")
    
    print("\nSummary:")
    print("- Successfully implemented cascade distillation training")
    print("- Trained with 4 quantization configurations simultaneously")
    print("- Different LoRA modules activated for different bit-width configurations")
    print("- Higher bits allocated to LM head for better performance")
    print("- Cascade training enables knowledge transfer from higher to lower precision models")


if __name__ == "__main__":
    main()