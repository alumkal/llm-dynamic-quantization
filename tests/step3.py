"""Step 3: Train GPT-2 with 4 switchable precision configurations (16, 8, 6, 4 bits)."""

from pathlib import Path
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.quant import QuantizationConfig
from src.switchable_precision import SwitchablePrecisionConfig, LoRAConfig, SwitchablePrecisionGPT2Model
from src.train import SwitchablePrecisionTrainer
from src.eval import SQuADEvaluator


def main():
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cuda:0")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Configure precision settings (sorted from highest to lowest quality)
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
                lm_head_bits=16,
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "6bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=6,
                mlp_bits=6,
                lm_head_bits=8,
            ),
            LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1)
        ),
        "5bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=5,
                mlp_bits=4,
                lm_head_bits=5,
            ),
            LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        ),
        "4bit": (
            QuantizationConfig.create_mixed_precision_config(
                attention_bits=4,
                mlp_bits=4,
                lm_head_bits=4,
            ),
            LoRAConfig(r=16, lora_alpha=64, lora_dropout=0.1)
        ),
    }
    switchable_config = SwitchablePrecisionConfig(precision_settings)
    print("Configurations:", ", ".join(precision_settings.keys()))

    # Create switchable precision model
    sp_model = SwitchablePrecisionGPT2Model(model, switchable_config)
    del model

    # Create trainer with ordered precision settings
    trainer = SwitchablePrecisionTrainer(
        model=sp_model,
        tokenizer=tokenizer,
        precision_settings=list(precision_settings.keys()),
        learning_rate=2e-4,
        batch_size=2
    )

    # Train the model with cascade distillation
    trainer.train(num_iterations=1000)

    # Evaluate each precision setting
    evaluator = SQuADEvaluator(tokenizer)
    results = {}
    for setting_name in precision_settings:
        sp_model.set_precision(setting_name)
        results[setting_name] = evaluator.evaluate_precision_setting(sp_model)

    # Print summary table
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"{'Setting':<10} {'Loss':<8} {'Accuracy':<10} {'Compression'}")
    print("=" * 50)
    for setting_name, result in results.items():
        print(f"{setting_name:<10} {result['loss']:<8.4f} "
              f"{result['accuracy']:<10.2%} {result['compression']:.2%}")
    print("=" * 50)

if __name__ == "__main__":
    main()

'''
No cascade:
==================================================
Setting    Loss     Accuracy   Compression
==================================================
16bit      0.8864   42.00%     61.32%
8bit       0.9050   39.40%     47.93%
6bit       0.9084   41.60%     38.63%
5bit       1.0301   38.50%     33.62%
4bit       1.2484   33.30%     32.31%
==================================================

With cascade:
==================================================
Setting    Loss     Accuracy   Compression
==================================================
16bit      0.8640   42.30%     61.32%
8bit       0.8496   44.50%     47.93%
6bit       0.8438   43.80%     38.63%
5bit       0.9337   41.20%     33.62%
4bit       1.0643   36.50%     32.31%
==================================================
'''
