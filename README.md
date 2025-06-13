# Efficient LLMs via Switchable and Dynamic Quantization

## Project Overview
This project implements switchable and dynamic quantization schemes for Large Language Models (LLMs) to improve the accuracy-efficiency trade-off. The implementation focuses on GPT-2 with per-layer quantization and adaptive LoRA modules.

## Completed Steps

### ✅ Step 1: Quantization Integration
- Implemented symmetric MinMax quantization for GPT-2
- Support for different bit-widths per layer
- Quantized linear layers with configurable precision

### ✅ Step 2: Multi-LoRA Integration  
- Added multiple LoRA modules to all linear layers
- Adaptive activation of different LoRA modules based on quantization configuration
- Switchable precision model with multiple quantization settings

### ✅ Step 3: Switchable Precision Training
- Implemented training with multiple quantization configurations simultaneously
- Random precision switching during training iterations
- Comprehensive evaluation on SQuAD dataset
- Achieved meaningful compression ratios (0.353x - 0.613x) with reasonable performance

## Usage

### Installation
```bash
uv sync
```

### Running Tests
```bash
# Step 1: Basic quantization
uv run python tests/step1.py

# Step 2: Switchable precision
uv run python tests/step2.py

# Step 3: Training with switchable precision
uv run python tests/step3.py
```

## Project Structure
```
src/
├── quant/                    # Basic quantization implementation
│   ├── config.py            # Quantization configuration
│   ├── model.py             # Quantized GPT-2 model
│   └── quantization.py      # Quantization algorithms
├── switchable_precision/     # Switchable precision implementation
│   ├── config.py            # Switchable precision configuration
│   ├── layers.py            # Multi-LoRA quantized layers
│   └── model.py             # Switchable precision model
├── training.py              # Training utilities
└── evaluation.py            # Evaluation utilities

tests/
├── step1.py                 # Step 1 testing
├── step2.py                 # Step 2 testing
└── step3.py                 # Step 3 training and evaluation
```

## Key Features
- **Per-layer Quantization**: Different bit-widths for different layers
- **Multi-LoRA Support**: Multiple LoRA modules for different precision settings
- **Switchable Training**: Random precision switching during training
- **Comprehensive Evaluation**: Performance assessment on SQuAD dataset
- **Minimal Dependencies**: Uses uv for fast package management

## Results Summary (Step 3)
| Precision | Compression | Perplexity | Training Loss |
|-----------|-------------|------------|---------------|
| High (16-bit) | 0.613x | 17.44 | 3.73 |
| Medium (8-bit) | 0.479x | 17.52 | 3.78 |
| Low (4-bit) | 0.353x | 39.16 | 5.06 |

## Next Steps
- **Step 4**: Comprehensive evaluation and optimal configuration discovery
- **Step 5**: Cyclic precision training implementation  
- **Step 6**: Adversarial robustness evaluation
