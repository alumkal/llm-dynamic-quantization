"""Cyclic Precision Training with SQuAD dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import datasets
from typing import Dict, List, Tuple
import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

# Add the parent directory to the path so we can import the cpt module
sys.path.append(str(Path(__file__).parent.parent))
from src.cpt import QuantizedGPT2Model, QuantizationConfig


def seed_everything(seed: int = 1):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything()


class SQuADDataset(torch.utils.data.Dataset):
    """SQuAD dataset for question answering fine-tuning."""

    def __init__(self, tokenizer: GPT2Tokenizer, split: str = "train"):
        self.tokenizer = tokenizer

        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SQuAD dataset
        self.dataset = datasets.load_dataset("squad", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prefix_text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        prefix_length = len(self.tokenizer.tokenize(prefix_text))
        text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer: {sample['answers']['text'][0]}\n"
        return text, prefix_length


class CPTTrainer:
    """Trainer for Cyclic Precision Training."""

    def __init__(
        self,
        model: QuantizedGPT2Model,
        tokenizer: GPT2Tokenizer,
        learning_rate: float,
        batch_size: int,
        max_length: int
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length

        # Setup dataset
        self.dataset = SQuADDataset(tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )

        # Setup optimizer for all model parameters
        self.optimizer = AdamW(self.model.model.parameters(), lr=learning_rate)

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)

    def _collate_fn(self, batch: List[Tuple[str, int]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        # Unpack batch
        texts, prefix_lengths = zip(*batch)

        # Tokenize text
        tokenized_text = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.model.device)

        # Labels are the same as input_ids, but with prefix tokens masked
        tokenized_text["labels"] = tokenized_text["input_ids"].clone()
        for i, prefix_length in enumerate(prefix_lengths):
            tokenized_text["labels"][i, :prefix_length] = self.tokenizer.pad_token_id

        return tokenized_text

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with cyclic precision."""
        self.model.train()

        # Update bit width based on current step
        self.model.step_cyclic_precision()

        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        # Standard cross-entropy loss
        logits = outputs.logits
        labels = batch["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_iterations: int):
        """Train the model with cyclic precision training."""
        self.model.model.train()

        total_loss = 0

        # Create data iterator
        data_iter = iter(self.dataloader)

        pbar = tqdm(range(num_iterations), desc="Training")
        for iteration in pbar:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset dataloader if we run out of data
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Training step
            step_loss = self.train_step(batch)
            total_loss += step_loss

            # Update progress bar
            avg_loss = total_loss / (iteration + 1)
            bits = self.model.cpt_scheduler.get_bit_width(iteration)
            pbar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "loss": f"{step_loss:.4f}",
                "bits": bits,
            })

        # Print final statistics
        avg_loss = total_loss / num_iterations

        return avg_loss


class SQuADEvaluator:
    """Evaluator for SQuAD dataset performance."""

    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load validation dataset
        self.dataset = SQuADDataset(tokenizer, split="validation[:1000]")

    def evaluate_precision_setting(
        self,
        model: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model performance for a specific precision setting."""
        model.eval()

        total_loss = 0.0
        correct_answers = 0

        with torch.no_grad():
            for sample in tqdm(self.dataset):
                # Prepare input
                text, prefix_length = sample
                tokenized_input = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)
                input_ids = tokenized_input["input_ids"]

                # Forward pass
                outputs = model(**tokenized_input)
                logits = outputs.logits

                # Calculate loss and answer correctness
                answer_logits = logits[:, prefix_length-1:-1, :].contiguous()
                answer_ids = input_ids[:, prefix_length:].contiguous()

                is_correct = torch.all((answer_logits.argmax(dim=-1) == answer_ids), dim=-1)
                loss = F.cross_entropy(
                    answer_logits.view(-1, answer_logits.size(-1)),
                    answer_ids.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='mean'
                )

                # Accumulate metrics
                total_loss += loss.item()
                correct_answers += is_correct.sum().item()

        # Calculate average metrics
        num_samples = len(self.dataset)
        avg_loss = total_loss / num_samples
        accuracy = correct_answers / num_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }


def main():
    parser = argparse.ArgumentParser(description="Cyclic Precision Training with SQuAD")
    parser.add_argument("--min_bits", type=int, default=3, help="Minimum bit width for CPT")
    parser.add_argument("--max_bits", type=int, default=8, help="Maximum bit width for CPT")
    parser.add_argument("--num_cycles", type=int, default=25, help="Number of cycles for CPT")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    args = parser.parse_args()

    # Set up model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create quantized model with cyclic precision training
    quantized_model = QuantizedGPT2Model(model)
    quantized_model.setup_cyclic_precision_training(
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        num_cycles=args.num_cycles,
        total_steps=args.num_iterations,
    )

    # Train the model
    trainer = CPTTrainer(
        model=quantized_model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    trainer.train(num_iterations=args.num_iterations)

    # Evaluate each precision setting
    evaluator = SQuADEvaluator(tokenizer)
    results = {}
    for bits in range(args.min_bits, args.max_bits + 1):
        quantization_config = QuantizationConfig.create_uniform_config(bits)
        quantized_model.set_quantization_config(quantization_config)
        results[bits] = evaluator.evaluate_precision_setting(quantized_model)

    # Print summary table
    print("Evaluation Results:")
    print("=" * 26)
    print(f"{'Bits':<6} {'Loss':<8} {'Accuracy':<10}")
    print("=" * 26)
    for bits, result in results.items():
        print(f"{bits:<6} {result['loss']:<8.4f} {result['accuracy']:<10.2%}")
    print("=" * 26)


if __name__ == "__main__":
    main()
