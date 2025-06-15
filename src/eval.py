"""Evaluation utilities for switchable precision models."""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Dict, List
from tqdm import tqdm

from .switchable_precision import SwitchablePrecisionGPT2Model


class SQuADEvaluator:
    """Evaluator for SQuAD dataset performance."""

    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load validation dataset
        self.dataset = load_dataset("squad", split="validation[:1000]")

    def evaluate_precision_setting(
        self,
        model: SwitchablePrecisionGPT2Model,
    ) -> Dict[str, float]:
        """Evaluate model performance for a specific precision setting."""
        model.eval()

        total_loss = 0.0
        correct_answers = 0

        with torch.no_grad():
            for sample in tqdm(self.dataset):
                # Prepare input
                text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer: {sample['answers']['text'][0]}\n"
                tokenized_input = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(model.device)
                input_ids = tokenized_input["input_ids"]

                # Calculate prefix length
                prefix_text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
                prefix_length = len(self.tokenizer.tokenize(prefix_text))

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
        compression_ratio = model.calculate_compression_ratio()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "compression": compression_ratio,
        }
