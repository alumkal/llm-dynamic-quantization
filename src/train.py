"""Training utilities for switchable precision quantization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
import datasets
from typing import Dict, List, Tuple
import random
from tqdm import tqdm

from .switchable_precision import SwitchablePrecisionGPT2Model, SwitchablePrecisionConfig


class SQuADDataset(torch.utils.data.Dataset):
    """SQuAD dataset for question answering fine-tuning."""

    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer

        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SQuAD dataset
        self.dataset = datasets.load_dataset("squad", split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prefix_text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        prefix_length = len(self.tokenizer.tokenize(prefix_text))
        text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer: {sample['answers']['text'][0]}\n"
        return text, prefix_length


class SwitchablePrecisionTrainer:
    """Trainer for switchable precision quantization."""

    def __init__(
        self,
        model: SwitchablePrecisionGPT2Model,
        tokenizer: GPT2Tokenizer,
        precision_settings: List[str],
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        max_length: int = 1024
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.precision_settings = precision_settings
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

        # Setup optimizer - only optimize LoRA parameters
        trainable_params = []
        for param in model.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        self.optimizer = AdamW(trainable_params, lr=learning_rate)

        print(f"Training {len(trainable_params)} parameter groups")
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params:,}")

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

    def train_step_cascade(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with cascade distillation loss."""
        self.model.train()

        # Assume precision settings are already sorted from highest to lowest quality
        sorted_settings = self.precision_settings

        # Forward pass for all precision settings
        all_outputs = {}
        all_losses = {}

        for setting in sorted_settings:
            self.model.set_precision(setting)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            all_outputs[setting] = outputs.logits

        # Calculate cascade losses
        total_loss = 0.0
        beta = 0.1  # trade-off parameter for distillation loss

        for i, current_setting in enumerate(sorted_settings):
            # Standard cross-entropy loss
            logits = all_outputs[current_setting]
            labels = batch["labels"]

            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )

            # Distillation loss from higher precision settings
            distill_loss = 0.0
            for j in range(i):  # All higher precision settings
                teacher_setting = sorted_settings[j]
                teacher_logits = all_outputs[teacher_setting].detach()  # Stop gradient

                mse_loss = F.mse_loss(
                    torch.log_softmax(logits, dim=-1),
                    torch.log_softmax(teacher_logits, dim=-1),
                    reduction='mean'
                )
                distill_loss += mse_loss / i

            # Combined loss for this precision setting
            setting_loss = ce_loss + beta * distill_loss
            all_losses[current_setting] = setting_loss.item()
            total_loss += setting_loss

        # Average loss across all precision settings
        total_loss = total_loss / len(sorted_settings)

        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return all_losses

    def train(self, num_iterations: int = 1000):
        """Train the model with switchable precision and cascade distillation."""
        self.model.train()

        total_loss = 0
        precision_losses = {setting: [] for setting in self.precision_settings}

        pbar = tqdm(range(num_iterations), desc="Training")

        for iteration in pbar:
            # Get next batch
            try:
                batch = next(iter(self.dataloader))
            except StopIteration:
                # Reset dataloader if we run out of data
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=self._collate_fn
                )
                batch = next(iter(self.dataloader))

            # Cascade training step - trains all precisions simultaneously
            step_losses = self.train_step_cascade(batch)

            # Update tracking
            avg_step_loss = sum(step_losses.values()) / len(step_losses)
            total_loss += avg_step_loss

            for setting, loss in step_losses.items():
                precision_losses[setting].append(loss)

            # Update progress bar
            avg_loss = total_loss / (iteration + 1)
            postfix = {"avg_loss": f"{avg_loss:.4f}"}
            for setting in self.precision_settings:
                postfix[setting] = f"{step_losses.get(setting, 0):.3f}"
            pbar.set_postfix(postfix)

        # Print final statistics
        print(f"Average loss: {total_loss / num_iterations:.4f}")

        for setting in self.precision_settings:
            if precision_losses[setting]:
                avg_loss = sum(precision_losses[setting]) / len(precision_losses[setting])
                print(f"{setting}: {avg_loss:.4f}")

        return total_loss / num_iterations
