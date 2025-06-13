"""Training utilities for switchable precision quantization."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple
import random
from tqdm import tqdm

from .switchable_precision import SwitchablePrecisionGPT2Model, SwitchablePrecisionConfig


class SQuADDataset:
    """SQuAD dataset for question answering fine-tuning."""
    
    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load SQuAD dataset
        dataset = load_dataset("squad", split="train[:1000]")  # Small subset for quick training
        self.data = self._prepare_data(dataset)
    
    def _prepare_data(self, dataset) -> List[Dict]:
        """Prepare SQuAD data for language modeling."""
        prepared_data = []
        
        for example in dataset:
            # Format as: "Question: {question} Context: {context} Answer: {answer}"
            text = f"Question: {example['question']} Context: {example['context']} Answer: {example['answers']['text'][0]}"
            
            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            prepared_data.append({
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": tokens["input_ids"].squeeze().clone()
            })
        
        return prepared_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SwitchablePrecisionTrainer:
    """Trainer for switchable precision quantization."""
    
    def __init__(
        self,
        model: SwitchablePrecisionGPT2Model,
        tokenizer: GPT2Tokenizer,
        precision_settings: List[str],
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.precision_settings = precision_settings
        self.batch_size = batch_size
        
        # Setup dataset
        self.dataset = SQuADDataset(tokenizer, max_length)
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        print(f"Training {len(trainable_params)} trainable parameter groups")
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_params:,}")
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor], precision_setting: str) -> float:
        """Single training step with specified precision setting."""
        self.model.train()
        self.model.set_precision(precision_setting)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        # Calculate loss
        logits = outputs.logits
        labels = batch["labels"]
        
        # Shift labels for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self, num_iterations: int = 10):
        """Train the model with switchable precision."""
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
            
            # Randomly select precision setting for this iteration
            precision_setting = random.choice(self.precision_settings)
            
            # Training step
            loss = self.train_step(batch, precision_setting)
            
            total_loss += loss
            precision_losses[precision_setting].append(loss)
            
            # Update progress bar
            avg_loss = total_loss / (iteration + 1)
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "precision": precision_setting
            })
        
        # Print final statistics
        print(f"\nTraining completed!")
        print(f"Average loss: {total_loss / num_iterations:.4f}")
        
        for setting in self.precision_settings:
            if precision_losses[setting]:
                avg_loss = sum(precision_losses[setting]) / len(precision_losses[setting])
                print(f"Average loss for {setting}: {avg_loss:.4f} ({len(precision_losses[setting])} steps)")
        
        return total_loss / num_iterations