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
    
    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load validation dataset
        dataset = load_dataset("squad", split="validation[:10]")  # Very small subset for quick evaluation
        self.data = self._prepare_data(dataset)
    
    def _prepare_data(self, dataset) -> List[Dict]:
        """Prepare SQuAD data for evaluation."""
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
    
    def evaluate_precision_setting(
        self, 
        model: SwitchablePrecisionGPT2Model, 
        precision_setting: str
    ) -> Dict[str, float]:
        """Evaluate model performance for a specific precision setting."""
        model.eval()
        model.set_precision(precision_setting)
        
        total_loss = 0.0
        total_perplexity = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for item in tqdm(self.data, desc=f"Evaluating {precision_setting}"):
                input_ids = item["input_ids"].unsqueeze(0)
                attention_mask = item["attention_mask"].unsqueeze(0)
                labels = item["labels"].unsqueeze(0)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='mean'
                )
                
                # Calculate perplexity
                perplexity = torch.exp(loss)
                
                total_loss += loss.item()
                total_perplexity += perplexity.item()
                num_samples += 1
        
        avg_loss = total_loss / num_samples
        avg_perplexity = total_perplexity / num_samples
        compression_ratio = model.calculate_compression_ratio()
        
        return {
            "loss": avg_loss,
            "perplexity": avg_perplexity,
            "compression": compression_ratio,
            "num_samples": num_samples
        }
    
    def evaluate_all_settings(
        self, 
        model: SwitchablePrecisionGPT2Model, 
        precision_settings: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance for all precision settings."""
        results = {}
        
        for setting in precision_settings:
            results[setting] = self.evaluate_precision_setting(model, setting)
        
        return results