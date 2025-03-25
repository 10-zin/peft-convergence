"""
PEFT Convergence Analysis - Proof of Concept Implementation

This script implements a baseline for analyzing PEFT method convergence
with different optimizers and learning rate schedules.
"""

import os
import time
import argparse
import torch
import numpy as np
import pandas as pd
import psutil
import wandb
from datetime import datetime
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    set_seed,
    T5ForConditionalGeneration,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PrefixTuningConfig,
)
from sklearn.metrics import accuracy_score, f1_score
from accelerate import Accelerator

# Setup command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="PEFT Convergence Analysis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        help="Model to use: google/flan-t5-small, facebook/opt-125m, etc.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ag_news",
        choices=["ag_news", "cola", "samsum", "e2e_nlg"],
        help="Dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--peft_method",
        type=str,
        default="lora",
        choices=["lora", "prefix_tuning", "ia3"],
        help="PEFT method to use",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adafactor", "lion"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "warmup"],
        help="Learning rate schedule to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max number of samples to use (for quick local testing)",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log metrics to Weights & Biases",
    )
    return parser.parse_args()


# Helper function to get optimizer
def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "adafactor":
        from transformers import Adafactor
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
    elif optimizer_name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr)
        except ImportError:
            print("Lion optimizer not available. Using AdamW instead.")
            return torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


# Function to prepare datasets
def prepare_dataset(dataset_name, tokenizer, max_samples=None, model_type="seq_cls"):
    if dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        
        # Subsample for quick local testing
        if max_samples:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            dataset["test"] = dataset["test"].select(range(min(max_samples//2, len(dataset["test"]))))
        
        # Define label maps for seq2seq models
        label_to_text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Technology"
        }
        
        if model_type == "seq_cls":
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        else:  # seq2seq
            def tokenize_function(examples):
                texts = examples["text"]
                labels = [label_to_text[label] for label in examples["label"]]
                
                # Tokenize inputs
                model_inputs = tokenizer(
                    texts, padding="max_length", truncation=True, max_length=128
                )
                
                # Tokenize labels
                with tokenizer.as_target_tokenizer():
                    labels_encodings = tokenizer(
                        labels, padding="max_length", truncation=True, max_length=8
                    )
                
                model_inputs["labels"] = labels_encodings["input_ids"]
                
                return model_inputs
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset, 4  # 4 classes for AG News
    else:
        raise ValueError(f"Dataset {dataset_name} preparation not implemented yet")


# Function to configure PEFT
def get_peft_config(peft_method, model_type="seq_cls"):
    if peft_method == "lora":
        if model_type == "seq_cls":
            return LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "v"]  # This will vary by model architecture
            )
        else:  # seq2seq
            return LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "v"]
            )
    elif peft_method == "prefix_tuning":
        if model_type == "seq_cls":
            return PrefixTuningConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                num_virtual_tokens=20,
                prefix_projection=True
            )
        else:  # seq2seq
            return PrefixTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                num_virtual_tokens=20,
                prefix_projection=True
            )
    else:
        raise ValueError(f"PEFT method {peft_method} configuration not implemented yet")


# Function to log metrics including memory usage
def log_metrics(step, metrics, log_wandb=False):
    # Add memory usage
    process = psutil.Process(os.getpid())
    metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
    
    # Print to console
    print(f"Step {step}: {metrics}")
    
    # Log to wandb
    if log_wandb:
        wandb.log(metrics, step=step)


# Helper function for T5 classification
def get_class_from_generated_text(text, label_map=None):
    # Default label mapping for ag_news
    if label_map is None:
        label_map = {
            "world": 0,
            "sports": 1,
            "business": 2,
            "technology": 3,
            "tech": 3,  # Handle abbreviations
        }
    
    text = text.lower().strip()
    for key, value in label_map.items():
        if key in text:
            return value
    return 0  # Default to first class if no match


# Main training function
def train_and_evaluate(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb if enabled
    if args.log_wandb:
        wandb.init(
            project="peft-convergence",
            config=vars(args),
            name=f"{args.peft_method}_{args.optimizer}_{args.lr_schedule}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Determine model type and setup
    if "t5" in args.model_name.lower():
        model_type = "seq2seq"
        if args.dataset_name in ["ag_news", "cola"]:
            # For classification with T5, we treat it as a seq2seq task
            # where output is a class name or index
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        model_type = "seq_cls"
        if args.dataset_name in ["ag_news", "cola"]:
            # For classification tasks
            num_labels = 4 if args.dataset_name == "ag_news" else 2
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels
            )
        else:
            # For generation tasks (not implemented in this POC)
            raise ValueError(f"Generation task with non-T5 model not implemented yet")
    
    # Prepare dataset
    print(f"Preparing dataset: {args.dataset_name}")
    dataset, num_labels = prepare_dataset(args.dataset_name, tokenizer, args.max_samples, model_type)
    
    # Create PEFT model
    print(f"Applying PEFT method: {args.peft_method}")
    peft_config = get_peft_config(args.peft_method, model_type)
    model = get_peft_model(model, peft_config)
    
    # Print number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({100 * trainable_params / total_params:.2f}%)")
    
    # Define label map for classification with T5
    label_map = {
        "world": 0,
        "sports": 1,
        "business": 2, 
        "technology": 3,
        "tech": 3,
    }
    
    # Prepare data loaders
    if model_type == "seq_cls":
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
                "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
                "labels": torch.tensor([x["labels"] for x in batch]),
            },
        )
        
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
                "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
                "labels": torch.tensor([x["labels"] for x in batch]),
            },
        )
    else:  # seq2seq
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
                "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
                "labels": torch.stack([torch.tensor(x["labels"]) for x in batch]),
                "decoder_attention_mask": torch.ones((len(batch), 8), dtype=torch.long),
            },
        )
        
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            collate_fn=lambda batch: {
                "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
                "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
                "labels": torch.stack([torch.tensor(x["labels"]) for x in batch]),
                "decoder_attention_mask": torch.ones((len(batch), 8), dtype=torch.long),
            },
        )
    
    # Setup optimizer
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
    
    # Setup learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    
    if args.lr_schedule == "constant":
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    elif args.lr_schedule == "linear":
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    elif args.lr_schedule == "cosine":
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    elif args.lr_schedule == "warmup":
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * max_train_steps),
            num_training_steps=max_train_steps,
        )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    best_accuracy = 0.0
    training_start_time = time.time()
    epoch_times = []
    
    # Track convergence
    convergence_step = None
    convergence_threshold = 0.01  # Consider converged if accuracy change < 1% for N steps
    convergence_counter = 0
    convergence_patience = 3  # Number of evaluations with minimal change to consider converged
    
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        for batch in train_dataloader:
            # Forward pass
            if model_type == "seq_cls":
                outputs = model(**batch)
                loss = outputs.loss
            else:  # seq2seq
                outputs = model(**batch)
                loss = outputs.loss
            
            # Check for NaN loss (gradient explosion)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected at step {global_step}")
                metrics = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": float("nan"),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "gradient_explosion": True
                }
                log_metrics(global_step, metrics, args.log_wandb)
                continue
            
            # Backward pass
            accelerator.backward(loss)
            
            # Log training metrics
            if global_step % 10 == 0:
                metrics = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                log_metrics(global_step, metrics, args.log_wandb)
            
            # Update model
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
        
        # Evaluate after each epoch
        model.eval()
        eval_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in eval_dataloader:
            with torch.no_grad():
                if model_type == "seq_cls":
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
                else:  # seq2seq
                    # Store original labels
                    original_labels = batch.pop("labels")
                    
                    # Generate output
                    generated_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=8,
                    )
                    
                    # Convert generated IDs to text
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Convert text to class index
                    preds = [get_class_from_generated_text(text, label_map) for text in generated_texts]
                    
                    # For T5, we need ground truth labels for comparison
                    # Convert from token IDs to class index
                    decoded_labels = tokenizer.batch_decode(original_labels, skip_special_tokens=True)
                    labels = [get_class_from_generated_text(text, label_map) for text in decoded_labels]
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log evaluation metrics
        metrics = {
            "epoch": epoch,
            "eval_loss": eval_loss if model_type == "seq_cls" else 0.0,
            "accuracy": accuracy,
            "f1_score": f1,
            "epoch_time_seconds": epoch_time,
        }
        log_metrics(global_step, metrics, args.log_wandb)
        
        # Check for convergence
        if epoch > 0:
            accuracy_change = abs(accuracy - best_accuracy)
            if accuracy_change < convergence_threshold:
                convergence_counter += 1
                if convergence_counter >= convergence_patience and convergence_step is None:
                    convergence_step = global_step
                    print(f"Model converged at step {convergence_step}")
            else:
                convergence_counter = 0
        
        # Update best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
    
    # Training complete
    training_time = time.time() - training_start_time
    
    # Log final metrics
    final_metrics = {
        "total_training_time_seconds": training_time,
        "avg_epoch_time_seconds": np.mean(epoch_times),
        "best_accuracy": best_accuracy,
        "convergence_step": convergence_step if convergence_step else "Not converged",
    }
    log_metrics("final", final_metrics, args.log_wandb)
    
    print("Training complete!")
    if args.log_wandb:
        wandb.finish()
    
    return final_metrics


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args) 