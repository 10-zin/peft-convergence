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
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
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
from tqdm.auto import tqdm
import evaluate # Import evaluate

# Helper function to print trainable parameters
def print_trainable_parameters(model):
    """
    Prints and returns the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # Avoid division by zero if the model has no parameters
    trainable_percent = 100 * trainable_params / all_param if all_param > 0 else 0
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {trainable_percent:.4f}%"
    )
    return trainable_params, all_param # Return counts

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
        choices=["lora", "bitfit"],  # Only implemented methods
        help="PEFT method to use (lora or bitfit)",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Whether to use PEFT (True) or full fine-tuning (False)",
    )
    parser.add_argument(
        "--full_ft",
        action="store_true",
        help="Use full fine-tuning instead of PEFT (overrides --use_peft)",
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
        "--weight_decay",
        type=float,
        default=0.01, # Common default
        help="Weight decay for optimizer (AdamW)",
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
        "--max_test_samples",
        type=int,
        default=None,
        help="Max number of test samples to use (defaults to max_samples // 2 if max_samples is set)",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="peft-convergence", # Default project name
        help="Weights & Biases project name",
    )
    args = parser.parse_args()
    
    # Override use_peft if full_ft is specified
    if args.full_ft:
        args.use_peft = False
    
    return args


# Helper function to get optimizer
def get_optimizer(optimizer_name, model, lr, weight_decay=0.01):
    if optimizer_name == "adamw":
        # Only AdamW typically uses weight_decay directly like this
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adafactor":
        from transformers import Adafactor
        # Adafactor has its own decay mechanism, ignore weight_decay arg
        print("Note: Adafactor optimizer does not use standard weight_decay argument.")
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
    elif optimizer_name == "lion":
        try:
            from lion_pytorch import Lion
            # Lion also has its own weight decay handling
            return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("Lion optimizer not available. Using AdamW instead.")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")


# Function to prepare datasets
def prepare_dataset(dataset_name, tokenizer, max_samples=None, model_type="seq_cls", max_test_samples=None):
    if dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        
        # Subsample train set if max_samples is specified
        if max_samples:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
        
        # Subsample test set
        num_test = max_test_samples if max_test_samples is not None else (max_samples // 2 if max_samples is not None else None)
        if num_test is not None:
             dataset["test"] = dataset["test"].select(range(min(num_test, len(dataset["test"]))))
        
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
            # Remove original text column after tokenization
            tokenized_dataset = tokenized_dataset.remove_columns(["text"])
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
    elif dataset_name == "samsum":
        # SAMSum requires trusting remote code for its loading script
        dataset = load_dataset("samsum", trust_remote_code=True)
        
        # Subsample train set if max_samples is specified
        if max_samples:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            
        # Subsample test/validation set
        test_split_name = "test"
        if test_split_name not in dataset:
             test_split_name = "validation" # Fallback to validation if no test split
        
        num_test = max_test_samples if max_test_samples is not None else (max_samples // 2 if max_samples is not None else None)
        if num_test is not None:
            if test_split_name in dataset:
                 dataset[test_split_name] = dataset[test_split_name].select(range(min(num_test, len(dataset[test_split_name]))))
            else:
                 print(f"Warning: Neither 'test' nor 'validation' split found for {dataset_name}. Cannot subsample test set.")
        elif test_split_name not in dataset: # Handle case where no subsampling needed but test split is missing
            raise ValueError(f"Required split '{test_split_name}' not found in dataset {dataset_name}.")
            
        # Rename the chosen test split to "test" for consistency downstream
        if test_split_name != "test" and test_split_name in dataset:
            dataset["test"] = dataset.pop(test_split_name)
            
        # Ensure model is appropriate for summarization task (Seq2Seq or CausalLM)
        if model_type not in ["seq2seq", "causal_lm"]:
            raise ValueError(f"SAMSum dataset requires a sequence-to-sequence or causal language model, but got model_type='{model_type}'")

        # --- Tokenization --- 
        if model_type == "seq2seq":
            prefix = "summarize: "
            def tokenize_samsum_seq2seq(examples):
                dialogues = [prefix + dialogue for dialogue in examples["dialogue"]]
                summaries = examples["summary"]

                model_inputs = tokenizer(
                    dialogues, padding="max_length", truncation=True, max_length=512 # Longer context for dialogues
                )
                
                # Tokenize labels (summaries)
                with tokenizer.as_target_tokenizer():
                    labels_encodings = tokenizer(
                        summaries, padding="max_length", truncation=True, max_length=128 # Max summary length
                    )
                
                model_inputs["labels"] = labels_encodings["input_ids"]
                return model_inputs
            
            tokenized_dataset = dataset.map(tokenize_samsum_seq2seq, batched=True)
            
        elif model_type == "causal_lm":
            # Format for Causal LM: Input includes both dialogue and summary
            # Example format: "Dialogue: <dialogue> Summary: <summary> <eos>"
            # The model learns to predict the summary part.
            def tokenize_samsum_causal(examples):
                outputs = []
                for dialogue, summary in zip(examples["dialogue"], examples["summary"]):
                    # Simple concatenation - adjust prompt format as needed
                    text = f"Dialogue:\n{dialogue}\n\nSummary:\n{summary}{tokenizer.eos_token}"
                    # Tokenize the combined text
                    # Use a max_length that accommodates dialogue + summary reasonably
                    tokenized = tokenizer(text, truncation=True, max_length=768) 
                    outputs.append(tokenized)
                
                # Collate the outputs (list of dicts) into a single dict of lists
                # This structure is expected by .map
                collated_output = {}
                for key in outputs[0].keys():
                    collated_output[key] = [d[key] for d in outputs]
                return collated_output

            tokenized_dataset = dataset.map(tokenize_samsum_causal, batched=True)
            # For Causal LM, labels are typically handled by the model/loss function (shifted inputs)
            # We might need a specific DataCollator later. For now, just remove summary col if present.
            if "summary" in tokenized_dataset["train"].column_names:
                tokenized_dataset = tokenized_dataset.remove_columns(["summary", "dialogue", "id"])

        return tokenized_dataset, 0 # num_labels not applicable for summarization
    else:
        raise ValueError(f"Dataset {dataset_name} preparation not implemented yet")


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
        # Create a detailed run name
        run_name_parts = [
            args.model_name.split('/')[-1], # Get model name part
            args.dataset_name,
            f"peft-{args.peft_method}" if args.use_peft else "full-ft",
            f"opt-{args.optimizer}",
            f"lr-{args.learning_rate:.0e}", # Scientific notation for LR
            f"bs-{args.batch_size}",
            f"epochs-{args.num_epochs}",
            f"wd-{args.weight_decay}"
        ]
        run_name = "-".join(run_name_parts)
        
        wandb.init(
            project=args.wandb_project, # Use the argument here
            config=vars(args), 
            name=run_name, # Use the detailed name
        )
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # === Add Pad Token if missing ===
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # We will set padding_side later, after model_type is determined
        tokenizer_pad_token_was_none = True 
    else:
        tokenizer_pad_token_was_none = False
    # ==============================

    # Determine model type and setup
    model_kwargs = {}
    if "t5" in args.model_name.lower():
        model_type = "seq2seq"
        if args.dataset_name in ["ag_news", "cola"]:
            # For classification with T5, we treat it as a seq2seq task
            # where output is a class name or index
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        # Non-T5 model: Determine if it's CLM or SeqClass based on dataset
        if args.dataset_name in ["ag_news", "cola"]:
            model_type = "seq_cls"
            num_labels = 4 if args.dataset_name == "ag_news" else 2
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                **model_kwargs 
            )
        elif args.dataset_name in ["samsum", "e2e_nlg"]:
            print(f"Loading {args.model_name} as AutoModelForCausalLM for generation task.")
            model_type = "causal_lm" 
            # Load base model for Causal LM. Add trust_remote_code if needed for specific models.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, 
                trust_remote_code=True, # Often needed for newer models
                **model_kwargs
            )
            # Ensure model has pad_token_id configured if tokenizer needed it
            if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        else:
            # If dataset is unknown for non-T5, raise error
            raise ValueError(f"Unsupported dataset '{args.dataset_name}' for non-T5 model '{args.model_name}'. Please implement handling.")
    
    # === Set Padding Side for Causal LM if Pad Token was added ===
    if model_type == "causal_lm" and tokenizer_pad_token_was_none:
        print("Setting padding_side='left' for Causal LM tokenizer.")
        tokenizer.padding_side = "left"
    # ==========================================================

    # === Update Model Config PAD TOKEN ID (moved here to ensure model is loaded) ===
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Explicitly set model.config.pad_token_id to: {tokenizer.pad_token_id}")
    # ========================================

    # Prepare dataset
    print(f"Preparing dataset: {args.dataset_name}")
    dataset, num_labels = prepare_dataset(
        args.dataset_name, tokenizer, args.max_samples, model_type, args.max_test_samples
    )
    
    # Apply PEFT method if enabled
    if args.use_peft:
        if args.peft_method == "lora":
            print(f"Applying LoRA PEFT method...")
            # Determine PEFT TaskType based on final model_type
            if model_type == "seq2seq":
                peft_task_type = TaskType.SEQ_2_SEQ_LM
            elif model_type == "seq_cls":
                peft_task_type = TaskType.SEQ_CLS
            elif model_type == "causal_lm":
                peft_task_type = TaskType.CAUSAL_LM
            else:
                raise ValueError(f"Unsupported model_type '{model_type}' for LoRA task type.")
                
            peft_config = LoraConfig(
                task_type=peft_task_type,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)
        elif args.peft_method == "bitfit":
            print(f"Applying BitFit PEFT method...")
            # Freeze all parameters first
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze bias parameters
            for name, param in model.named_parameters():
                if "bias" in name:
                    param.requires_grad = True
        else:
            # This should not happen due to argparse choices, but good practice
            raise ValueError(f"Unsupported PEFT method: {args.peft_method}")
    else:
        print("Using Full Fine-tuning (no PEFT method applied).")

    # Print number of trainable parameters *after* setup
    trainable_params, all_params = print_trainable_parameters(model) # Get counts

    # Add check for PEFT methods resulting in no trainable parameters
    if args.use_peft and trainable_params == 0:
        raise ValueError(
            f"PEFT method '{args.peft_method}' resulted in 0 trainable parameters "
            f"for the model '{args.model_name}'. This PEFT method might be incompatible "
            f"with the model architecture (e.g., BitFit requires bias parameters). "
            f"Try a different PEFT method like LoRA or use --full_ft."
        )

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
        # Pre-padding was done during tokenization for seq_cls
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Use dedicated collator
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use DataCollatorWithPadding
        )
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use DataCollatorWithPadding
        )
    elif model_type == "seq2seq":
        # Handles padding and label formatting for seq2seq models like T5
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use Seq2Seq collator
        )
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use Seq2Seq collator
        )
    elif model_type == "causal_lm":
        # Handles padding and labels creation for causal language modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # mlm=False for CLM
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use LM collator
        )
        eval_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            collate_fn=data_collator, # Use LM collator
        )
    else:
         # Should not happen based on earlier checks
         raise ValueError(f"Invalid model_type '{model_type}' for DataLoader creation.")

    # Setup optimizer
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay)
    
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
    
    # Load ROUGE metric if needed
    rouge_metric = None
    if args.dataset_name == "samsum":
        rouge_metric = evaluate.load("rouge")
    
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
        
        # Wrap train_dataloader with tqdm for progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}", leave=False)
        for batch in progress_bar:
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
        all_preds = []
        all_labels = []
        all_preds_text = [] # For ROUGE
        all_labels_text = [] # For ROUGE
        total_eval_loss = 0 # Initialize eval loss accumulator
        num_eval_batches = 0 # Count batches for averaging
        
        # Wrap eval_dataloader with tqdm for progress bar
        eval_progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False)
        for batch in eval_progress_bar:
            with torch.no_grad():
                # Store original labels before popping
                # original_labels_ids = batch["labels"].clone() # Moved logic inside model_type branches
                
                if model_type == "seq_cls":
                    outputs = model(**batch)
                    # Fix: Accumulate loss correctly for seq_cls
                    total_eval_loss += outputs.loss.item() 
                    num_eval_batches += 1
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                elif model_type == "seq2seq":
                    # === Calculate Eval Loss ===
                    # Need to pass labels to model for loss calculation
                    batch_for_loss = {k: v for k, v in batch.items()} # Create a copy for loss calc
                    batch_for_loss["labels"] = batch["labels"].clone()
                    loss_outputs = model(**batch_for_loss)
                    total_eval_loss += loss_outputs.loss.item()
                    num_eval_batches += 1
                    # ==========================
                    
                    # === Generate Output ===
                    # Use accelerator.unwrap_model if model is wrapped
                    unwrapped_model = accelerator.unwrap_model(model)
                    generated_ids = unwrapped_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=128, # Use max_new_tokens instead
                        num_beams=4,  # Add beam search
                    )
                    
                    # Decode generated and reference texts
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    # Decode T5 labels properly
                    original_labels_ids = batch["labels"].clone()
                    decoded_labels = tokenizer.batch_decode(original_labels_ids, skip_special_tokens=True)
                    
                    if args.dataset_name == "samsum":
                        all_preds_text.extend(generated_texts)
                        all_labels_text.extend(decoded_labels)
                        # Add batch for ROUGE computation
                        rouge_metric.add_batch(predictions=generated_texts, references=decoded_labels)
                    elif args.dataset_name in ["ag_news", "cola"]: # T5 classification
                        # Convert text to class index for classification metrics
                        preds = [get_class_from_generated_text(text, label_map) for text in generated_texts]
                        labels = [get_class_from_generated_text(text, label_map) for text in decoded_labels]
                        all_preds.extend(preds)
                        all_labels.extend(labels)
                    # No else needed, other datasets aren't implemented
                elif model_type == "causal_lm": # New branch for Causal LM evaluation
                    # === Calculate Eval Loss ===
                    # Pass batch directly, loss calculation handles -100 labels internally
                    loss_outputs = model(**batch)
                    total_eval_loss += loss_outputs.loss.item()
                    num_eval_batches += 1
                    # ==========================

                    # === Generate Output === 
                    # Generate from input_ids only
                    unwrapped_model = accelerator.unwrap_model(model)
                    generated_ids = unwrapped_model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=128, 
                        num_beams=4,  
                        pad_token_id=tokenizer.pad_token_id # Important for CLM generation with padding
                    )
                    
                    # Decode generated text only
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Cannot decode batch["labels"] because of -100. Need original summaries for ROUGE.
                    # TODO: Implement passing original summaries through dataloader for Causal LM ROUGE 
                    if args.dataset_name == "samsum":
                        print("Warning: ROUGE calculation not fully implemented for Causal LM evaluation yet.")
                        all_preds_text.extend(generated_texts)
                        # all_labels_text.extend([]) # Skip reference labels for now
                        # rouge_metric.add_batch(...) # Skip ROUGE update
                
        
        # Calculate metrics
        eval_metrics = {}
        avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0
        eval_metrics["eval_loss"] = avg_eval_loss # Add eval loss
        
        if args.dataset_name in ["ag_news", "cola"]:
            eval_metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            eval_metrics["f1"] = f1_score(all_labels, all_preds, average='weighted')
        elif args.dataset_name == "samsum":
            # Compute ROUGE only if it was used (for seq2seq model_type on samsum)
            if model_type == "seq2seq" and rouge_metric is not None:
                try:
                    rouge_results = rouge_metric.compute()
                    eval_metrics.update(rouge_results) # Adds rouge1, rouge2, rougeL, rougeLsum
                except ValueError as e:
                    # Handle cases where compute is called without add/add_batch (e.g., empty eval set)
                    print(f"Warning: ROUGE compute failed: {e}. Setting ROUGE scores to 0.")
                    eval_metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0})
            
            # Set accuracy/f1 to 0 for consistency in logs, regardless of model_type for samsum
            eval_metrics["accuracy"] = 0.0 
            eval_metrics["f1"] = 0.0
        else: # Fallback for unimplemented datasets
            eval_metrics["accuracy"] = 0.0
            eval_metrics["f1"] = 0.0 
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log evaluation metrics
        metrics = {"epoch": epoch}
        metrics.update(eval_metrics) # Add calculated metrics (now includes eval_loss for samsum)
        metrics["epoch_time_seconds"] = epoch_time
        
        log_metrics(global_step, metrics, args.log_wandb)
        
        # Check for convergence (based on accuracy for classification)
        if args.dataset_name in ["ag_news", "cola"]:
             current_metric_val = eval_metrics["accuracy"]
             if epoch > 0:
                 metric_change = abs(current_metric_val - best_accuracy)
                 if metric_change < convergence_threshold:
                     convergence_counter += 1
                     if convergence_counter >= convergence_patience and convergence_step is None:
                         convergence_step = global_step
                         print(f"Model converged at step {convergence_step} based on accuracy")
                 else:
                     convergence_counter = 0
             # Update best accuracy
             if current_metric_val > best_accuracy:
                 best_accuracy = current_metric_val
                 print(f"New best accuracy: {best_accuracy:.4f}")
        # Add convergence check based on ROUGE or loss for samsum later if needed
        # else:
        #     pass # No convergence check for samsum yet
    
    # Training complete
    # training_time = time.time() - training_start_time
    
    # Log final metrics separately without a step number
    # print(f"Step final: {final_metrics}")
    # if args.log_wandb:
    #     # Add final memory usage
    #     process = psutil.Process(os.getpid())
    #     final_metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
    #     wandb.log(final_metrics)

    print("Training complete!")
    if args.log_wandb:
        wandb.finish()
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args) 