import os
import time
import argparse
import torch
import numpy as np
import psutil
import wandb
from datetime import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from accelerate import Accelerator
def parse_args():
    parser = argparse.ArgumentParser(description="PEFT Convergence Analysis - Upgraded for LLaMA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--peft_method", type=str, default="lora", choices=["lora"])
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--full_ft", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor", "lion"])
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "linear", "cosine", "warmup", "plateau"])
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_wandb", action="store_true")
    return parser.parse_args()
def get_optimizer(name, model, lr, weight_decay):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adafactor":
        from transformers import Adafactor
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("Lion optimizer not found. Falling back to AdamW.")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
def get_peft_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
def prepare_dataset(dataset_name, tokenizer, max_samples):
    dataset = load_dataset(dataset_name)
    dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
    dataset["test"] = dataset["test"].select(range(min(max_samples // 2, len(dataset["test"]))))

    label_to_text = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}

    def tokenize_function(examples):
        prompts = [f"Classify: {text}" for text in examples["text"]]
        labels = [label_to_text[label] for label in examples["label"]]
        texts = [f"{p} -> {l}" for p, l in zip(prompts, labels)]
        model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(tokenize_function, batched=True)
def log_metrics(step, metrics, use_wandb=False):
    metrics["memory_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Step {step}: {metrics}")
    if use_wandb:
        wandb.log(metrics, step=step)
def collate_fn(batch):
    return {
        key: torch.stack([torch.as_tensor(x[key], dtype=torch.long) for x in batch])
        for key in ["input_ids", "attention_mask", "labels"]
    }
def train(args):
    set_seed(args.seed)

    if args.log_wandb:
        wandb.init(
            project="peft-convergence-llama-ag-news",
            config=vars(args),
            name=f"llama1b_{args.peft_method}_{args.optimizer}_{args.lr_schedule}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    accelerator = Accelerator()
    token = "HUGGING_FACE_TOKEN"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=token, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float32 if args.full_ft else torch.float16,
        use_auth_token=token
    )
    dataset = prepare_dataset(args.dataset_name, tokenizer, args.max_samples)
    if args.full_ft:
        print("Full fine-tuning: training ALL model parameters.")
        for param in model.parameters():
            param.requires_grad = True
    elif args.use_peft:
        print("Using PEFT (LoRA): training adapter layers only.")
        model = get_peft_model(model, get_peft_config())
    else:
        print("Warning: Neither full_ft nor use_peft set! Defaulting to full fine-tuning.")
        for param in model.parameters():
            param.requires_grad = True
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay)
    train_loader = DataLoader(dataset["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_loader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = args.num_epochs * len(train_loader)
    is_plateau = False
    if args.lr_schedule == "warmup":
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    elif args.lr_schedule in ["linear", "cosine", "constant"]:
        lr_scheduler = get_scheduler(args.lr_schedule, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)
    elif args.lr_schedule == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        is_plateau = True
    else:
        raise ValueError("Unknown lr_schedule")

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )
    model.train()
    global_step = 0
    best_eval_loss = float('inf')
    convergence_counter = 0
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            if torch.isnan(loss):
                print(f"NaN detected at step {global_step}")
                exit(1)
            accelerator.backward(loss)
            optimizer.step()
            if not is_plateau:
                lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 10 == 0:
                log_metrics(global_step, {"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, args.log_wandb)
            global_step += 1
        model.eval()
        eval_loss = 0
        for batch in eval_loader:
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
        eval_loss /= len(eval_loader)

        if is_plateau:
            lr_scheduler.step(eval_loss)

        log_metrics(global_step, {"eval_loss": eval_loss}, args.log_wandb)

        if abs(eval_loss - best_eval_loss) / best_eval_loss < 0.01:
            convergence_counter += 1
        else:
            convergence_counter = 0
        best_eval_loss = min(best_eval_loss, eval_loss)
        if convergence_counter >= 3:
            print(f"Model converged at step {global_step}")
            break
        model.train()
    if args.log_wandb:
        wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    if args.full_ft:
        args.use_peft = False 
    train(args)