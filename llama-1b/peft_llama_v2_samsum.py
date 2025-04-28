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
from accelerate import Accelerator
from evaluate import load as load_metric
def parse_args():
    parser = argparse.ArgumentParser(description="Full Finetuning - LLaMA 3.2 - 1B for SAMSum")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_name", type=str, default="samsum")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor", "lion"])
    parser.add_argument("--lr_schedule", type=str, default="warmup", choices=["constant", "linear", "cosine", "warmup", "plateau"])
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
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

def prepare_dataset(tokenizer, max_samples):
    dataset = load_dataset("samsum", trust_remote_code=True)
    dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
    dataset["test"] = dataset["test"].select(range(min(max_samples // 2, len(dataset["test"]))))

    def tokenize_function(examples):
        prompts = [f"Summarize: {dialogue}" for dialogue in examples["dialogue"]]
        summaries = examples["summary"]
        full_texts = [prompt + " " + summary for prompt, summary in zip(prompts, summaries)]

        tokenized = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)

        labels = []
        for prompt, summary, input_ids in zip(prompts, summaries, tokenized["input_ids"]):
            prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
            prompt_len = len(prompt_ids)
            label = [-100] * prompt_len + input_ids[prompt_len:]
            label = label[:512]
            labels.append(label)

        tokenized["labels"] = labels
        return tokenized

    return dataset.map(tokenize_function, batched=True)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
        "labels": torch.stack([torch.tensor(x["labels"]) for x in batch]),
    }

def log_metrics(step, metrics, use_wandb=False):
    metrics["memory_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Step {step}: {metrics}")
    if use_wandb:
        wandb.log(metrics, step=step)

def train(args):
    set_seed(args.seed)

    if args.log_wandb:
        wandb.init(
            project="full-ft-llama-samsum",
            config=vars(args),
            name=f"llama1b_samsum_fullft_{args.optimizer}_{args.lr_schedule}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    accelerator = Accelerator()
    token = os.getenv("HF_TOKEN") or "YOUR TOKEN HERE"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_auth_token=token,
        use_fast=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_auth_token=token
    )
    print("Doing FULL finetuning (all parameters trainable).")
    optimizer = get_optimizer(args.optimizer, model, args.learning_rate, args.weight_decay)
    dataset = prepare_dataset(tokenizer, args.max_samples)
    train_loader = DataLoader(dataset["train"], shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)
    eval_loader = DataLoader(dataset["test"], batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = args.num_epochs * len(train_loader)
    is_plateau = False
    if args.lr_schedule == "warmup":
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    elif args.lr_schedule in ["linear", "cosine", "constant"]:
        lr_scheduler = get_scheduler(args.lr_schedule, optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    elif args.lr_schedule == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        is_plateau = True
    else:
        raise ValueError("Unknown lr_schedule")

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )
    rouge = load_metric("rouge")
    model.train()
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            if not is_plateau:
                lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % 10 == 0:
                log_metrics(global_step, {"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, args.log_wandb)
            global_step += 1
        model.eval()
        preds = []
        refs = []

        for batch in eval_loader:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=128
                )
                preds += tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                decoded_labels = []
                for label in batch["labels"]:
                    label = torch.where(label == -100, torch.tensor(tokenizer.pad_token_id).to(label.device), label)
                    decoded_labels.append(label)
                decoded_labels = torch.stack(decoded_labels)
                refs += tokenizer.batch_decode(decoded_labels, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        rougeL = rouge_output["rougeL"]
        if is_plateau:
            lr_scheduler.step(rougeL)
        log_metrics(global_step, {"rougeL": rougeL}, args.log_wandb)
        model.train()
    if args.log_wandb:
        wandb.finish()
if __name__ == "__main__":
    args = parse_args()
    train(args)
