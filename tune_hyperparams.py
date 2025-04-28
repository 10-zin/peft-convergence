"""
Hyperparameter Tuning Script for PEFT Convergence Analysis

This script runs multiple training jobs with varying hyperparameters 
(e.g., learning rates) and logs results to Weights & Biases.
"""

import os
import subprocess
import sys
import argparse
import itertools

def parse_tuning_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for PEFT Convergence")
    
    # --- Configurable Hyperparameters for Tuning ---
    parser.add_argument(
        "--learning_rates", 
        nargs='+', 
        type=float, 
        default=[1e-1, 1e-2, 1e-3, 1e-4], 
        help="List of learning rates to test."
    )
    parser.add_argument(
        "--lr_schedules", 
        nargs='+', 
        type=str, 
        default=['constant', 'linear', 'warmup'], 
        choices=["constant", "linear", "cosine", "warmup"],
        help="List of learning rate schedules to test."
    )
    parser.add_argument(
        "--batch_sizes", 
        nargs='+', 
        type=int, 
        default=[16, 32, 64],
        help="List of batch sizes to test."
    )
    parser.add_argument(
        "--dataset_names",
        nargs='+',
        type=str,
        default=['ag_news', 'samsum'],
        choices=["ag_news", "samsum"], # Add more choices as needed
        help="List of dataset names to test."
    )
    parser.add_argument(
        "--peft_methods",
        nargs='+',
        type=str,
        default=['lora', 'bitfit', 'none'],
        choices=["lora", "bitfit", "none", "full"], # Added none/full for FT
        help="List of PEFT methods (or 'none'/'full' for Full Fine-Tuning) to test."
    )
    parser.add_argument(
        "--optimizers",
        nargs='+',
        type=str,
        default=['adamw', 'adafactor'], # Default optimizers to test
        choices=["adamw", "adafactor", "lion"],
        help="List of optimizers to test."
    )
    # Add more hyperparameters to tune here (e.g., --optimizers)
    
    # --- Fixed Parameters for this Tuning Run ---
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small", help="Base model name.")
    parser.add_argument("--dataset_name", type=str, default="ag_news", choices=["ag_news", "samsum"], help="Dataset to use.")
    parser.add_argument("--peft_method", type=str, default="lora", choices=["lora", "bitfit", "full"], help="PEFT method (or 'none' for full FT).")
    parser.add_argument("--use_peft", action="store_true", default=True, help="Set to true to use PEFT method specified, false for full FT.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor", "lion"], help="Optimizer.")
    parser.add_argument("--lr_schedule", type=str, default="constant", help="Learning rate schedule.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None for full dataset).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Max number of test samples to use across all runs (defaults to max_samples // 2 if max_samples is set in peft_training.py)."
    )
    
    # --- Wandb Configuration ---
    parser.add_argument("--wandb_project", type=str, default="peft-hyperparam-tuning", help="Wandb project name for tuning runs.")

    args = parser.parse_args()

    # Handle full fine-tuning logic based on INITIAL argument
    # This might need adjustment if peft_method itself is tuned
    # if args.peft_method.lower() == 'none' or args.peft_method.lower() == 'full':
    #     args.use_peft = False
    #     args.peft_method = 'none' # Use a consistent internal value

    return args

def run_training_job(config, base_cmd):
    """Constructs and runs a single training job command."""
    
    cmd = base_cmd.copy()
    
    # Add hyperparameters from the current config
    cmd.extend(["--learning_rate", str(config['learning_rate'])])
    cmd.extend(["--lr_schedule", config['lr_schedule']])
    cmd.extend(["--batch_size", str(config['batch_size'])])
    cmd.extend(["--dataset_name", config['dataset_name']]) # Pass dataset from config
    cmd.extend(["--optimizer", config['optimizer']]) # Pass optimizer from config
    # Add other tuned hyperparameters here (like peft_method itself)
    
    # Add fixed parameters (excluding those being tuned)
    fixed_keys_to_add = [
        "model_name", "weight_decay", # Removed optimizer
        "num_epochs", "seed", "wandb_project"
    ]
    for key in fixed_keys_to_add:
         cmd.extend([f"--{key}", str(config[key])])
    
    # Always log for tuning
    cmd.append("--log_wandb")

    # --- Handle PEFT Method from Config --- 
    current_peft_method = config['peft_method']
    if current_peft_method.lower() not in ['none', 'full']:
        cmd.extend(["--use_peft", "--peft_method", current_peft_method])
        # Special handling for BitFit high LR / low WD if needed
        if current_peft_method == 'bitfit':
             print("Note: Consider setting specific LR/WD for BitFit if defaults aren't optimal.")
             # Example: Overwrite WD for BitFit if desired
             # cmd[cmd.index('--weight_decay') + 1] = '0.0' 
    else:
        # Explicitly add --full_ft flag for clarity if peft_training.py uses it,
        cmd.append("--full_ft")

    if config['max_samples'] is not None:
        cmd.extend(["--max_samples", str(config['max_samples'])])
    
    if config['max_test_samples'] is not None:
        cmd.extend(["--max_test_samples", str(config['max_test_samples'])])

    print(f"\n--- Running Tuning Config --- KERNEL: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"--- Config Completed Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Config FAILED --- ")
        print(e)
    except FileNotFoundError:
        print("Error: 'python' command not found. Make sure Python is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    tuning_args = parse_tuning_args()

    # --- Define Hyperparameter Grid ---
    param_grid = {
        'learning_rate': tuning_args.learning_rates,
        'lr_schedule': tuning_args.lr_schedules,
        'batch_size': tuning_args.batch_sizes,
        'dataset_name': tuning_args.dataset_names, 
        'peft_method': tuning_args.peft_methods, 
        'optimizer': tuning_args.optimizers, # Add optimizer here
        # Add other lists of parameters to test, e.g.
        # 'optimizer': ['adamw', 'lion'], 
    }

    # Create all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting hyperparameter tuning for {len(hyperparameter_combinations)} configurations...")

    # Base command for peft_training.py
    base_cmd = [sys.executable, "peft_training.py"]

    # Convert tuning_args Namespace to dict for easier merging
    fixed_params = vars(tuning_args)

    # Run training for each combination
    for i, hyperparams in enumerate(hyperparameter_combinations):
        print(f"\n--- Running Job {i+1}/{len(hyperparameter_combinations)} --- Hyperparameters: {hyperparams} ---")
        
        # Combine fixed params and current hyperparams
        current_config = fixed_params.copy()
        current_config.update(hyperparams) 
        
        run_training_job(current_config, base_cmd)

    print("\n--- Hyperparameter Tuning Completed ---") 