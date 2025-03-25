"""
Quick test script for PEFT convergence analysis

This script runs a very small training job with minimal parameters
to verify everything works correctly on your MacBook before running
in Google Colab with GPUs.
"""

import os
import subprocess
import sys

def run_test():
    print("Running a minimal PEFT training test...")
    
    # Use a very small dataset subset for quick testing
    cmd = [
        sys.executable,
        "peft_training.py",
        "--model_name", "google/flan-t5-small",  # Small model for local testing
        "--dataset_name", "ag_news",             # Classification dataset
        "--peft_method", "lora",                 # LoRA is typically the fastest PEFT method
        "--optimizer", "adamw",                  # Standard optimizer
        "--lr_schedule", "constant",             # Simple LR schedule
        "--learning_rate", "5e-4",               # Typical learning rate for PEFT
        "--batch_size", "4",                     # Small batch size for MacBook
        "--num_epochs", "1",                     # Just one epoch for testing
        "--max_samples", "32",                   # Very few samples for quick test
    ]
    
    # Add wandb logging if wanted (commented out by default)
    # cmd.append("--log_wandb")
    
    # Run the command
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print("\nTest completed! If this ran without errors, the setup is working correctly.")
    print("For a full training run in Google Colab, you can use:")
    print("- Larger models (facebook/opt-350m, google/flan-t5-base, etc.)")
    print("- More data (increase max_samples or remove it completely)")
    print("- More epochs (3-10)")
    print("- Enable wandb logging with --log_wandb")

if __name__ == "__main__":
    run_test() 