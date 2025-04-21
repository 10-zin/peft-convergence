"""
Quick test script for PEFT convergence analysis

This script runs very small training jobs with minimal parameters
to verify the different modes (Full FT, LoRA, BitFit) work correctly
on your MacBook before running in Google Colab with GPUs.
"""

import os
import subprocess
import sys
import argparse

def run_training_test(use_peft, peft_method, learning_rate, dataset_name="ag_news"):
    """Runs peft_training.py with specific test parameters."""
    
    test_mode = "Full Fine-tuning"
    if use_peft:
        test_mode = f"PEFT ({peft_method})"
    print(f"\n----- Testing {test_mode} -----")

    # Use a very small dataset subset for quick testing
    cmd = [
        sys.executable,
        "peft_training.py",
        "--model_name", "google/flan-t5-small",  # Small model for local testing
        "--dataset_name", dataset_name,
        "--optimizer", "adamw",                  # Standard optimizer
        "--lr_schedule", "constant",             # Simple LR schedule
        "--learning_rate", learning_rate,         # Learning rate for PEFT
        "--batch_size", "4",                     # Small batch size for MacBook
        "--num_epochs", "1",                     # Just one epoch for testing
        "--max_samples", "32",                   # Very few samples for quick test
    ]

    if use_peft:
        cmd.extend(["--use_peft", "--peft_method", peft_method])
    else:
        # Explicitly turn off peft for full fine-tuning test
        # We don't need to add --use_peft False, as it's the default absence
        # But we could add --full_ft if we preferred that flag logic
        pass # No extra flags needed for full FT if --use_peft default is True and not specified

    # Add wandb logging if wanted (commented out by default)
    # cmd.append("--log_wandb")

    # Run the command
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"----- {test_mode} Test Completed Successfully -----")
    except subprocess.CalledProcessError as e:
        print(f"----- {test_mode} Test FAILED -----")
        print(e)
    except FileNotFoundError:
        print("Error: 'python' command not found. Make sure Python is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    # Test Full Fine-tuning
    # run_training_test(use_peft=False, peft_method='none', learning_rate='5e-4') # peft_method is ignored when use_peft=False

    # Test PEFT with LoRA
    # run_training_test(use_peft=True, peft_method='lora', learning_rate='5e-4')

    # Test PEFT with BitFit on AG News
    # # Use a higher learning rate for BitFit as per recommendations
    # run_training_test(use_peft=True, peft_method='bitfit', learning_rate='3e-3', dataset_name='ag_news')
    
    # Test PEFT with LoRA on SAMSum
    run_training_test(use_peft=True, peft_method='lora', learning_rate='5e-4', dataset_name='samsum')

    print("\nAll tests completed! If they ran without errors, the setup is working correctly.")
    print("For a full training run in Google Colab, you can use:")
    print("- Larger models (facebook/opt-350m, google/flan-t5-base, etc.)")
    print("- More data (increase max_samples or remove it completely)")
    print("- More epochs (3-10)")
    print("- Enable wandb logging with --log_wandb") 