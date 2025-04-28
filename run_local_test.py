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

def run_training_test(use_peft, peft_method, learning_rate, dataset_name="ag_news", weight_decay="0.01"):
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
        "--weight_decay", weight_decay,           # Weight decay
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
    output_log = []
    try:
        # Capture output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line, end='') # Print output in real-time
            output_log.append(line.strip())
        process.wait()
        if process.returncode != 0:
             raise subprocess.CalledProcessError(process.returncode, cmd)
        print(f"----- {test_mode} Test Completed Successfully -----")
        return True, "\n".join(output_log) # Return success and log
    except subprocess.CalledProcessError as e:
        print(f"----- {test_mode} Test FAILED ----- Error Code: {e.returncode}")
        # print(e)
        return False, "\n".join(output_log) # Return failure and log
    except FileNotFoundError:
        print("Error: 'python' command not found. Make sure Python is in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"----- {test_mode} Test FAILED with unexpected error ----- ")
        print(e)
        return False, "\n".join(output_log) # Return failure and log

if __name__ == "__main__":
    # Test Full Fine-tuning
    # run_training_test(use_peft=False, peft_method='none', learning_rate='5e-4', weight_decay='0.01')

    # # Test PEFT with LoRA
    # run_training_test(use_peft=True, peft_method='lora', learning_rate='5e-4', weight_decay='0.01')

    # # Test PEFT with BitFit on AG News
    # # Use a higher learning rate for BitFit as per recommendations
    # run_training_test(use_peft=True, peft_method='bitfit', learning_rate='3e-3', dataset_name='ag_news', weight_decay='0.0') # Bitfit often uses 0 wd
    
    # Test PEFT with LoRA on SAMSum
    samsum_success, samsum_log = run_training_test(use_peft=True, peft_method='lora', learning_rate='5e-4', dataset_name='samsum', weight_decay='0.01')

    # --- Verification Step for SAMSum ROUGE --- 
    if samsum_success:
        print("\n----- Verifying SAMSum ROUGE Metrics ----- ")
        rouge_found = False
        for line in samsum_log.splitlines():
            if "'rouge1':" in line and "'rouge2':" in line and "'rougeL':" in line:
                 print(f"Found ROUGE metrics log line: {line}")
                 # Basic check: ensure scores are not exactly zero (they should be floats)
                 if "'rouge1': 0.0," not in line.replace(" ", "") and \
                    "'rougeL': 0.0," not in line.replace(" ", ""):
                     rouge_found = True
                     print("ROUGE scores seem present and non-zero. Evaluation likely worked.")
                     break
                 else:
                     print("Warning: ROUGE scores found but appear to be zero. Check calculation.")
                     break # Found the line, no need to continue
        if not rouge_found:
             print("Error: Could not find ROUGE metrics in the SAMSum test output log. Evaluation might have failed silently.")
    else:
        print("\nSAMSum test failed, skipping ROUGE verification.")
    # --------------------------------------------

    print("\nAll tests completed! If they ran without errors, the setup is working correctly.")
    print("For a full training run in Google Colab, you can use:")
    print("- Larger models (facebook/opt-350m, google/flan-t5-base, etc.)")
    print("- More data (increase max_samples or remove it completely)")
    print("- More epochs (3-10)")
    print("- Enable wandb logging with --log_wandb") 