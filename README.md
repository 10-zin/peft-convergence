# PEFT Convergence Analysis

This repository contains code for analyzing the convergence speed of Parameter-Efficient Fine-Tuning (PEFT) methods with different optimizers and learning rate schedules.

## Project Overview

The goal of this project is to investigate whether techniques such as conscious learning-rate scheduling, better regularization techniques, and optimizer selection can overcome slow convergence problems with PEFT methods, especially in low-data scenarios.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/peft-convergence.git
cd peft-convergence
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running a Quick Test

To verify your setup works correctly, you can run a quick test on your local machine (even without a GPU):

### PEFT Mode (Default)
```bash
python run_local_test.py
```

### Full Fine-tuning Mode (Comparison)
```bash
python run_local_test.py --full_ft
```

This will run a minimal training job with a small model (FLAN-T5-small) and a very small subset of the AG News dataset.

## Running Full Experiments

For full experiments, we recommend using Google Colab with GPU acceleration. Open the `peft_convergence_colab.ipynb` notebook in Google Colab and follow the instructions there.

### Command-line Options

For manual experimentation, you can run training directly with the following options:

#### PEFT Training (Default)
```bash
python peft_training.py \
    --model_name google/flan-t5-small \
    --dataset_name ag_news \
    --peft_method lora \
    --optimizer adamw \
    --lr_schedule warmup \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --num_epochs 3 \
    --max_samples 1000 \
    --log_wandb
```

#### Full Fine-tuning (Comparison)
```bash
python peft_training.py \
    --model_name google/flan-t5-small \
    --dataset_name ag_news \
    --peft_method lora \
    --full_ft \
    --optimizer adamw \
    --lr_schedule warmup \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --num_epochs 3 \
    --max_samples 1000 \
    --log_wandb
```

### Available Models

- `google/flan-t5-small` (for local testing)
- `google/flan-t5-base`
- `meta-llama/Llama-2-7b-hf` (requires Hugging Face access)
- `mistralai/Mistral-7B-v0.1` (requires Hugging Face access)

### Available Datasets

- `ag_news`: AG News classification dataset
- `cola`: CoLA grammatical correctness dataset

### PEFT Methods

- `lora`: Low-Rank Adaptation
- `prefix_tuning`: Prefix Tuning

### Training Modes

- `Default`: Uses PEFT method specified by `--peft_method`
- `--full_ft`: Uses full fine-tuning (no PEFT)

### Optimizers

- `adamw`: AdamW optimizer
- `adafactor`: Adafactor optimizer
- `lion`: Lion optimizer (if library is installed)

### Learning Rate Schedules

- `constant`: Constant learning rate
- `linear`: Linear decay
- `cosine`: Cosine decay
- `warmup`: Linear warmup followed by linear decay

## Analyzing Results

If you've enabled Weights & Biases logging (`--log_wandb`), you can view experiment results on the W&B dashboard. Otherwise, training metrics will be printed to the console.

## Repository Structure

- `peft_training.py`: Main script for PEFT training and evaluation
- `run_local_test.py`: Script for running a quick local test
- `peft_convergence_colab.ipynb`: Notebook for running experiments in Google Colab
- `requirements.txt`: Required Python packages
- `README.md`: This file

## Contributing

Feel free to open issues or submit pull requests with improvements or bug fixes.