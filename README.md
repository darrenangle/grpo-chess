# Chess GRPO Trainer

Training language models to predict chess moves using Group Relative Policy Optimization (GRPO).

## Overview

This project trains language models to predict the next move in a chess game using GRPO, a reinforcement learning approach that leverages Stockfish evaluations. The model is prompted with chess game sequences and learns to predict high-quality moves through multiple reward functions.

## Key Features

- **Chess PGN Parsing**: Processes chess games from PGN files in the `/games` directory
- **GRPO Training**: Uses multiple reward functions to train the model:
  - Format reward: Ensures correct XML format
  - Legal move validation: Guarantees moves are valid
  - Stockfish evaluation: Rewards higher quality moves
  - Correctness reward: Rewards game-matching moves
- **Optimized for H200 GPU**: Takes advantage of 141GB VRAM for training
- **Hugging Face Hub Integration**: Automatically saves checkpoints to the Hub

## Requirements

- Python 3.8+
- PyTorch 2.1.0+ (with CUDA support)
- Transformers 4.35.0+, TRL 0.7.1+, PEFT 0.5.0+
- Hugging Face Hub, Weights & Biases
- Stockfish (for move evaluation)
- NVIDIA H200 GPU (141GB VRAM)

## Setup

### Quick Start with Virtual Environment

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Stockfish
# Ubuntu/Debian: sudo apt-get install stockfish
# macOS: brew install stockfish
# Windows: Download from https://stockfishchess.org/download/

# Verify setup
python check_env.py
```

## Usage

### Training the Model

```bash
# Standard training with Stockfish rewards
python chess_grpo.py

# Training with only format and legality rewards (ablation study)
python chess_grpo.py --ablation

# Push checkpoints to Hugging Face Hub
python chess_grpo.py --push-to-hub username/model-name
```

## Running on RunPod with H200 GPU

### Automated Setup (Recommended)

1. Launch a RunPod H200 instance
2. Run the setup script:
   ```bash
   curl -O https://raw.githubusercontent.com/yourusername/grpo-chess/main/setup_runpod.sh
   chmod +x setup_runpod.sh
   ./setup_runpod.sh --wandb-key your_wandb_api_key --hf-token your_huggingface_token --hub-repo yourusername/chess-grpo-h200
   ```

The script will install dependencies, configure authentication for W&B and Hugging Face, and start training in a tmux session.

### Monitoring Training

```bash
# Attach to training session
tmux attach-session -t grpo-training

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Model Configuration

### H200 Configuration

- Base model: nvidia/AceInstruct-7B
- LoRA with r=32, alpha=64, bias="lora"
- Target modules: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
- Learning rate: 8e-6 with warmup ratio 0.05
- Batch size: 16
- Training: 3 epochs
- Context: 1024 tokens prompt, 4096 tokens completion
- Generations per prompt: 4

## GRPO Implementation

The GRPO approach:
1. Generates multiple completions for each chess position
2. Evaluates the quality of each completion with reward functions
3. Optimizes the model by maximizing expected rewards while staying close to the reference model
4. Uses validation checkpoints to save the best model

Reward weighting:
- Format reward (proper XML): 0.5
- Legal move validation: 1.0
- Stockfish move quality (top 20 moves with non-linear decay):
  - Top move: 1.0
  - 2nd best: 0.95
  - 3rd best: 0.90
  - 5th best: 0.80
  - 10th best: 0.50
  - 20th best: 0.10
  - Legal moves not in top 20: 0.05
- Correctness reward: Match with game move + bonus based on Stockfish ranking

The training process balances exploration (finding better moves) with exploitation (refining known good moves) to improve the model's chess playing ability.