# Chess GRPO Trainer

A project for training language models on chess move prediction using Group Relative Policy Optimization (GRPO).

## Overview

This project trains language models to predict the next move in a chess game using GRPO, a reinforcement learning approach. The model is prompted with a sequence of chess moves and asked to predict the next move. The model's predictions are evaluated using multiple reward functions, including Stockfish engine evaluations.

## Key Features

- **PGN Game Loading**: Loads and processes chess games from PGN files in the `/games` directory
- **Smart Sequence Generation**: Creates training sequences by splitting games at positions that are 50-80% through the game
- **Reward-based Learning**: Uses multiple reward functions to train the model:
  - Move legality validation
  - Stockfish evaluation (rewards better moves higher)
  - Correct formatting of responses
  - Match with game moves (with quality evaluation)
- **PEFT-based Training**: Uses Parameter-Efficient Fine-Tuning with LoRA for efficient model training

## How It Works

### Data Preparation

1. The system loads PGN files from the `/games` directory
2. For each game, it selects a random point 50-80% through the game
3. It creates a sequence of moves up to that point
4. The next move in the actual game becomes the target move to predict
5. For white move prediction, the sequence ends with a move number (e.g., "15.")
6. For black move prediction, the sequence ends with white's move

### GRPO Training

Group Relative Policy Optimization (GRPO) improves upon PPO by comparing groups of model responses rather than individual responses. The training procedure:

1. Generates multiple completions for each prompt
2. Evaluates completions using reward functions
3. Updates the model to maximize expected reward while minimizing divergence from the reference model
4. Uses KL regularization (controlled by beta parameter) to prevent the model from diverging too far from the reference model

### Reward Functions

1. **Format Reward**: Ensures the model responds in the expected XML format
2. **Legal Move Reward**: Verifies that predicted moves are legal in the given position
3. **Stockfish Reward**: Uses the Stockfish chess engine to evaluate move quality
   - Best move: 1.0 reward
   - Second best: 0.7 reward
   - Third best: 0.4 reward
   - Other legal moves: 0.1 reward
4. **Correctness Reward**: Rewards moves that match the actual game move, with the reward scaled based on Stockfish's evaluation of that move

## Requirements

- Python 3.8+
- PyTorch
- chess
- transformers
- trl (for GRPO implementation)
- peft (for efficient fine-tuning)
- Stockfish (optional, for move evaluation)

## Usage

To train the model:

```bash
python chess_grpo.py
```

To test the PGN parsing functionality:

```bash
# Edit chess_grpo.py to uncomment the test_pgn_parsing() call
python chess_grpo.py
```

## Model Configuration

The default configuration:
- Base model: Qwen2.5-1.5B-Instruct
- LoRA with r=16, alpha=64
- GRPO with beta=0.04, epsilon=0.2
- Learning rate: 5e-6
- Weight decay: 0.1
- Warmup ratio: 0.1
- Scheduler: cosine

## Customization

You can customize the model by modifying:

- The base model in `model_name`
- GRPO parameters in `training_args`
- LoRA configuration in `peft_config`
- Reward weights and functions