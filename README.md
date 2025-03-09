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

## Setup

### Virtual Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Linux/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Stockfish chess engine (optional but recommended for full functionality):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install stockfish
   
   # On macOS with Homebrew
   brew install stockfish
   
   # For Windows, download from https://stockfishchess.org/download/
   ```

5. Verify your environment:
   ```bash
   python check_env.py
   ```
   This will check that all required packages are installed and report their versions.

## Usage

### Training the Model

To train the model with full rewards (including Stockfish move quality evaluation):

```bash
# Edit chess_grpo.py to set ablation = False
python chess_grpo.py
```

To train the model with ablation (only format and move legality rewards):

```bash
# Edit chess_grpo.py to set ablation = True
python chess_grpo.py
```

### Testing PGN Parsing

To test the PGN parsing functionality:

```bash
# Edit chess_grpo.py to uncomment the test_pgn_parsing() call
python chess_grpo.py
```

### Ablation Study

This project includes an ablation study to demonstrate the power of GRPO with quality-based rewards:

1. **Full Model (GRPO with Quality Rewards)**: Trained with comprehensive reward functions
   - Format reward (correct XML format)
   - Move legality reward (valid chess moves)
   - Stockfish evaluation reward (move quality)
   - Correctness reward (matching game moves with quality adjustment)

2. **Ablation Model (GRPO without Quality)**: Trained with only basic structural rewards
   - Format reward (correct XML format)
   - Move legality reward (valid chess moves)
   
The purpose of this ablation study is to demonstrate that GRPO with quality-based rewards can significantly elevate a small language model's chess-playing abilities far beyond its base ELO rating. By comparing against the ablation model (which only learns to make legal moves with proper formatting), we can isolate and quantify the specific impact of the quality-based rewards.

This experiment showcases how targeted reward functions in GRPO can efficiently teach specialized skills to smaller models, potentially allowing them to compete with much larger models on specific tasks by focusing the learning specifically on high-quality outputs.

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

## Evaluation

After training both the full and ablation models, we can evaluate their chess playing strength:

1. **Model Game Analysis**: Have each model play against Stockfish at various ELO levels to determine approximate playing strength
   
2. **Comparative Analysis**: Compare the moves chosen by:
   - Base model (untrained)
   - Ablation model (format + legality)
   - Full model (format + legality + quality)
   - Stockfish

3. **Key Metrics**:
   - Move match rate with Stockfish's top recommendations
   - Average position evaluation after model moves
   - Success rate in tactical positions
   - Win rate against different ELO-rated opponents

The hypothesis is that the full model with quality-based rewards will demonstrate significantly higher chess playing ability than both the base model and the ablation model, despite being a relatively small language model.