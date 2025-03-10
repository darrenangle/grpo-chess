import torch
import chess
import chess.pgn
import io
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import transformers
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from stockfish import Stockfish
import random
import os
import torch.multiprocessing as mp
from accelerate import Accelerator
import sys

# Check if stockfish is available
STOCKFISH_PATH = "/usr/games/stockfish"  # Path to Stockfish executable

try:
    stockfish = Stockfish(path=STOCKFISH_PATH)
    STOCKFISH_AVAILABLE = True
    print(f"Stockfish engine found at: {STOCKFISH_PATH}")
except Exception as e:
    print(f"Warning: Stockfish not available - {e}")
    print("Make sure Stockfish is installed and the path is correct.")
    STOCKFISH_AVAILABLE = False

# System prompt and formatting
SYSTEM_PROMPT = """
You are a chess assistant. Given a sequence of chess moves, predict the next best move.

The sequence will end with either:
1. A move number followed by a period (e.g., "10.") - this means you should predict White's next move
2. A move without a following move number - this means you should predict Black's next move

Respond in the following format:

<reasoning>
Analyze the position carefully. Consider various candidate moves and their consequences.
</reasoning>
<answer>
e4
</answer>
"""

XML_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extract the answer from the XML format."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def is_valid_move(move_str: str, fen: str) -> bool:
    """Check if a move is valid in the given position."""
    if not move_str or len(move_str.strip()) == 0:
        return False
        
    try:
        board = chess.Board(fen)
        # Try to parse the move string
        move = chess.Move.from_uci(move_str.lower())
        return move in board.legal_moves
    except:
        try:
            # Try to parse as SAN
            board = chess.Board(fen)
            move = board.parse_san(move_str)
            return move in board.legal_moves
        except Exception as e:
            # Specific error for debugging
            if "illegal san" in str(e):
                # Just return False quietly for illegal SAN
                pass
            else:
                # Print other errors for debugging
                print(f"Move validation error: {e} for move '{move_str}' in position {fen}")
            return False

def evaluate_move(move_str: str, fen: str, stockfish_instance: Stockfish) -> float:
    """Evaluate a move using Stockfish."""
    if not STOCKFISH_AVAILABLE:
        return 0.0
    
    try:
        # Create a new Stockfish instance or reset the current one
        # This helps prevent any lingering state issues
        if stockfish_instance is None:
            stockfish_instance = Stockfish(path=STOCKFISH_PATH)
        
        # Set up the position
        stockfish_instance.set_fen_position(fen)
        
        # Get top moves from Stockfish
        top_moves = stockfish_instance.get_top_moves(3)
        
        # Check if our move is among the top moves
        for i, move_info in enumerate(top_moves):
            top_move = move_info['Move']
            if move_str.lower() == top_move.lower():
                # Return reward based on rank (best = 1.0, second = 0.7, third = 0.4)
                return 1.0 - (i * 0.3)
        
        # Move not in top 3
        return 0.1  # Small reward for any legal move
    except Exception as e:
        print(f"Error evaluating move: {e}")
        return 0.0

def process_pgn_game(pgn_text):
    """Process a PGN game and extract the positions and next moves."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if not game:
            return []
        
        samples = []
        board = game.board()
        
        # Process each move in the game
        for move in game.mainline_moves():
            fen = board.fen()
            move_san = board.san(move)
            
            # Make the move on the board
            board.push(move)
            
            # Get the full PGN up to this point
            position_pgn = pgn_text.split(move_san)[0] + move_san
            
            samples.append({
                "position": position_pgn,
                "fen": fen,
                "next_move": move_san
            })
            
        return samples
    except Exception as e:
        print(f"Error processing game: {e}")
        return []

def load_game_from_pgn_file(pgn_file_path):
    """Load a chess game from a PGN file with robust encoding handling."""
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(pgn_file_path, 'r', encoding=encoding) as f:
                game = chess.pgn.read_game(f)
                while game is not None:
                    yield game
                    game = chess.pgn.read_game(f)
                # If we get here without error, we're done
                break
        except UnicodeDecodeError:
            # Try the next encoding
            if encoding == encodings[-1]:
                # If this was the last encoding to try, skip this file
                print(f"Warning: Could not decode {pgn_file_path} with any encoding. Skipping.")
                return
            continue
        except Exception as e:
            print(f"Error loading {pgn_file_path}: {e}")
            return

def get_game_sequence(game, split_percentage=None):
    """
    Process a game and create a sequence for prediction.
    
    Splits the game at a position that is 50-80% through the game,
    returning the sequence up to that point and the next move to predict.
    """
    try:
        if split_percentage is None:
            # Random split between 50% and 80% of the game
            split_percentage = random.uniform(0.5, 0.8)
        
        # Convert game to a list of moves
        moves = list(game.mainline_moves())
        
        if len(moves) < 5:  # Skip very short games
            return None, None, None
        
        # Determine the split point
        split_index = int(len(moves) * split_percentage)
        
        # Create a board and make moves up to the split point
        board = game.board()
        sequence = []
        
        # Track move numbers and moves played
        move_number = 1
        is_white_move = True
        
        for i, move in enumerate(moves[:split_index]):
            try:
                if is_white_move:
                    sequence.append(f"{move_number}.")
                
                move_san = board.san(move)
                sequence.append(move_san)
                board.push(move)
                
                # Update move counters for next iteration
                if not is_white_move:
                    move_number += 1
                is_white_move = not is_white_move
            except Exception as e:
                # Skip problematic moves and continue
                print(f"Error processing move {i} in game: {e}")
                continue
        
        # The move to predict is the next move after the split
        if split_index < len(moves):
            try:
                next_move = board.san(moves[split_index])
                
                # For prediction, we want:
                # - If predicting white's move: sequence ends with move number (e.g., "15.")
                # - If predicting black's move: sequence ends with white's move
                
                if not is_white_move:
                    # We're predicting black's move, so sequence already ends with white's move
                    pass
                else:
                    # We're predicting white's move, so add the move number
                    sequence.append(f"{move_number}.")
                
                if sequence:
                    return ' '.join(sequence), next_move, board.fen()
            except Exception as e:
                print(f"Error extracting next move: {e}")
                return None, None, None
        
        return None, None, None
    except Exception as e:
        print(f"Error processing game sequence: {e}")
        return None, None, None

# Function to process a single PGN file - must be outside the main function to be picklable
def process_pgn_file(pgn_file):
    """Process a single PGN file and extract game samples."""
    local_white_samples = []
    local_black_samples = []
    
    print(f"Processing {pgn_file}...")
    for game in load_game_from_pgn_file(pgn_file):
        sequence, next_move, fen = get_game_sequence(game)
        if sequence and next_move and fen:
            # Determine if this is predicting white's or black's move
            is_predicting_white = sequence.strip().endswith('.')
            
            sample = {
                "position": sequence,
                "fen": fen,
                "next_move": next_move
            }
            
            # Add to appropriate list
            if is_predicting_white:
                local_white_samples.append(sample)
            else:
                local_black_samples.append(sample)
    
    return local_white_samples, local_black_samples

def prepare_chess_dataset(use_multiprocessing=True):
    """Prepare the chess dataset from local PGN files with balanced white/black move predictions.
    Optimized for multi-GPU training with parallel processing."""
    from concurrent.futures import ProcessPoolExecutor
    import math
    
    # Get all PGN files in the games directory
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    pgn_files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.pgn')]
    
    # Use multiple processes to parse PGN files in parallel
    white_move_samples = []
    black_move_samples = []
    
    if use_multiprocessing:
        try:
            # Determine number of workers (use number of cores, but max 8)
            num_workers = min(os.cpu_count(), 8)
            
            print(f"Processing PGN files with {num_workers} workers in parallel...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_pgn_file, pgn_files))
                
                # Combine results
                for white_samples, black_samples in results:
                    if white_samples and black_samples:  # Check for None results
                        white_move_samples.extend(white_samples)
                        black_move_samples.extend(black_samples)
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            print("Falling back to sequential processing...")
            use_multiprocessing = False
    
    # Sequential processing fallback
    if not use_multiprocessing:
        print("Processing PGN files sequentially...")
        for pgn_file in pgn_files:
            white_samples, black_samples = process_pgn_file(pgn_file)
            if white_samples and black_samples:  # Check for None results
                white_move_samples.extend(white_samples)
                black_move_samples.extend(black_samples)
    
    print(f"Generated {len(white_move_samples)} white move predictions")
    print(f"Generated {len(black_move_samples)} black move predictions")
    
    # Balance the dataset to ensure 50/50 split
    min_samples = min(len(white_move_samples), len(black_move_samples))
    
    # Randomly select to get equal numbers
    white_move_samples = random.sample(white_move_samples, min_samples)
    black_move_samples = random.sample(black_move_samples, min_samples)
    
    # Combine and shuffle
    all_samples = white_move_samples + black_move_samples
    random.shuffle(all_samples)
    
    # Format the dataset for the GRPO trainer (simple version without nested functions)
    formatted_data = []
    for sample in all_samples:
        formatted_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Based on this chess game sequence, predict the next move:\n\n{sample['position']}"}
            ],
            "fen": sample["fen"],
            "next_move": sample["next_move"]
        })
    
    print(f"Created balanced dataset with {len(formatted_data)} samples (50% white, 50% black moves)")
    
    # Use optimized dataset creation
    return Dataset.from_list(formatted_data)

# Reward functions
def legal_move_reward_func(prompts, completions, fen, **kwargs) -> list[float]:
    """Reward function that checks if the move is legal."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    return [1.0 if is_valid_move(move, f) else 0.0 
            for move, f in zip(extracted_moves, fen)]

def stockfish_reward_func(prompts, completions, fen, **kwargs) -> list[float]:
    """Reward function that uses Stockfish to evaluate the move quality."""
    if not STOCKFISH_AVAILABLE:
        return [0.0] * len(completions)
    
    try:
        stockfish_instance = Stockfish(path=STOCKFISH_PATH)
    except Exception as e:
        print(f"Error creating Stockfish instance: {e}")
        return [0.0] * len(completions)
    
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    return [evaluate_move(move, f, stockfish_instance) 
            for move, f in zip(extracted_moves, fen)]

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) is not None for r in responses]
    
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func(prompts, completions, next_move, fen, **kwargs) -> list[float]:
    """
    Reward function that evaluates the move played in the actual game.
    If the predicted move matches the game move, the reward is based on how good that move is
    according to Stockfish.
    """
    if not STOCKFISH_AVAILABLE:
        return [0.0] * len(completions)
    
    try:
        stockfish_instance = Stockfish(path=STOCKFISH_PATH)
    except Exception as e:
        print(f"Error creating Stockfish instance: {e}")
        return [0.0] * len(completions)
    
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    for move, nm, f in zip(extracted_moves, next_move, fen):
        if move.strip() == nm.strip():
            # If the move matches the game move, evaluate the game move with Stockfish
            try:
                # Create a fresh instance to prevent state issues
                fresh_stockfish = Stockfish(path=STOCKFISH_PATH)
                fresh_stockfish.set_fen_position(f)
                top_moves = fresh_stockfish.get_top_moves(3)
                
                # Check where the game move ranks in Stockfish's evaluation
                for i, move_info in enumerate(top_moves):
                    top_move = move_info['Move']
                    if nm.lower() == top_move.lower():
                        # Game move is in top 3, reward based on rank
                        rewards.append(1.0 - (i * 0.3))
                        break
                else:
                    # Game move not in top 3, give smaller reward
                    rewards.append(0.3)
            except Exception as e:
                print(f"Error evaluating game move: {e}")
                rewards.append(0.3)  # Default reward if evaluation fails
        else:
            rewards.append(0.0)  # No reward if predicted move doesn't match game move
    
    return rewards

def configure_wandb_logging(ablation=False):
    """Configure Weights & Biases logging with useful tags for the chess GRPO experiment."""
    try:
        import wandb
        # Don't initialize if already running
        if wandb.run is not None:
            return True
            
        # Add useful tags for experiment tracking
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        rewards = "format,legal" if ablation else "format,legal,stockfish,correctness"
        
        wandb.init(
            project="chess-grpo",
            tags=["chess", "rl", "grpo", "dual-4090"],
            config={
                "model": model_name,
                "reward_functions": rewards,
                "hardware": "dual-rtx-4090",
                "ablation": ablation,
            }
        )
        return True
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        print("Training will continue without wandb logging.")
        return False

def main(ablation=False, disable_multiprocessing=False):
    """
    Main training function with optional ablation mode.
    
    Args:
        ablation: If True, only train with format and legality rewards,
                 ignoring move quality rewards from Stockfish.
        disable_multiprocessing: If True, disable multiprocessing for dataset preparation.
    """
    # Initialize accelerator to handle distributed training
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process
    
    if is_main_process:
        print(f"Running with accelerator config: {accelerator.state}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Device: {accelerator.device}")
    
    # Set up multiprocessing method for PyTorch
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Method might already be set in distributed mode
        pass
    
    # Configure wandb logging with useful tags
    if "wandb" in sys.modules:
        configure_wandb_logging(ablation=ablation)
    # Model and training configuration
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Prepare output directory with ablation indicator if needed
    if ablation:
        output_dir = "outputs/Chess-GRPO-Ablation"
        run_name = "Chess-GRPO-Ablation-Training"
    else:
        output_dir = "outputs/Chess-GRPO"
        run_name = "Chess-GRPO-Training"
    
    # Training configuration optimized for dual 4090s
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,        # Increased batch size
        gradient_accumulation_steps=2,        # Reduced GAStep since batch size is higher
        num_generations=4,                    # Must be divisible by per_device_train_batch_size
        max_prompt_length=512,                # Chess games can be long
        max_completion_length=256,
        num_train_epochs=1,
        save_strategy="steps",               # Save at regular steps
        save_steps=100,                      # Save every 100 steps (must match eval_steps)
        save_total_limit=10,                 # Keep last 10 checkpoints
        save_safetensors=True,               # Use faster safetensors format
        load_best_model_at_end=True,         # Load best model at end based on reward
        metric_for_best_model="eval_reward", # Use reward as metric for best model
        greater_is_better=True,              # Higher reward is better
        eval_strategy="steps",               # Evaluate at regular steps (instead of evaluation_strategy)
        eval_steps=100,                      # Evaluate every 100 steps
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        ddp_find_unused_parameters=False,     # Optimize DDP
        dataloader_num_workers=0,             # Disable parallel data loading to fix pickling errors
        # Let accelerate handle distributed training properly
    )
    
    # PEFT configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    
    # Let GRPOTrainer handle model loading and distribution
    # Load the model with simple configuration, don't use device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    print(f"Loading model: {model_name}")
    
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    dataset = prepare_chess_dataset(use_multiprocessing=not disable_multiprocessing)
    
    # Create validation split for better checkpointing
    if len(dataset) > 100:  # Only split if we have enough data
        # Shuffle the dataset indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Take ~10% for validation, with a cap of 1000 samples
        val_size = min(len(dataset) // 10, 1000)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create the splits
        train_dataset = dataset.select(train_indices)
        eval_dataset = dataset.select(val_indices)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(eval_dataset)}")
    else:
        # Not enough data for validation
        train_dataset = dataset
        eval_dataset = None
        print(f"Dataset size: {len(dataset)} (no validation split)")
    
    # Set up checkpoint directories for different epochs
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose reward functions based on ablation setting
    if ablation:
        reward_functions = [
            format_reward_func,
            legal_move_reward_func
        ]
        print("ABLATION MODE: Using only format and legality rewards (ignoring move quality)")
    else:
        reward_functions = [
            format_reward_func,
            legal_move_reward_func,
            stockfish_reward_func, 
            correctness_reward_func
        ]
        print("FULL MODE: Using all rewards including move quality from Stockfish")
    
    # Initialize the trainer with multi-GPU optimizations and checkpointing
    # Don't pass FSDP config directly - let the training args handle it
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[
            # Add early stopping as well
            transformers.EarlyStoppingCallback(early_stopping_patience=5)
        ]
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Save checkpoint with timestamp (to be able to reference this run later)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    final_checkpoint_dir = os.path.join(output_dir, f"final-checkpoint-{timestamp}")
    trainer.save_model(final_checkpoint_dir)
    print(f"Final model saved to: {final_checkpoint_dir}")
    
    # Save a small metadata file with information about the run
    metadata = {
        "timestamp": timestamp,
        "model_name": model_name,
        "ablation": ablation,
        "batch_size": training_args.per_device_train_batch_size,
        "num_generations": training_args.num_generations,
        "dataset_size": len(train_dataset),
    }
    
    # Save metadata to a JSON file
    import json
    with open(os.path.join(final_checkpoint_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete! Models and checkpoints saved to {output_dir}")
    print(f"Final model with timestamp saved to: {final_checkpoint_dir}")

def test_pgn_parsing():
    """Test function to demonstrate PGN parsing and sequence generation."""
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    pgn_files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.pgn')]
    
    # Pick a random PGN file
    pgn_file = random.choice(pgn_files)
    print(f"Testing with file: {pgn_file}")
    
    # Load a game
    games = list(load_game_from_pgn_file(pgn_file))
    if not games:
        print("No games found in the file.")
        return
    
    # Pick a random game
    game = random.choice(games)
    
    # Test with various split percentages
    for split_pct in [0.5, 0.6, 0.7, 0.8]:
        print(f"\nSplit at {split_pct*100:.0f}% of the game:")
        sequence, next_move, fen = get_game_sequence(game, split_pct)
        
        if sequence and next_move:
            print(f"Sequence: {sequence}")
            print(f"Next move to predict: {next_move}")
            
            # Check if we're predicting white or black's move
            if sequence.strip().endswith('.'):
                print("Predicting WHITE's move")
            else:
                print("Predicting BLACK's move")
        else:
            print("Could not generate a valid sequence")

def launch_distributed_training():
    """Launch training using torchrun for distributed training."""
    import subprocess
    import sys
    
    print("Launching distributed training across multiple GPUs...")
    
    # Get number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPUs")
    
    # Default to 2 GPUs for the 2x 4090 setup
    num_gpus = min(2, gpu_count)
    
    # Build command line arguments to pass through
    cmd_args = []
    if "--ablation" in sys.argv:
        cmd_args.append("--ablation")
    if "--seq" in sys.argv or "--sequential" in sys.argv:
        cmd_args.append("--seq")
    
    # Launch with torchrun
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        sys.argv[0],
        "--distributed"
    ] + cmd_args
    
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess GRPO Training")
    parser.add_argument("--ablation", action="store_true", help="Run in ablation mode")
    parser.add_argument("--test-parsing", action="store_true", help="Test PGN parsing")
    parser.add_argument("--seq", "--sequential", action="store_true", 
                        help="Use sequential processing for dataset preparation")
    args = parser.parse_args()
    
    if args.test_parsing:
        test_pgn_parsing()
    else:
        # Use Accelerate to handle distribution - just call main directly
        main(ablation=args.ablation, disable_multiprocessing=args.seq)