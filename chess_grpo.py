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

You MUST respond in the following format and ONLY this format:

<reasoning>
Analyze the sequence of moves (PGN) carefully. Consider various candidate moves and their consequences.
</reasoning>
<answer>
e4
</answer>

Make sure to use the exact format with the <reasoning> and <answer> tags. Your suggested move in the <answer> tag must be a valid chess move in Standard Algebraic Notation (SAN) (e.g., e4, Nf3, O-O, exd5, Qxf7+, Rfd1). All other formats (like UCI) will be rejected.
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
        print(f"WARNING: No <answer> tag found in response: '{text}'")
        return ""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception as e:
        print(f"Error extracting answer: {e} from: {text}")
        return ""

def is_valid_move(move_str: str, fen: str) -> bool:
    """Check if a move is valid in the given position using Standard Algebraic Notation (SAN)."""
    if not move_str or len(move_str.strip()) == 0:
        return False
        
    try:
        # Only validate using SAN format
        board = chess.Board(fen)
        move = board.parse_san(move_str)
        return move in board.legal_moves
    except Exception as e:
        # Print errors for debugging
        if "illegal san" not in str(e):
            print(f"Move validation error: {e} for move '{move_str}' in position {fen}")
        return False

def evaluate_move(move_str: str, fen: str, stockfish_instance: Stockfish) -> float:
    """
    Evaluate a move using Stockfish with 20 top moves.
    
    Returns a reward based on the move's ranking in Stockfish's evaluation:
    - Top move: 1.0
    - Uses a non-linear decay for rewards (exponential)
    - Even the 20th move gets a small reward (0.1)
    - Legal moves outside top 20 get minimal reward (0.05)
    """
    if not STOCKFISH_AVAILABLE:
        return 0.0
    
    try:
        # Create a new Stockfish instance or reset the current one
        # This helps prevent any lingering state issues
        if stockfish_instance is None:
            stockfish_instance = Stockfish(path=STOCKFISH_PATH)
        
        # Set up the position
        stockfish_instance.set_fen_position(fen)
        
        # Get top 20 moves from Stockfish
        top_moves = stockfish_instance.get_top_moves(20)
        
        # Check if we got moves back
        if not top_moves:
            print(f"Warning: No moves returned by Stockfish for position {fen}")
            return 0.0
        
        # Check if our move is among the top moves
        for i, move_info in enumerate(top_moves):
            top_move = move_info['Move']
            if move_str.lower() == top_move.lower():
                # Non-linear reward decay:
                # 1st: 1.0, 2nd: 0.95, 3rd: 0.9, 5th: 0.8, 10th: 0.5, 20th: 0.1
                return max(0.1, 1.0 * (0.95 ** i))
        
        # Move not in top 20, check if it's at least legal
        if is_valid_move(move_str, fen):
            return 0.05  # Small reward for legal moves not in top 20
        
        # Illegal move
        return 0.0
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
    
    # Format the dataset for the GRPO trainer with explicit color and tag instructions
    formatted_data = []
    for sample in all_samples:
        # Determine if we're predicting White's or Black's move
        is_white_move = sample['position'].strip().endswith('.')
        color = "White" if is_white_move else "Black"
        
        formatted_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Based on this chess game sequence, pick the next best move for {color}. Don't forget to put your answer in <answer></answer> tags.\n\n{sample['position']}"}
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
    
    # Debug first example
    if len(responses) > 0:
        print(f"\n--- Legal Move Check ---")
        print(f"Response: '{responses[0][:100]}...'")
        print(f"Extracted move: '{extracted_moves[0]}'")
        print(f"Is valid: {is_valid_move(extracted_moves[0], fen[0])}")
    
    return [1.0 if is_valid_move(move, f) else 0.0 
            for move, f in zip(extracted_moves, fen)]

def stockfish_reward_func(prompts, completions, fen, **kwargs) -> list[float]:
    """
    Reward function that uses Stockfish to evaluate the move quality.
    
    Gets the top 20 moves from Stockfish and provides rewards with a bias toward better moves:
    - Top move: 1.0
    - Reward decays non-linearly for lower-ranked moves
    - All moves in top 20 get some reward
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
    
    # Print debug info for the first example
    if len(extracted_moves) > 0:
        print(f"\n--- Stockfish Reward ---")
        print(f"Extracted move: '{extracted_moves[0]}'")
    
    rewards = []
    for move, f in zip(extracted_moves, fen):
        if not move or not f:  # Skip invalid moves/positions
            rewards.append(0.0)
            continue
            
        try:
            stockfish_instance.set_fen_position(f)
            # Get top 20 moves from Stockfish
            top_moves = stockfish_instance.get_top_moves(20)
            
            # Check if we actually got moves back
            if not top_moves:
                print(f"Warning: No moves returned by Stockfish for position {f}")
                rewards.append(0.0)
                continue
                
            # If this is the first example, print top moves for debugging
            if len(rewards) == 0:
                print(f"Top 5 Stockfish moves:")
                for i, m in enumerate(top_moves[:5]):
                    print(f"  {i+1}. {m['Move']} (Centipawn: {m.get('Centipawn', 'N/A')})")
            
            # Check if the move is in the top 20
            for i, move_info in enumerate(top_moves):
                if move.lower() == move_info['Move'].lower():
                    # Non-linear reward function:
                    # - Top move gets 1.0
                    # - Reward decays faster for lower ranks
                    # - Even the 20th move gets a small reward
                    
                    # For non-linear decay, use a sigmoid-like or exponential decay
                    # This formula gives approximately:
                    # 1st: 1.0, 2nd: 0.95, 3rd: 0.9, 5th: 0.8, 10th: 0.5, 20th: 0.1
                    reward = max(0.1, 1.0 * (0.95 ** i))
                    
                    rewards.append(reward)
                    
                    # Debug message for first example
                    if len(rewards) == 1:
                        print(f"Move '{move}' found at rank {i+1}, reward: {reward:.3f}")
                    break
            else:
                # Not in top 20, smaller reward for legal moves that aren't terrible
                # Check if the move is at least legal
                if is_valid_move(move, f):
                    minimal_reward = 0.05  # Small reward for legal moves
                    rewards.append(minimal_reward)
                    if len(rewards) == 1:
                        print(f"Move '{move}' not in top 20 but legal, reward: {minimal_reward}")
                else:
                    rewards.append(0.0)
                    if len(rewards) == 1:
                        print(f"Move '{move}' not in top 20 and not legal, reward: 0.0")
                
        except Exception as e:
            print(f"Error in stockfish evaluation: {e}")
            rewards.append(0.0)
    
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    # Check for both strict and loose formats
    strict_pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>"
    loose_pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    
    responses = [completion[0]["content"] for completion in completions]
    strict_matches = [re.search(strict_pattern, r, re.DOTALL) is not None for r in responses]
    loose_matches = [re.search(loose_pattern, r, re.DOTALL) is not None for r in responses]
    
    # Print full first response to debug format issues
    if responses:
        print(f"\n--- Format Reward ---")
        print(f"Full response:\n{responses[0]}")
        print(f"Has strict format (<reasoning>\\n...\\n</reasoning>): {strict_matches[0]}")
        print(f"Has loose format (<reasoning>...</reasoning>): {loose_matches[0]}")
    
    # Give higher reward for strict format, but still reward loose format
    return [1.0 if strict_match else (0.5 if loose_match else 0.0) 
            for strict_match, loose_match in zip(strict_matches, loose_matches)]

def correctness_reward_func(prompts, completions, next_move, fen, **kwargs) -> list[float]:
    """
    Reward function that evaluates the move played in the actual game.
    If the predicted move matches the game move, the reward is based on how good that move is
    according to Stockfish's top 20 moves.
    
    Uses the same non-linear reward decay as the stockfish_reward_func.
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
    
    # Debug output for first example
    if responses and extracted_moves and next_move:
        print(f"\n--- Correctness Reward ---")
        print(f"Extracted move: '{extracted_moves[0]}'")
        print(f"Expected move: '{next_move[0]}'")
        print(f"Match: {extracted_moves[0].strip() == next_move[0].strip()}")
    
    rewards = []
    for move, nm, f in zip(extracted_moves, next_move, fen):
        if not move or not nm or not f:  # Skip invalid data
            rewards.append(0.0)
            continue
            
        if move.strip() == nm.strip():
            # If the move matches the game move, evaluate the game move with Stockfish
            try:
                # Create a fresh instance to prevent state issues
                fresh_stockfish = Stockfish(path=STOCKFISH_PATH)
                fresh_stockfish.set_fen_position(f)
                top_moves = fresh_stockfish.get_top_moves(20)
                
                # Check if we got moves back
                if not top_moves:
                    print(f"Warning: No moves returned by Stockfish for position {f}")
                    rewards.append(0.5)  # Default reward for matching game move when Stockfish fails
                    continue
                
                # If this is the first example, print where the game move ranks
                if len(rewards) == 0:
                    print(f"Checking game move '{nm}' against Stockfish rankings")
                
                # Check where the game move ranks in Stockfish's evaluation
                for i, move_info in enumerate(top_moves):
                    top_move = move_info['Move']
                    if nm.lower() == top_move.lower():
                        # Same non-linear decay as in stockfish_reward_func
                        # 1st: 1.0, 2nd: 0.95, 3rd: 0.9, 5th: 0.8, 10th: 0.5, 20th: 0.1
                        reward = max(0.1, 1.0 * (0.95 ** i))
                        
                        # Bonus for matching the exact move played in the game
                        reward += 0.2  # Bonus for matching historical moves
                        reward = min(1.0, reward)  # Cap at 1.0
                        
                        rewards.append(reward)
                        if len(rewards) == 1:  # Only for first example
                            print(f"Game move at Stockfish rank {i+1}, reward: {reward:.3f}")
                        break
                else:
                    # Game move not in top 20, but it's a match so give small reward
                    reward = 0.2  # Reward for matching the game move even if not top rated
                    rewards.append(reward)
                    if len(rewards) == 1:  # Only for first example
                        print(f"Game move not in Stockfish top 20, but matched. Reward: {reward}")
            except Exception as e:
                print(f"Error evaluating game move: {e}")
                rewards.append(0.1)  # Small reward if matching but evaluation fails
        else:
            rewards.append(0.0)  # No reward if predicted move doesn't match game move
            if len(rewards) == 1:  # Only for first example
                print("No match with game move, reward: 0.0")
    
    return rewards

def count_xml_tags(text):
    """Count the XML tags in the response to provide partial rewards for formatting."""
    count = 0.0
    if "<reasoning>" in text:
        count += 0.125
    if "</reasoning>" in text:
        count += 0.125
    if "<answer>" in text:
        count += 0.125
    if "</answer>" in text:
        count += 0.125
    return count

def xml_tag_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that gives partial credit for having correct XML tags."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = [count_xml_tags(r) for r in responses]
    
    # Debug first example
    if responses:
        print(f"\n--- XML Tag Count Reward ---")
        print(f"Response has these tags:")
        print(f"  <reasoning>: {'<reasoning>' in responses[0]}")
        print(f"  </reasoning>: {'</reasoning>' in responses[0]}")
        print(f"  <answer>: {'<answer>' in responses[0]}")
        print(f"  </answer>: {'</answer>' in responses[0]}")
        print(f"Tag reward: {rewards[0]}")
    
    return rewards

def configure_wandb_logging(ablation=False):
    """Configure Weights & Biases logging with useful tags for the chess GRPO experiment."""
    try:
        import wandb
        # Don't initialize if already running
        if wandb.run is not None:
            return True
            
        # Add useful tags for experiment tracking
        model_name = "nvidia/AceInstruct-7B"
        rewards = "format,legal" if ablation else "format,legal,stockfish,correctness"
        
        # Generate a unique run ID for both WandB and HF Hub
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        wandb.init(
            project="chess-grpo",
            tags=["chess", "rl", "grpo", "h200", "3-epochs"],
            config={
                "model": model_name,
                "reward_functions": rewards,
                "hardware": "runpod-h200-141gb",
                "batch_size": 16,
                "epochs": 3,
                "ablation": ablation,
                "run_id": run_id
            }
        )
        return True, run_id
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        print("Training will continue without wandb logging.")
        return False, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def push_to_hub(model, tokenizer, trainer, checkpoint_dir, repo_name, message=None):
    """
    Push model, tokenizer and training artifacts to the Hugging Face Hub.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used for training
        trainer: The GRPO trainer instance
        checkpoint_dir: The directory containing the checkpoint to push
        repo_name: The name of the repository to push to (e.g., 'username/model-name')
        message: Optional commit message
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import HfApi, create_repo
        import subprocess
        import sys
        
        # Make sure we have a valid repo name
        if "/" not in repo_name:
            print(f"Error: repo_name must include username (e.g., 'username/{repo_name}')")
            return False
        
        print(f"Pushing model to Hugging Face Hub: {repo_name}")
        
        # Create the repo if it doesn't exist
        try:
            create_repo(repo_name, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create repo: {e}")
            print(f"Will attempt to push to existing repo.")
        
        # Get API instance
        api = HfApi()
        
        # Check if we have login credentials or token
        try:
            # Check if we have a stored token
            api.whoami()
            print("Already logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Not logged in to Hugging Face Hub: {e}")
            # Check if HF_TOKEN is set
            if "HF_TOKEN" in os.environ:
                print("Using HF_TOKEN from environment")
            else:
                print("Warning: Not logged in and no HF_TOKEN found in environment")
                print("Please login with `huggingface-cli login` or provide --hub-token")
                return False
        
        # Upload training artifacts
        if not message:
            message = f"Chess GRPO model trained on {len(trainer.train_dataset)} samples"
        
        # Try to push the model using the trainer's built-in method
        try:
            print("Pushing model to Hub using trainer.push_to_hub...")
            trainer.push_to_hub(
                repo_id=repo_name,
                commit_message=message
            )
        except Exception as e:
            print(f"Error pushing with trainer: {e}")
            print("Falling back to manual push...")
            
            # Fallback: Manual push using subprocess
            try:
                # Install transformers CLI if needed
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "transformers"
                ])
                
                # Push the model using the transformers CLI
                subprocess.check_call([
                    "transformers-cli", "upload", checkpoint_dir, repo_name,
                    "--organization" if "/" in repo_name else None,
                    repo_name.split("/")[0] if "/" in repo_name else None
                ])
            except Exception as e2:
                print(f"Error with manual push: {e2}")
                print("Fallback also failed. Model only available locally.")
                return False
        
        print(f"Successfully pushed model to Hugging Face Hub: {repo_name}")
        print(f"Model available at: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        print("Training artifacts will only be available locally.")
        return False

def main(ablation=False, disable_multiprocessing=False, push_to_hub_name=None):
    """
    Main training function with optional ablation mode.
    
    Args:
        ablation: If True, only train with format and legality rewards,
                 ignoring move quality rewards from Stockfish.
        disable_multiprocessing: If True, disable multiprocessing for dataset preparation.
        push_to_hub_name: If provided, push the model to the Hugging Face Hub with this name
                        (should be in format 'username/model-name').
    """
    # Initialize accelerator for high-end single GPU (H200)
    accelerator = Accelerator()
    
    # Print GPU info
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        print(f"Training on: {gpu_properties.name}")
        print(f"GPU Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Optimize performance for large GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"Device: {accelerator.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using high-performance settings for H200 (141GB VRAM)")
    
    # Configure wandb logging with useful tags
    run_id = None
    if "wandb" in sys.modules:
        wandb_success, run_id = configure_wandb_logging(ablation=ablation)
    else:
        # Generate a unique run ID even if wandb is not used
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Model and training configuration
    model_name = "nvidia/AceInstruct-7B"
    
    # Prepare output directory with ablation indicator and unique run ID
    if ablation:
        output_dir = f"outputs/Chess-GRPO-Ablation-{run_id}"
        run_name = f"Chess-GRPO-Ablation-{run_id}"
    else:
        output_dir = f"outputs/Chess-GRPO-{run_id}"
        run_name = f"Chess-GRPO-{run_id}"
    
    # Training configuration with fixed batch sizes
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=8e-6,                  # Slightly higher learning rate
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,                  # Less regularization for multi-epoch
        warmup_ratio=0.05,                  # Shorter warmup with multiple epochs
        lr_scheduler_type='cosine',         # Cosine decay over the 3 epochs
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=16,      # Large batch size for H200 with 141GB VRAM
        gradient_accumulation_steps=1,       # No need for gradient accumulation with large GPU
        num_generations=4,                   # More generations per prompt for better GRPO
        max_prompt_length=1024,              # Allow long context for chess games
        max_completion_length=4096,          # Long completion for detailed reasoning
        num_train_epochs=3,                  # Train for 3 full epochs
        save_strategy="steps",               # Save at regular steps
        save_steps=100,                      # Save every 100 steps (must match eval_steps)
        save_total_limit=50,                 # Keep last 50 checkpoints
        save_safetensors=True,               # Use faster safetensors format
        load_best_model_at_end=True,         # Load best model at end based on reward
        metric_for_best_model="eval_reward", # Use reward as metric for best model
        greater_is_better=True,              # Higher reward is better
        eval_strategy="steps",               # Evaluate at regular steps (instead of evaluation_strategy)
        eval_steps=100,                      # Evaluate every 100 steps
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        dataloader_num_workers=12,            # High parallelism for data loading on H200
        gradient_checkpointing=False,         # Disable gradient checkpointing (plenty of VRAM)
        optim="adamw_torch",                  # Use PyTorch's efficient AdamW implementation
        # Configure HuggingFace Hub integration if requested
        push_to_hub=bool(push_to_hub_name),
        hub_model_id=push_to_hub_name,
        hub_strategy="checkpoint",            # Push model at each checkpoint save
        # H200 high-performance configuration
    )
    
    # Enhanced PEFT configuration for H200
    peft_config = LoraConfig(
        r=32,                   # Larger rank for better adaptation with H200's VRAM
        lora_alpha=64,          # Keep alpha in reasonable range
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,      # Standard dropout
        bias="lora",            # Train biases as well for better adaptation
    )
    
    # Load the model optimized for H200 with 141GB VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Use auto device mapping for H200
        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
        use_cache=True  # Enable KV-cache for faster inference
    )
    print(f"Loading model: {model_name} on {accelerator.device} (H200)")
    
    if torch.cuda.is_available():
        # Print initial memory usage 
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"Initial GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    
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
            xml_tag_reward_func,    # Add partial XML tag rewards
            format_reward_func,
            legal_move_reward_func
        ]
        print("ABLATION MODE: Using only format and legality rewards (ignoring move quality)")
    else:
        reward_functions = [
            xml_tag_reward_func,    # Add partial XML tag rewards
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
    
    # Custom callback to push to Hub after each checkpoint
    class PushToHubCallback(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if hasattr(trainer, "push_to_hub") and args.push_to_hub and args.hub_model_id:
                print(f"Pushing checkpoint at step {state.global_step} to the Hub...")
                # The trainer will automatically handle pushing to the hub
                return control
            return control
    
    # Add our callback if needed
    if push_to_hub_name:
        trainer.add_callback(PushToHubCallback())
        print(f"Added callback to push checkpoints to Hugging Face Hub: {push_to_hub_name}")
        # Install huggingface_hub if not already installed
        try:
            import huggingface_hub
            print(f"Using huggingface_hub version: {huggingface_hub.__version__}")
        except ImportError:
            print("huggingface_hub not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            import huggingface_hub
            print(f"Installed huggingface_hub version: {huggingface_hub.__version__}")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    # Save checkpoint with timestamp (to be able to reference this run later)
    final_checkpoint_dir = os.path.join(output_dir, f"final-checkpoint-{run_id}")
    trainer.save_model(final_checkpoint_dir)
    print(f"Final model saved to: {final_checkpoint_dir}")
    
    # Save a small metadata file with information about the run
    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "ablation": ablation,
        "batch_size": training_args.per_device_train_batch_size,
        "num_generations": training_args.num_generations,
        "dataset_size": len(train_dataset),
        "hub_model_id": push_to_hub_name if push_to_hub_name else None,
    }
    
    # Save metadata to a JSON file
    import json
    with open(os.path.join(final_checkpoint_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Final push to the Hugging Face Hub if requested
    if push_to_hub_name:
        if not training_args.push_to_hub:  # If we didn't already set up automatic pushing
            try:
                print(f"Pushing final model to Hugging Face Hub: {push_to_hub_name}")
                # Push the final model to the Hub
                trainer.push_to_hub(commit_message=f"Final model: Chess GRPO {run_id}")
            except Exception as e:
                print(f"Error pushing final model to Hub: {e}")
                # Try manual push as fallback
                try:
                    push_to_hub(model, tokenizer, trainer, final_checkpoint_dir, 
                               push_to_hub_name, message=f"Final model: Chess GRPO {run_id}")
                except Exception as e2:
                    print(f"Failed to push to Hub with fallback method: {e2}")
    
    print(f"Training complete! Models and checkpoints saved to {output_dir}")
    print(f"Final model saved to: {final_checkpoint_dir}")
    if push_to_hub_name:
        print(f"Model also available on Hugging Face Hub: https://huggingface.co/{push_to_hub_name}")

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

def launch_single_gpu_training():
    """Launch training on a single GPU with accelerate."""
    import subprocess
    import sys
    
    print("Launching training on a single GPU with accelerate...")
    
    # Build command line arguments to pass through
    cmd_args = []
    
    # Check for ablation mode
    if "--ablation" in sys.argv:
        cmd_args.append("--ablation")
    
    # Check for sequential processing
    if "--seq" in sys.argv or "--sequential" in sys.argv:
        cmd_args.append("--seq")
    
    # Always add direct flag when using accelerate
    cmd_args.append("--direct")
    
    # Check for HuggingFace Hub integration
    for i, arg in enumerate(sys.argv):
        if arg == "--push-to-hub" and i+1 < len(sys.argv):
            cmd_args.extend(["--push-to-hub", sys.argv[i+1]])
        elif arg.startswith("--push-to-hub="):
            cmd_args.append(arg)
        elif arg == "--hub-token" and i+1 < len(sys.argv):
            cmd_args.extend(["--hub-token", sys.argv[i+1]])
        elif arg.startswith("--hub-token="):
            cmd_args.append(arg)
    
    # Launch with accelerate
    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        sys.argv[0]
    ] + cmd_args
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chess GRPO Training")
    parser.add_argument("--ablation", action="store_true", help="Run in ablation mode")
    parser.add_argument("--test-parsing", action="store_true", help="Test PGN parsing")
    parser.add_argument("--seq", "--sequential", action="store_true", 
                        help="Use sequential processing for dataset preparation")
    parser.add_argument("--direct", action="store_true", 
                        help="Run directly without using accelerate launcher")
    parser.add_argument("--push-to-hub", type=str, metavar="USERNAME/REPO_NAME", 
                        help="Push model checkpoints to Hugging Face Hub (format: username/model-name)")
    parser.add_argument("--hub-token", type=str, metavar="TOKEN", 
                        help="Hugging Face API Token (alternatively, use HF_TOKEN env variable)")
    args = parser.parse_args()
    
    # Set HF token in environment if provided
    if args.hub_token:
        os.environ["HF_TOKEN"] = args.hub_token
        print("Using provided Hugging Face token")
    
    if args.test_parsing:
        test_pgn_parsing()
    elif args.direct:
        # Call main directly when launched manually
        main(ablation=args.ablation, 
             disable_multiprocessing=args.seq, 
             push_to_hub_name=args.push_to_hub)
    else:
        # Launch with accelerate for optimal single GPU performance
        launch_single_gpu_training()