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
Think about the best move to make and why.
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
        # Truncate very long text in warning to avoid memory issues
        text_sample = text[:100] + "..." if len(text) > 100 else text
        print(f"WARNING: No <answer> tag found in response: '{text_sample}'")
        return ""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception as e:
        # Truncate very long text in error message
        text_sample = text[:100] + "..." if len(text) > 100 else text
        print(f"Error extracting answer: {e} from: '{text_sample}'")
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
    """Evaluate a move using Stockfish with 10 top moves."""
    if not STOCKFISH_AVAILABLE:
        return 0.0
    
    try:
        # Create a new Stockfish instance or reset the current one
        # This helps prevent any lingering state issues
        if stockfish_instance is None:
            stockfish_instance = Stockfish(path=STOCKFISH_PATH)
        
        # Set up the position
        stockfish_instance.set_fen_position(fen)
        
        # Get top 10 moves from Stockfish
        top_moves = stockfish_instance.get_top_moves(10)
        
        # Check if our move is among the top moves
        for i, move_info in enumerate(top_moves):
            top_move = move_info['Move']
            if move_str.lower() == top_move.lower():
                # Return reward based on rank (best = 1.0, gradually decreasing)
                # Top move gets 1.0, then 0.9, 0.8, etc.
                return 1.0 - (i * 0.1)
        
        # Move not in top 10
        return 0.0  # No reward if not in top 10
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
    
    # Debug first example with minimal output
    if len(responses) > 0:
        print(f"\n--- Legal Move Check ---")
        # Just print a short sample of the response to avoid large outputs
        response_sample = responses[0][:50] + "..." if len(responses[0]) > 50 else responses[0]
        print(f"Response sample: '{response_sample}'")
        print(f"Extracted move: '{extracted_moves[0]}'")
        print(f"Is valid: {is_valid_move(extracted_moves[0], fen[0])}")
    
    # Clean up CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return [1.0 if is_valid_move(move, f) else 0.0 
            for move, f in zip(extracted_moves, fen)]

def stockfish_reward_func(prompts, completions, fen, **kwargs) -> list[float]:
    """Reward function that uses Stockfish to evaluate the move quality."""
    if not STOCKFISH_AVAILABLE:
        return [0.0] * len(completions)
    
    try:
        # Use a single Stockfish instance for all evaluations
        stockfish_instance = Stockfish(path=STOCKFISH_PATH)
    except Exception as e:
        print(f"Error creating Stockfish instance: {e}")
        return [0.0] * len(completions)
    
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    rewards = []
    for move, f in zip(extracted_moves, fen):
        try:
            stockfish_instance.set_fen_position(f)
            top_moves = stockfish_instance.get_top_moves(10)
            
            # Check if the move is in the top 10
            for i, move_info in enumerate(top_moves):
                if move.lower() == move_info['Move'].lower():
                    # Reward based on rank (1.0 for best, decreasing by 0.1)
                    rewards.append(1.0 - (i * 0.1))
                    break
            else:
                # Not in top 10, zero reward
                rewards.append(0.0)
        except Exception as e:
            print(f"Error in stockfish evaluation: {e}")
            rewards.append(0.0)
    
    # Clean up CUDA cache 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    # Check for both strict and loose formats
    strict_pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>"
    loose_pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    
    responses = [completion[0]["content"] for completion in completions]
    strict_matches = [re.search(strict_pattern, r, re.DOTALL) is not None for r in responses]
    loose_matches = [re.search(loose_pattern, r, re.DOTALL) is not None for r in responses]
    
    # Print truncated first response to debug format issues
    if responses:
        print(f"\n--- Format Reward ---")
        response_sample = responses[0][:150] + "..." if len(responses[0]) > 150 else responses[0]
        print(f"Response (truncated):\n{response_sample}")
        print(f"Has strict format: {strict_matches[0]}")
        print(f"Has loose format: {loose_matches[0]}")
    
    # Clean up CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Give higher reward for strict format, but still reward loose format
    return [1.0 if strict_match else (0.5 if loose_match else 0.0) 
            for strict_match, loose_match in zip(strict_matches, loose_matches)]

def correctness_reward_func(prompts, completions, next_move, fen, **kwargs) -> list[float]:
    """
    Reward function that evaluates if the predicted move matches the actual game move.
    Always reward 1.0 for a correct match, regardless of Stockfish evaluation.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    # Debug output for first example only
    if responses and extracted_moves and next_move:
        print(f"\n--- Correctness Reward ---")
        print(f"Extracted move: '{extracted_moves[0]}'")
        print(f"Expected move: '{next_move[0]}'")
        print(f"Match: {extracted_moves[0].strip() == next_move[0].strip()}")
    
    rewards = []
    for move, nm in zip(extracted_moves, next_move):
        if move.strip() == nm.strip():
            # Correct match with actual game move gets full reward
            rewards.append(1.0)
            if len(rewards) == 1:  # Only for first example
                print("Correct match with game move, reward: 1.0")
        else:
            rewards.append(0.0)  # No reward if predicted move doesn't match game move
            if len(rewards) == 1:  # Only for first example
                print("No match with game move, reward: 0.0")
    
    # Clean up CUDA cache after batch processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
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
    
    # Debug first example with minimal output
    if responses:
        print(f"\n--- XML Tag Count Reward ---")
        print(f"Tag reward: {rewards[0]} (has {int(rewards[0]/0.125)} tags)")
    
    # Clean up CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
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

def main(ablation=False, disable_multiprocessing=False, single_gpu=False, tiny_mode=False):
    """
    Main training function with optional ablation mode.
    
    Args:
        ablation: If True, only train with format and legality rewards,
                 ignoring move quality rewards from Stockfish.
        disable_multiprocessing: If True, disable multiprocessing for dataset preparation.
        single_gpu: If True, force only using a single GPU (GPU 0) even if multiple are available.
        tiny_mode: If True, use extreme memory optimization settings with tiny dataset, batch size and model config.
    """
    # Set extremely aggressive CUDA memory allocation settings to avoid fragmentation and OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:16,garbage_collection_threshold:0.8"
    
    # Limit to single GPU if requested
    if single_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Forcing single GPU mode. Only using GPU 0 out of {torch.cuda.device_count()} available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Free up GPU memory before starting training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set low memory limits for CUDA/cuDNN workspace
        torch.backends.cudnn.benchmark = False  # Disable benchmarking to save memory
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster/reduced memory
    
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
    model_name = "nvidia/AceInstruct-7B"
    
    # Prepare output directory with ablation indicator if needed
    if ablation:
        output_dir = "outputs/Chess-GRPO-Ablation"
        run_name = "Chess-GRPO-Ablation-Training"
    else:
        output_dir = "outputs/Chess-GRPO"
        run_name = "Chess-GRPO-Training"
    
    # Training configuration with extreme memory optimization
    if tiny_mode:
        # Ultra minimal config for tiny mode
        training_args = GRPOConfig(
            output_dir=output_dir,
            run_name=f"{run_name}-TinyMode",
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            logging_steps=1,
            fp16=True,                           # Use FP16 instead of BF16
            per_device_train_batch_size=1,       # Keep batch size at minimum
            gradient_accumulation_steps=1,       # No gradient accumulation for tiny mode
            num_generations=2,                   # Must be divisible into global batch size
            max_prompt_length=128,               # Ultra short context
            max_completion_length=128,           # Ultra short output
            num_train_epochs=1,
            save_strategy="no",                  # Don't save checkpoints in tiny mode
            save_total_limit=1,
            save_safetensors=True,
            load_best_model_at_end=False,        # Don't load best model to save memory
            eval_strategy="steps", 
            eval_steps=10,                       # Evaluate more frequently
            max_grad_norm=1.0,
            report_to="wandb",
            log_on_each_node=False,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=0,
            gradient_checkpointing=False,
            optim="adamw_torch",
            log_completions=False,
        )
    else:
        # Standard memory-optimized config
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
            bf16=False,
            per_device_train_batch_size=1,       # Keep batch size very small
            gradient_accumulation_steps=16,      # Doubled to 16 to reduce memory pressure
            num_generations=2,                   # Must be divisible into global batch size
            max_prompt_length=1024,               # Drastically reduced to save memory
            max_completion_length=1024,           # Drastically reduced to save memory
            num_train_epochs=1,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,                  # Keep only 1 checkpoint to save space
            save_safetensors=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_reward",
            greater_is_better=True,
            eval_strategy="steps", 
            eval_steps=100,
            max_grad_norm=1.0,
            report_to="wandb",
            log_on_each_node=False,
            ddp_find_unused_parameters=False,    # Set to False to avoid extra memory usage
            dataloader_num_workers=0,            # Avoid parallelism to reduce memory usage
            gradient_checkpointing=False,        # Disable gradient checkpointing - causing issues
            optim="adamw_torch",                 # Use PyTorch's memory-efficient AdamW
            log_completions=False,               # Disable completion logging to save memory
            # FP16 mixed precision to save memory
            fp16=True                           # Use FP16 instead of BF16 to save memory
        )
    
    # Print out the memory optimization settings
    print(f"Memory optimization level: {'EXTREME (tiny mode)' if tiny_mode else 'HIGH'}")
    
    # PEFT configuration - more lightweight LoRA config for memory efficiency
    if tiny_mode:
        # Ultra minimal LoRA for tiny mode
        peft_config = LoraConfig(
            r=4,                         # Even smaller rank
            lora_alpha=16,               # Reduced alpha
            target_modules=["q_proj"],   # Only target one module type to save memory
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
    else:
        # Regular memory-optimized LoRA
        peft_config = LoraConfig(
            r=8,                         # Reduced rank to 8 to save memory
            lora_alpha=32,               # Reduced alpha
            target_modules=["q_proj", "v_proj"],  # Only target some attention modules to save memory
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
    
    # Load the model with extreme memory-optimized settings
    if tiny_mode:
        # Ultra lightweight config for tiny mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for lower memory
            use_cache=True, 
            low_cpu_mem_usage=True,
            device_map=None,
            max_memory={0: "8GiB"},     # Even stricter memory limit
            load_in_8bit=True,          # Load in 8-bit for extreme memory savings
            offload_folder="offload",
        )
    else:
        # Standard memory-optimized config
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for lower memory
            use_cache=True,             # Enable caching when not using gradient checkpointing
            low_cpu_mem_usage=True,     # Optimize CPU memory usage during loading
            device_map=None,            # Let the accelerator handle device mapping
            max_memory={0: "12GiB"},    # Explicitly limit GPU memory usage
            offload_folder="offload",   # Set up offload folder for CPU offloading if needed
        )
    # Explicitly set model to train mode
    model.train()
    
    # Only set trainable parameters to require gradients, to save memory
    for name, param in model.named_parameters():
        if any(target in name for target in ["q_proj", "v_proj"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Force aggressive garbage collection and empty cache to ensure clean memory state
    import gc
    gc.collect()
    
    # Create offload directory if it doesn't exist
    os.makedirs("offload", exist_ok=True)
    
    if torch.cuda.is_available():
        # Multiple rounds of memory cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Make sure CUDA operations are complete
        gc.collect()
        torch.cuda.empty_cache()  # Clean up before training
        
    # Set a very conservative memory fraction
    if torch.cuda.is_available():
        # Reserve 20% of memory for system/other processes
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)
            
        # Extra memory cleanup
        torch.cuda.empty_cache()
        
        # Set memory allocator to be very conservative
        if hasattr(torch.cuda, 'memory_stats'):
            print(f"CUDA memory before optimization: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
            torch.cuda.empty_cache()
            print(f"CUDA memory after empty_cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
        
    print(f"Loading model: {model_name} with memory optimizations, use_cache=False")
    
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset - use multiprocessing unless explicitly disabled
    dataset = prepare_chess_dataset(use_multiprocessing=not disable_multiprocessing)
    
    # Only reduce dataset in tiny mode
    if tiny_mode:
        print(f"TINY MODE ENABLED: Original dataset size: {len(dataset)}")
        # Take a small subset for debugging memory issues
        dataset = dataset.select(range(min(50, len(dataset))))
        print(f"Reduced dataset to {len(dataset)} samples for extreme memory optimization")
    else:
        # Use full dataset for real training
        print(f"Using full dataset with {len(dataset)} samples")
    
    # Create validation split
    if len(dataset) > 100:  # Only split if we have enough data
        # Shuffle the dataset indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Take ~10% for validation
        val_size = max(10, min(len(dataset) // 10, 1000))  # At least 10, at most 1000
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
    
    # Initialize the trainer with extreme memory optimizations
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[
            # Add early stopping with shorter patience
            transformers.EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )
    
    # Print command for running with single process
    print("\nRecommended command for running with single process to avoid batch size issues:")
    print("accelerate launch --num_processes=1 chess_grpo.py --single-gpu --tiny")
    
    # More extreme memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    try:
        # Save checkpoint with timestamp (to be able to reference this run later)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        final_checkpoint_dir = os.path.join(output_dir, f"final-checkpoint-{timestamp}")
        
        # Create the directory first
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        
        trainer.save_model(final_checkpoint_dir)
        print(f"Final model saved to: {final_checkpoint_dir}")
        
        # Save a small metadata file with information about the run
        metadata = {
            "timestamp": timestamp,
            "model_name": model_name,
            "ablation": ablation,
            "tiny_mode": tiny_mode,
            "batch_size": training_args.per_device_train_batch_size,
            "num_generations": training_args.num_generations,
            "dataset_size": len(train_dataset),
        }
        
        # Save metadata to a JSON file
        import json
        with open(os.path.join(final_checkpoint_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        # Don't let saving failures stop the training from completing
        print(f"Warning: Could not save final checkpoint: {e}")
    
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
    parser.add_argument("--single-gpu", action="store_true",
                        help="Force using only a single GPU (GPU 0) even if multiple are available")
    parser.add_argument("--tiny", action="store_true",
                        help="Use extreme memory optimizations with tiny model for debugging")
    args = parser.parse_args()
    
    if args.test_parsing:
        test_pgn_parsing()
    else:
        # Use Accelerate to handle distribution - just call main directly
        main(ablation=args.ablation, 
             disable_multiprocessing=args.seq,
             single_gpu=args.single_gpu,
             tiny_mode=args.tiny)