import torch
import chess
import chess.pgn
import io
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from stockfish import Stockfish
import random
import os

# Check if stockfish is available
try:
    stockfish = Stockfish()
    STOCKFISH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Stockfish not available - {e}")
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
        except:
            return False

def evaluate_move(move_str: str, fen: str, stockfish_instance: Stockfish) -> float:
    """Evaluate a move using Stockfish."""
    if not STOCKFISH_AVAILABLE:
        return 0.0
    
    try:
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
    """Load a chess game from a PGN file."""
    with open(pgn_file_path, 'r') as f:
        game = chess.pgn.read_game(f)
        while game is not None:
            yield game
            game = chess.pgn.read_game(f)

def get_game_sequence(game, split_percentage=None):
    """
    Process a game and create a sequence for prediction.
    
    Splits the game at a position that is 50-80% through the game,
    returning the sequence up to that point and the next move to predict.
    """
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
        if is_white_move:
            sequence.append(f"{move_number}.")
        
        move_san = board.san(move)
        sequence.append(move_san)
        board.push(move)
        
        # Update move counters for next iteration
        if not is_white_move:
            move_number += 1
        is_white_move = not is_white_move
    
    # The move to predict is the next move after the split
    if split_index < len(moves):
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
        
        return ' '.join(sequence), next_move, board.fen()
    
    return None, None, None

def prepare_chess_dataset():
    """Prepare the chess dataset from local PGN files."""
    # Get all PGN files in the games directory
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    pgn_files = [os.path.join(games_dir, f) for f in os.listdir(games_dir) if f.endswith('.pgn')]
    
    # Process the PGN games
    all_samples = []
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        for game in load_game_from_pgn_file(pgn_file):
            sequence, next_move, fen = get_game_sequence(game)
            if sequence and next_move and fen:
                all_samples.append({
                    "position": sequence,
                    "fen": fen,
                    "next_move": next_move
                })
    
    # Randomly shuffle the dataset
    random.shuffle(all_samples)
    
    # Format the dataset for the GRPO trainer
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
    
    print(f"Created dataset with {len(formatted_data)} samples")
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
    
    stockfish_instance = Stockfish()
    
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

def correctness_reward_func(prompts, completions, next_move, **kwargs) -> list[float]:
    """Reward function that checks if the move exactly matches the one played in the game."""
    responses = [completion[0]["content"] for completion in completions]
    extracted_moves = [extract_xml_answer(r) for r in responses]
    
    return [1.0 if move.strip() == nm.strip() else 0.0 
            for move, nm in zip(extracted_moves, next_move)]

def main():
    # Model and training configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Prepare output directory
    output_dir = "outputs/Chess-GRPO"
    run_name = "Chess-GRPO-Training"
    
    # Training configuration
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=512,  # Chess games can be long
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )
    
    # PEFT configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the dataset
    dataset = prepare_chess_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            legal_move_reward_func,
            stockfish_reward_func, 
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()

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

if __name__ == "__main__":
    # Comment/uncomment to run the desired function
    # test_pgn_parsing()
    main()