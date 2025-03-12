#!/usr/bin/env python3
"""
Test script to verify Stockfish functionality used in chess_grpo.py
This tests all Stockfish features our GRPO training relies on.
"""

import chess
import chess.pgn
import io
import sys
import os
from stockfish import Stockfish, StockfishException

# Path to Stockfish executable - replace with your path if needed
STOCKFISH_PATH = "/usr/games/stockfish"

def test_stockfish_availability():
    """Test Stockfish availability and version."""
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH)
        version = stockfish.get_stockfish_major_version()
        print(f"✅ Stockfish is available (Version: {version})")
        return True
    except Exception as e:
        print(f"❌ Stockfish not available: {e}")
        print(f"Make sure Stockfish is installed and the path '{STOCKFISH_PATH}' is correct.")
        return False

def test_get_top_moves(n_moves=20):
    """Test getting top N moves from different positions."""
    stockfish = Stockfish(path=STOCKFISH_PATH)
    
    # Test cases: (FEN, description)
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "After 1.e4 e5 2.Nf3 Nc6"),
        ("r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6", "Giuoco Piano position")
    ]
    
    print(f"\n=== Testing get_top_moves({n_moves}) ===")
    
    for fen, description in test_positions:
        print(f"\nPosition: {description}")
        print(f"FEN: {fen}")
        
        stockfish.set_fen_position(fen)
        
        try:
            # Test getting top moves
            top_moves = stockfish.get_top_moves(n_moves)
            
            # Check if we got moves back
            if not top_moves:
                print(f"❌ No moves returned for position")
                continue
                
            print(f"✅ get_top_moves({n_moves}) returned {len(top_moves)} moves")
            
            # Display top 5 moves
            print("Top 5 moves:")
            for i, move_info in enumerate(top_moves[:5]):
                if i >= len(top_moves):
                    break
                centipawn = move_info.get('Centipawn', 'N/A')
                mate = move_info.get('Mate', 'N/A')
                print(f"  {i+1}. {move_info['Move']} (Centipawn: {centipawn}, Mate: {mate})")
            
            # Verify the format of returned data
            first_move = top_moves[0]
            if 'Move' in first_move:
                print(f"✅ 'Move' key exists in returned data")
            else:
                print(f"❌ 'Move' key missing from returned data")
                
            # Check for centipawn or mate score
            if 'Centipawn' in first_move or 'Mate' in first_move:
                print(f"✅ Evaluation scores present")
            else:
                print(f"❌ Missing evaluation scores")
        
        except Exception as e:
            print(f"❌ Error getting top moves: {e}")

def test_move_validation():
    """Test move validation using is_move_correct."""
    stockfish = Stockfish(path=STOCKFISH_PATH)
    
    # Test cases: (FEN, valid_moves, invalid_moves)
    test_cases = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ["e2e4", "d2d4", "g1f3"],  # Valid moves
            ["e2e5", "a2a5", "h1h3"]   # Invalid moves
        ),
        (
            "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
            ["e1g1", "d2d3", "c4d5"],  # Valid moves
            ["e1e2", "c4c5", "a1a3"]   # Invalid moves
        )
    ]
    
    print(f"\n=== Testing Move Validation ===")
    
    for fen, valid_moves, invalid_moves in test_cases:
        print(f"\nPosition FEN: {fen}")
        stockfish.set_fen_position(fen)
        
        print("Testing valid moves:")
        for move in valid_moves:
            is_valid = stockfish.is_move_correct(move)
            result = "✅" if is_valid else "❌"
            print(f"  {result} {move} -> {is_valid}")
            
        print("Testing invalid moves:")
        for move in invalid_moves:
            is_valid = stockfish.is_move_correct(move)
            result = "❌" if is_valid else "✅"  # For invalid moves, we expect False
            print(f"  {result} {move} -> {is_valid}")

def test_san_to_uci_conversion():
    """Test conversion between SAN and UCI move formats."""
    print(f"\n=== Testing SAN to UCI Conversion ===")
    
    # Test cases: (FEN, SAN moves, expected UCI moves)
    test_cases = [
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ["e4", "Nf3", "d4", "O-O"],
            ["e2e4", "g1f3", "d2d4", "e1g1"]
        ),
        (
            "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
            ["O-O", "Qe2", "Bb3", "exd5"],
            ["e1g1", "d1e2", "c4b3", "e4d5"]
        )
    ]
    
    for fen, san_moves, expected_uci in test_cases:
        print(f"\nPosition FEN: {fen}")
        board = chess.Board(fen)
        
        for san, expected in zip(san_moves, expected_uci):
            try:
                # Convert SAN to UCI using python-chess
                move = board.parse_san(san)
                uci = move.uci()
                
                # Check if conversion matches expected
                matches = uci == expected
                result = "✅" if matches else "❌"
                print(f"  {result} {san} -> {uci} (expected: {expected})")
                
                # Create a copy of the board and make the move
                test_board = board.copy()
                test_board.push(move)
                
                # Test if Stockfish accepts this move as valid
                stockfish = Stockfish(path=STOCKFISH_PATH)
                stockfish.set_fen_position(fen)
                is_valid_stockfish = stockfish.is_move_correct(uci)
                valid_result = "✅" if is_valid_stockfish else "❌"
                print(f"    {valid_result} Stockfish validation: {is_valid_stockfish}")
                
            except Exception as e:
                print(f"  ❌ Error converting {san}: {e}")

def test_our_reward_functions():
    """Test the reward calculation functions used in our GRPO implementation."""
    print(f"\n=== Testing Our Reward Functions ===")
    
    # Test exponential decay calculation
    print("\nTesting exponential decay reward function:")
    for i in range(20):
        reward = max(0.1, 1.0 * (0.95 ** i))
        print(f"  Move rank {i+1}: reward = {reward:.3f}")
    
    # Test Stockfish evaluation combining our functions
    print("\nTesting Stockfish evaluation with our reward function:")
    stockfish = Stockfish(path=STOCKFISH_PATH)
    
    # Test positions with known good moves
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", ["e2e4", "d2d4"], "Starting position"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", ["f1c4", "d2d4"], "Open position"),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5", ["e8g8", "d7d6"], "Middle game")
    ]
    
    for fen, test_moves, description in test_positions:
        print(f"\nPosition: {description}")
        stockfish.set_fen_position(fen)
        
        try:
            # Get top 20 moves from Stockfish
            top_moves = stockfish.get_top_moves(20)
            
            if not top_moves:
                print(f"❌ No moves returned for position")
                continue
                
            # Display top 5 suggested moves
            print("Stockfish top 5 moves:")
            for i, move_info in enumerate(top_moves[:5]):
                print(f"  {i+1}. {move_info['Move']} (Centipawn: {move_info.get('Centipawn', 'N/A')})")
            
            # Test our test moves against Stockfish rankings
            print("Testing our moves:")
            for move in test_moves:
                # Find position in Stockfish ranking
                found = False
                for i, move_info in enumerate(top_moves):
                    if move.lower() == move_info['Move'].lower():
                        # Calculate reward using our formula
                        reward = max(0.1, 1.0 * (0.95 ** i))
                        print(f"  ✅ {move} found at rank {i+1}, reward: {reward:.3f}")
                        found = True
                        break
                
                if not found:
                    # Check if move is at least legal
                    is_legal = stockfish.is_move_correct(move)
                    if is_legal:
                        print(f"  ⚠️ {move} is legal but not in top 20, reward: 0.05")
                    else:
                        print(f"  ❌ {move} is not legal, reward: 0.0")
        
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")

def main():
    """Run all tests."""
    print("=== Stockfish Testing for GRPO Chess ===")
    
    # Test Stockfish availability
    if not test_stockfish_availability():
        print("Cannot continue tests without Stockfish available.")
        return 1
    
    # Run tests
    test_get_top_moves(20)  # Test getting top 20 moves
    test_move_validation()  # Test move validation
    test_san_to_uci_conversion()  # Test SAN to UCI conversion
    test_our_reward_functions()  # Test our reward functions
    
    print("\n=== Testing Complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())