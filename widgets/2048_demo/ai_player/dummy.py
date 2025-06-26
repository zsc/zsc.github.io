# 2048_demo/ai_player/dummy.py
import random
# To test, we need access to the Game class.
# This is a bit awkward for standalone testing if Game is in a sibling directory.
# For proper module structure, you might run tests from the project root.
# For simple in-file test:
try:
    from game import Game # If running test from 2048_demo/ai_player/
except ImportError:
    import sys
    sys.path.append('..') # Add parent directory to path
    from game import Game # If running test from 2048_demo/

class DummyAI:
    def __init__(self):
        pass

    def get_move(self, game_instance):
        """
        Returns a random legal move.
        :param game_instance: An instance of the Game class.
        :return: A string representing the move ('up', 'down', 'left', 'right') or None if no moves.
        """
        legal_moves = game_instance.get_legal_moves()
        if legal_moves:
            return random.choice(legal_moves)
        return None

# Self-contained tests
if __name__ == "__main__":
    test_game = Game()
    ai = DummyAI()

    # Test 1: Game with possible moves
    test_game.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    test_game.score = 0
    # Manually ensure history is set for consistency if game.move modifies it
    test_game.history = [(test_game.board, test_game.score)]
    
    print("Test AI with board:")
    for r in test_game.board: print(r)
    
    move = ai.get_move(test_game)
    print(f"Dummy AI suggests move: {move}")
    assert move in ['down', 'right'], f"Expected 'down' or 'right', got {move}"

    # Test 2: Game with no possible moves (full board, no merges)
    test_game.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 16]]
    test_game.history = [(test_game.board, test_game.score)]

    print("\nTest AI with full board (no moves):")
    for r in test_game.board: print(r)

    move = ai.get_move(test_game)
    print(f"Dummy AI suggests move: {move}")
    assert move is None, f"Expected None, got {move}"
    
    # Test 3: Game with mergeable moves
    test_game.board = [[2, 2, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    test_game.history = [(test_game.board, test_game.score)]
    print("\nTest AI with mergeable board:")
    for r in test_game.board: print(r)
    move = ai.get_move(test_game)
    print(f"Dummy AI suggests move: {move}")
    # Expected moves: 'left' (merges 2s), 'right' (moves 2s right), 'down' (moves 4 down), 'up' (moves 4 up)
    # Actually, up will not change the first 2,2. Only 'left', 'right', 'down'
    expected_moves = test_game.get_legal_moves() # get actual legal moves
    assert move in expected_moves, f"Expected one of {expected_moves}, got {move}"


    print("\nDummyAI tests passed.")
