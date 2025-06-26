# game.py
import random
import torch # For PyTorch operations
import time

class Game:
    DIRECTIONS_MAP = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    ACTION_LIST = ['up', 'down', 'left', 'right'] # Corresponds to indices 0,1,2,3

    def __init__(self, size=4, batch_size=1, device=None):
        self.size = size
        self.batch_size = batch_size
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Board values are tile numbers (2, 4, 8, ...). bfloat16 can represent small integers exactly.
        self.boards = torch.zeros((batch_size, size, size), dtype=torch.bfloat16, device=self.device)
        self.scores = torch.zeros(batch_size, dtype=torch.int32, device=self.device) # Scores are integers
        
        # For compatibility with app.py's single-game mode that used history for undo.
        # Batched undo is complex and not implemented.
        self.history_stub = [] 

        self.reset()

    def reset(self, reset_mask=None):
        """
        Resets game states.
        If reset_mask (boolean tensor of shape [batch_size]) is provided,
        only resets the specified games. Otherwise, resets all games.
        """
        if reset_mask is None:
            effective_reset_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        else:
            effective_reset_mask = reset_mask
            if not effective_reset_mask.any():
                return # Nothing to reset

        num_to_reset = effective_reset_mask.sum().item()

        if num_to_reset == self.batch_size: # Resetting all
            self.boards.zero_()
            self.scores.zero_()
            self._add_random_tile() # Add to all
            self._add_random_tile() # Add to all
        elif num_to_reset > 0: # Resetting a subset
            self.boards[effective_reset_mask].zero_()
            self.scores[effective_reset_mask].zero_()
            # Add tiles only to the reset boards
            self._add_random_tile(add_to_mask=effective_reset_mask)
            self._add_random_tile(add_to_mask=effective_reset_mask)


    def _compress_lines_gpu(self, lines: torch.Tensor):
        # lines: (N, size), where N is num_lines (e.g., batch_size * size)
        # Shifts non-zero elements to the left.
        temp_lines = lines.clone()
        for _ in range(self.size -1): # Number of passes needed to shift all zeros to the right
            for c_idx in range(self.size - 1): # Columns to check for zero
                mask_shift = (temp_lines[:, c_idx] == 0) & (temp_lines[:, c_idx+1] != 0)
                if mask_shift.any(): # Only perform assignment if there's something to shift
                    temp_lines[mask_shift, c_idx] = temp_lines[mask_shift, c_idx+1]
                    temp_lines[mask_shift, c_idx+1] = 0
        return temp_lines

    def _process_lines(self, lines: torch.Tensor):
        # lines: (N, size) tensor
        # Returns: final_lines (N, size), line_score_increases (N,), lines_moved_mask (N,)
        original_lines = lines.clone()
        
        # 1. Compress
        current_lines = self._compress_lines_gpu(lines)
        
        # 2. Merge
        line_score_increases = torch.zeros(lines.shape[0], dtype=torch.int32, device=self.device)
        for j in range(self.size - 1): # Iterate through columns j to j+1 for merging
            mask_merge = (current_lines[:, j] != 0) & (current_lines[:, j] == current_lines[:, j+1])
            if mask_merge.any():
                current_lines[mask_merge, j] *= 2
                line_score_increases[mask_merge] += current_lines[mask_merge, j].int() # Scores are int
                current_lines[mask_merge, j+1] = 0
            
        # 3. Compress again
        final_lines = self._compress_lines_gpu(current_lines)
        
        lines_moved_mask = ~torch.all(final_lines == original_lines, dim=1)
        return final_lines, line_score_increases, lines_moved_mask

    def _add_random_tile(self, add_to_mask=None):
        # add_to_mask: (batch_size,) boolean tensor, indicates which boards to add a tile to.
        # If None, attempts to add to all boards.
        if add_to_mask is None:
            effective_add_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        else:
            effective_add_mask = add_to_mask

        # Identify boards that are part of the effective_add_mask AND have empty cells
        empty_cell_counts_per_board = (self.boards == 0).view(self.batch_size, -1).sum(dim=1)
        eligible_for_add_mask = effective_add_mask & (empty_cell_counts_per_board > 0)

        if not eligible_for_add_mask.any():
            return False # No tile added to any board that was eligible

        eligible_indices = eligible_for_add_mask.nonzero(as_tuple=True)[0]
        num_eligible_to_add = eligible_indices.shape[0]

        target_boards_for_tile_add = self.boards[eligible_indices]
        
        # Create probability distribution for sampling empty cells on these target_boards
        probs = (target_boards_for_tile_add == 0).view(num_eligible_to_add, -1).float()
        probs_sum = probs.sum(dim=1, keepdim=True)
        
        actually_has_empty_mask = probs_sum.squeeze(1) > 0
        if not actually_has_empty_mask.any():
             # This can happen if, for example, a board was in eligible_for_add_mask
             # but between then and now (e.g. in a concurrent setup, though not here)
             # or due to a bug, it became full.
             # Or more likely, if float precision with probs_sum led to issues for very sparse boards,
             # though sum > 0 should catch actual empty cells.
            return False 

        final_eligible_indices_in_probs = actually_has_empty_mask.nonzero(as_tuple=True)[0]
        
        global_indices_for_final_add = eligible_indices[final_eligible_indices_in_probs]
        num_final_add = global_indices_for_final_add.shape[0]

        if num_final_add == 0: return False

        probs_to_sample = probs[final_eligible_indices_in_probs]
        # It's possible probs_sum_to_sample contains zeros if a board had no empty cells after all
        # This is guarded by actually_has_empty_mask and num_final_add check.
        probs_sum_to_sample = probs_sum[final_eligible_indices_in_probs] 
        
        probs_normalized = probs_to_sample / probs_sum_to_sample # Safe due to checks
        
        flat_chosen_indices = torch.multinomial(probs_normalized, 1).squeeze(1)
        
        chosen_rows = flat_chosen_indices // self.size
        chosen_cols = flat_chosen_indices % self.size
        
        rand_values_for_4 = torch.rand(num_final_add, device=self.device) < 0.1
        new_tile_values = torch.where(rand_values_for_4, 
                                      torch.tensor(4.0, device=self.device, dtype=self.boards.dtype), 
                                      torch.tensor(2.0, device=self.device, dtype=self.boards.dtype))
        
        self.boards[global_indices_for_final_add, chosen_rows, chosen_cols] = new_tile_values
        return True

    def move(self, actions: torch.Tensor):
        # actions: (batch_size,) tensor of action indices (0:up, 1:down, 2:left, 3:right)
        original_boards_clone = self.boards.clone()
        
        transformed_boards_for_lines = torch.empty_like(self.boards)

        up_mask = (actions == 0)
        if up_mask.any():
            transformed_boards_for_lines[up_mask] = self.boards[up_mask].transpose(1, 2)
        
        down_mask = (actions == 1)
        if down_mask.any():
            transformed_boards_for_lines[down_mask] = torch.flip(self.boards[down_mask].transpose(1,2), dims=[2])

        left_mask = (actions == 2)
        if left_mask.any():
            transformed_boards_for_lines[left_mask] = self.boards[left_mask]

        right_mask = (actions == 3)
        if right_mask.any():
            transformed_boards_for_lines[right_mask] = torch.flip(self.boards[right_mask], dims=[2])
        
        lines_to_process = transformed_boards_for_lines.reshape(self.batch_size * self.size, self.size)
        processed_lines, line_score_increases, _ = self._process_lines(lines_to_process)
        
        self.scores += line_score_increases.reshape(self.batch_size, self.size).sum(dim=1)
        
        processed_boards_transformed_view = processed_lines.reshape(self.batch_size, self.size, self.size)

        if up_mask.any():
            self.boards[up_mask] = processed_boards_transformed_view[up_mask].transpose(1,2)
        if down_mask.any():
            self.boards[down_mask] = torch.flip(processed_boards_transformed_view[down_mask], dims=[2]).transpose(1,2)
        if left_mask.any():
            self.boards[left_mask] = processed_boards_transformed_view[left_mask]
        if right_mask.any():
            self.boards[right_mask] = torch.flip(processed_boards_transformed_view[right_mask], dims=[2])

        board_changed_mask = ~torch.all(self.boards.view(self.batch_size, -1) == original_boards_clone.view(self.batch_size, -1), dim=1)
        
        if board_changed_mask.any():
            self._add_random_tile(add_to_mask=board_changed_mask)
            
        return board_changed_mask

    def is_game_over(self):
        has_empty_cells_mask = torch.any(self.boards.view(self.batch_size, -1) == 0, dim=1)
        can_merge_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        non_zero_boards = self.boards != 0 # To ensure merges only happen with actual tiles
        # Horizontal merges
        h_merge_possible = (self.boards[:, :, :-1] == self.boards[:, :, 1:]) & \
                           non_zero_boards[:, :, :-1] & non_zero_boards[:, :, 1:]
        can_merge_mask |= torch.any(h_merge_possible.view(self.batch_size, -1), dim=1)

        # Vertical merges
        v_merge_possible = (self.boards[:, :-1, :] == self.boards[:, 1:, :]) & \
                           non_zero_boards[:, :-1, :] & non_zero_boards[:, 1:, :]
        can_merge_mask |= torch.any(v_merge_possible.view(self.batch_size, -1), dim=1)
        
        game_over_final_mask = ~has_empty_cells_mask & ~can_merge_mask
        return game_over_final_mask

    def get_legal_moves(self):
        legal_moves_matrix = torch.zeros((self.batch_size, 4), dtype=torch.bool, device=self.device)
        original_boards_temp = self.boards.clone() 

        for i_action in range(4): 
            transformed_boards_for_lines_sim = torch.empty_like(original_boards_temp) # Ensure this is on the right device
            
            current_boards_to_transform = original_boards_temp # Use the clone for each simulation
            if i_action == 0: 
                transformed_boards_for_lines_sim = current_boards_to_transform.transpose(1, 2)
            elif i_action == 1: 
                transformed_boards_for_lines_sim = torch.flip(current_boards_to_transform.transpose(1,2), dims=[2])
            elif i_action == 2: 
                transformed_boards_for_lines_sim = current_boards_to_transform
            elif i_action == 3: 
                transformed_boards_for_lines_sim = torch.flip(current_boards_to_transform, dims=[2])
            
            lines_to_process_sim = transformed_boards_for_lines_sim.reshape(self.batch_size * self.size, self.size)
            _, _, lines_moved_mask_flat = self._process_lines(lines_to_process_sim) 
            
            board_would_change_mask = lines_moved_mask_flat.reshape(self.batch_size, self.size).any(dim=1)
            legal_moves_matrix[:, i_action] = board_would_change_mask
            
        return legal_moves_matrix

    def get_board(self):
        if self.batch_size == 1:
            return self.boards[0].cpu().tolist()
        raise ValueError("get_board() called in batched mode without index. Use game.boards.")

    def get_score(self):
        if self.batch_size == 1:
            return self.scores[0].item()
        raise ValueError("get_score() called in batched mode without index. Use game.scores.")

    def undo(self): 
        if self.batch_size == 1 and self.history_stub:
            print("Warning: Undo is not fully supported in the new Game engine.")
            return False 
        return False
    
    def _store_state(self):
        if self.batch_size == 1:
            pass

    def __str__(self):
        if self.batch_size == 1:
            s = f"Score: {self.get_score()}\n"
            board_list = self.get_board()
            for row in board_list:
                s += "\t".join(map(str, [int(x) for x in row])) + "\n"
            return s
        else:
            return f"Batched Game (Batch Size: {self.batch_size}, Device: {self.device})"

# --- Unit Tests ---
if __name__ == "__main__":
    test_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Game Unit Tests on {test_device} ---")

    def assert_boards_equal(b1, b2, msg=""):
        # If b1 or b2 are Python lists, convert them to tensors.
        # Ensure they are on the `test_device` and have `torch.bfloat16` dtype
        # to match the game's board tensors for comparison.
        default_dtype = torch.bfloat16

        if isinstance(b1, list):
            b1 = torch.tensor(b1, dtype=default_dtype, device=test_device)
        if isinstance(b2, list):
            b2 = torch.tensor(b2, dtype=default_dtype, device=test_device)
        
        # At this point, b1 and b2 should be tensors.
        # If their devices or dtypes don't match, it might indicate an issue
        # in test setup or the function being tested. For robustness in assertion,
        # we can try to move/cast one to match the other, but it's better if they align.
        # For this specific problem, ensuring list conversions use test_device is key.
        if b1.device != b2.device:
            # This can happen if one tensor was already on a different device.
            # Make b2 match b1's device for comparison.
            b2 = b2.to(b1.device)
        if b1.dtype != b2.dtype:
            # Similarly, match dtype if they differ.
            b2 = b2.to(b1.dtype)

        # For clearer error messages with bfloat16, cast to float32 for printing.
        b1_is_tensor = isinstance(b1, torch.Tensor)
        b2_is_tensor = isinstance(b2, torch.Tensor)

        b1_print_val = b1.to(torch.float32) if b1_is_tensor and b1.dtype == torch.bfloat16 else b1
        b2_print_val = b2.to(torch.float32) if b2_is_tensor and b2.dtype == torch.bfloat16 else b2
        
        b1_info = f"device {b1.device}, dtype {b1.dtype}" if b1_is_tensor else "Non-Tensor"
        b2_info = f"device {b2.device}, dtype {b2.dtype}" if b2_is_tensor else "Non-Tensor"

        assert torch.all(b1.eq(b2)), f"{msg}\nExpected ({b1_info}):\n{b1_print_val}\nGot ({b2_info}):\n{b2_print_val}"


    # Test _compress_lines_gpu
    print("Testing _compress_lines_gpu...")
    game_compress_test = Game(batch_size=1, device=test_device) 
    lines = torch.tensor([[2,0,2,0], [0,0,4,4], [0,0,0,2], [8,4,2,0]], dtype=torch.bfloat16, device=test_device)
    compressed = game_compress_test._compress_lines_gpu(lines)
    expected_compressed = torch.tensor([[2,2,0,0], [4,4,0,0], [2,0,0,0], [8,4,2,0]], dtype=torch.bfloat16, device=test_device)
    assert_boards_equal(compressed, expected_compressed, "_compress_lines_gpu basic")
    print("_compress_lines_gpu tests passed.")

    # Test _process_lines
    print("Testing _process_lines...")
    game_process_test = Game(batch_size=1, device=test_device)
    lines = torch.tensor([[2,2,0,4], [0,2,2,4], [4,4,4,4], [2,0,0,2]], dtype=torch.bfloat16, device=test_device)
    final_lines, scores, moved = game_process_test._process_lines(lines)
    
    expected_final_lines = torch.tensor([[4,4,0,0], [4,4,0,0], [8,8,0,0], [4,0,0,0]], dtype=torch.bfloat16, device=test_device)
    expected_scores = torch.tensor([4, 4, 16, 4], dtype=torch.int32, device=test_device) 
    expected_moved = torch.tensor([True, True, True, True], dtype=torch.bool, device=test_device)
    
    assert_boards_equal(final_lines, expected_final_lines, "_process_lines lines")
    assert torch.all(scores.eq(expected_scores)), f"_process_lines scores. Expected {expected_scores}, Got {scores}"
    assert torch.all(moved.eq(expected_moved)), f"_process_lines moved. Expected {expected_moved}, Got {moved}"
    print("_process_lines tests passed.")

    # Test _add_random_tile
    print("Testing _add_random_tile...")
    game_add_tile = Game(batch_size=2, device=test_device, size=4) # Ensure size is explicit for clarity
    game_add_tile.boards.zero_() 
    game_add_tile.scores.zero_()
    game_add_tile._add_random_tile()
    assert (game_add_tile.boards != 0).view(2, -1).sum(dim=1).tolist() == [1,1], "Should add 1 tile to each board"
    
    game_add_tile.boards[0].fill_(2) # Fill board 0
    # Board 1 currently has 1 tile. Add another random tile.
    # It should only affect board 1 as board 0 is full (no empty cells).
    game_add_tile._add_random_tile() 
    
    num_nz_board0 = (game_add_tile.boards[0] != 0).sum().item()
    num_nz_board1 = (game_add_tile.boards[1] != 0).sum().item()
    assert num_nz_board0 == game_add_tile.size * game_add_tile.size, "Board 0 should remain full"
    # Board 1 had 1 tile, _add_random_tile was called.
    # If board 1 was not full, it should receive another tile.
    # The first call to _add_random_tile adds 1 tile to each board (so board 1 has 1 tile).
    # The second call to _add_random_tile (with no mask) tries to add to both.
    # Board 0 is full, so no tile added. Board 1 is not full, so it gets a tile.
    # Thus, board 1 should have 1 (initial) + 1 (new) = 2 tiles.
    assert num_nz_board1 == 2, f"Board 1 should have 2 tiles now. Has {num_nz_board1}"
    print("_add_random_tile tests passed.")

    # Test move
    print("Testing move...")
    game_move = Game(batch_size=1, device=test_device)
    game_move.boards[0] = torch.tensor([[2,2,0,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.bfloat16, device=test_device)
    game_move.scores[0] = 0
    
    game_move._deterministic_next_tile_val = 2.0 # Use float for bfloat16 consistency
    game_move._deterministic_next_tile_pos = (1,0) 
    def mock_add_random_tile(add_to_mask=None):
        if add_to_mask is None or add_to_mask[0]: 
            # Ensure assigned value is compatible with board's dtype and device
            tile_val_tensor = torch.tensor(game_move._deterministic_next_tile_val, 
                                           dtype=game_move.boards.dtype, 
                                           device=game_move.boards.device)
            game_move.boards[0, game_move._deterministic_next_tile_pos[0], game_move._deterministic_next_tile_pos[1]] = tile_val_tensor
        return True
    original_art = game_move._add_random_tile
    game_move._add_random_tile = mock_add_random_tile

    actions = torch.tensor([Game.DIRECTIONS_MAP['left']], device=test_device)
    moved_mask = game_move.move(actions)
    
    game_move._add_random_tile = original_art 

    expected_board_after_move = [[4,4,0,0],[2,0,0,0],[0,0,0,0],[0,0,0,0]] # This is a Python list
    assert_boards_equal(game_move.boards[0], expected_board_after_move, "Move left output")
    assert game_move.scores[0].item() == 4, f"Move left score. Expected 4, Got {game_move.scores[0].item()}"
    assert moved_mask[0].item() == True, "Move left should change board"
    print("Move test (left) passed.")

    # Test is_game_over
    print("Testing is_game_over...")
    game_over_test = Game(batch_size=3, device=test_device)
    game_over_test.boards[0] = torch.tensor([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,8]], dtype=torch.bfloat16, device=test_device)
    game_over_test.boards[1] = torch.tensor([[2,4,2,4],[4,2,4,2],[2,4,0,4],[4,2,4,8]], dtype=torch.bfloat16, device=test_device)
    game_over_test.boards[2] = torch.tensor([[2,2,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,8]], dtype=torch.bfloat16, device=test_device)
    
    over_mask = game_over_test.is_game_over()
    expected_over_mask = torch.tensor([True, False, False], dtype=torch.bool, device=test_device)
    assert torch.all(over_mask.eq(expected_over_mask)), f"is_game_over. Expected {expected_over_mask}, Got {over_mask}"
    print("is_game_over tests passed.")

    # Test get_legal_moves
    print("Testing get_legal_moves...")
    game_legal_test = Game(batch_size=2, device=test_device)
    game_legal_test.boards[0] = torch.tensor([[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.bfloat16, device=test_device)
    game_legal_test.boards[1] = torch.tensor([[2,4,8,16],[4,8,16,32],[8,16,32,64],[16,32,64,128]], dtype=torch.bfloat16, device=test_device)
    
    legal_matrix = game_legal_test.get_legal_moves()
    expected_legal_matrix = torch.tensor([[False, True, False, True], 
                                          [False, False, False, False]],
                                          dtype=torch.bool, device=test_device)
    assert torch.all(legal_matrix.eq(expected_legal_matrix)), f"get_legal_moves. Expected:\n{expected_legal_matrix}\nGot:\n{legal_matrix}"
    print("get_legal_moves tests passed.")

    print("Running basic speed test placeholder...")
    num_batches_speed = 100
    speed_batch_size = 128
    game_speed_test = Game(batch_size=speed_batch_size, device=test_device)
    
    start_time = time.perf_counter()
    for i in range(num_batches_speed):
        actions = torch.randint(0, 4, (speed_batch_size,), device=test_device)
        game_speed_test.move(actions)
        if (i+1) % 20 == 0: 
             game_speed_test.reset() 

    end_time = time.perf_counter()
    duration = end_time - start_time
    moves_per_sec = (num_batches_speed * speed_batch_size) / duration
    print(f"Speed test: {moves_per_sec:.2f} game moves per second on {test_device} (batch size {speed_batch_size}).")

    print("--- Game Unit Tests Completed ---")
