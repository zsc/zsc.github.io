#!/usr/bin/env python3
# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import collections
from typing import Tuple, List, Dict
import unittest

# Import the game logic. Assuming game.py is in the same directory.
from game import Game2048

# Initialize the game's lookup tables once.
Game2048._init_tables()

class PositionalEncoding(nn.Module):
    """ Standard positional encoding for Transformers. """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer model to estimate Q-values.
    Input: A sequence of 80 tokens representing the current state and the
           deterministic next states for each of the 4 actions.
           [current_board (16)] + [next_board_up (16)] + [next_board_down (16)] +
           [next_board_left (16)] + [next_board_right (16)]
    Output: 4 Q-values, one for each action.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, n_tokens: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(n_tokens, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=80)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # The output head projects the aggregated information to 4 Q-values
        self.output_head = nn.Linear(embed_dim, 4)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, 80]
        """
        # Embed tokens and add positional encoding
        x = self.token_embedding(src) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # Pass through transformer
        transformer_output = self.transformer_encoder(x)

        # Aggregate the output embeddings (e.g., by averaging) and get Q-values
        aggregated_output = transformer_output.mean(dim=1)
        q_values = self.output_head(aggregated_output)

        return q_values

class ReplayBuffer:
    """A simple FIFO experience replay buffer for a DQN agent."""
    def __init__(self, capacity: int):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Save an experience."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class Agent:
    """
    Double-DQN Agent for 2048.
    """
    def __init__(self,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.0,
                 epsilon_decay: float = 10000,
                 target_update_freq: int = 500,
                 buffer_size: int = 10000,
                 batch_size: int = 128,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = TransformerModel(embed_dim, num_heads, num_layers).to(self.device)
        self.target_net = TransformerModel(embed_dim, num_heads, num_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Action mapping
        self.moves = {
            0: Game2048.move_up,
            1: Game2048.move_down,
            2: Game2048.move_left,
            3: Game2048.move_right,
        }

    @staticmethod
    def _board_to_tokens(board: int) -> List[int]:
        """Converts a 64-bit board integer to a list of 16 tokens."""
        return [(board >> (i * 4)) & 0xF for i in range(16)]

    def _create_input_tensor(self, board: int) -> torch.Tensor:
        """
        Creates the 80-token input tensor for the transformer model from a single board.
        """
        tokens = self._board_to_tokens(board)
        for i in range(4):
            # Get the deterministic next state (before adding a random tile)
            next_b, _, moved = self.moves[i](board)
            if moved:
                tokens.extend(self._board_to_tokens(next_b))
            else:
                # For illegal moves, use the original board tokens instead of zeros
                tokens.extend(self._board_to_tokens(board))
        
        if len(tokens) != 80:
             raise ValueError(f"Token length error. Expected 80, got {len(tokens)}")

        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def get_legal_actions(self, board: int) -> List[int]:
        """Returns a list of legal actions (0-3) for a given board."""
        legal = []
        for i in range(4):
            _, _, moved = self.moves[i](board)
            if moved:
                legal.append(i)
        return legal

    def act(self, board: int, is_eval: bool = False) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        During evaluation, it always chooses the best action.
        """
        legal_actions = self.get_legal_actions(board)
        if not legal_actions:
            return -1 # No legal moves

        if is_eval:
            epsilon = -1.0 # Force greedy
        else:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      math.exp(-1. * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
        
        if random.random() > epsilon:
            with torch.no_grad():
                input_tensor = self._create_input_tensor(board).unsqueeze(0)
                q_values = self.policy_net(input_tensor)[0]
                
                # Mask illegal actions
                mask = torch.full((4,), float('-inf'), device=self.device)
                mask[legal_actions] = 0
                masked_q_values = q_values + mask
                
                return masked_q_values.argmax().item()
        else:
            return random.choice(legal_actions)

    def learn(self):
        """
        Performs one step of learning from the replay buffer.
        """
        if len(self.memory) < self.batch_size:
            return None # Not enough samples yet

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Create batch tensors
        state_tensors = torch.stack([self._create_input_tensor(s) for s in states])
        next_state_tensors = torch.stack([self._create_input_tensor(ns) for ns in next_states])
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Q(s, a) for the actions that were actually taken
        q_pred = self.policy_net(state_tensors).gather(1, action_batch)

        # DDQN target calculation
        with torch.no_grad():
            # Select best actions for next_states using the policy_net
            next_q_policy = self.policy_net(next_state_tensors)
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            
            # Evaluate those actions using the target_net
            next_q_target = self.target_net(next_state_tensors).gather(1, best_next_actions)
            
            # Compute the expected Q values
            target_q = reward_batch + (self.gamma * next_q_target.squeeze() * (1 - done_batch))

        # Compute loss (Huber loss is often more stable)
        loss = F.smooth_l1_loss(q_pred.squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Optional
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_net()
        
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
        print(f"Checkpoint loaded from {path}")

    def act_with_search(self, board: int, budget: int = 100000000, max_depth: int = 3) -> int:
        """
        Chooses an action by performing an Expectimax search. The search is
        pruned by a node visit budget and maximum depth, and leaf nodes are evaluated
        using the policy network as a heuristic.
        """
        legal_actions = self.get_legal_actions(board)
        if not legal_actions:
            return -1
        if len(legal_actions) == 1:
            return legal_actions[0]

        memo = {}
        nodes_visited = {'count': 0}

        def evaluate_heuristic(b: int) -> float:
            """Evaluates a board using the policy network and returns the max Q-value."""
            # Check if we have a custom heuristic function (for testing)
            if hasattr(self, '_get_heuristic_value'):
                return self._get_heuristic_value(b)
            
            with torch.no_grad():
                input_tensor = self._create_input_tensor(b).unsqueeze(0)
                q_values = self.policy_net(input_tensor)[0]
                return q_values.max().item()

        def expectimax_search(current_board: int, depth: int = 0) -> float:
            """
            Performs the recursive search.
            Returns the estimated value of the state.
            """
            # Memoization check
            if current_board in memo:
                return memo[current_board]

            # Budget, depth, and termination checks
            nodes_visited['count'] += 1
            if (nodes_visited['count'] > budget or 
                depth >= max_depth or 
                Game2048.is_game_over(current_board)):
                value = evaluate_heuristic(current_board)
                memo[current_board] = value
                return value

            # This is a MAX node (our turn to move)
            legal_actions = self.get_legal_actions(current_board)
            if not legal_actions:
                value = 0.0
                memo[current_board] = value
                return value

            best_action_value = float('-inf')
            for action in legal_actions:
                board_after_move, score, _ = self.moves[action](current_board)
                
                # This is a CHANCE node (environment places a tile)
                chance_node_value = calculate_chance_node_value(board_after_move, depth + 1)
                
                current_action_value = score + chance_node_value
                best_action_value = max(best_action_value, current_action_value)
            
            memo[current_board] = best_action_value
            return best_action_value
        
        def calculate_chance_node_value(board_after_move: int, depth: int) -> float:
            """
            Calculates the expected value of a chance node by averaging over
            all possible random tile placements.
            """
            # Early termination checks
            if (nodes_visited['count'] > budget or 
                depth >= max_depth):
                return evaluate_heuristic(board_after_move)
            
            empty_cells = [i for i in range(16) if ((board_after_move >> (4 * i)) & 0xF) == 0]
            if not empty_cells:
                # If the move fills the board, evaluate this state
                return expectimax_search(board_after_move, depth)

            # Limit the number of empty cells we consider to prevent explosion
            max_cells_to_consider = min(8, len(empty_cells))
            cells_to_consider = empty_cells[:max_cells_to_consider]

            total_expected_value = 0.0
            for pos in cells_to_consider:
                # Check budget before each recursive call
                if nodes_visited['count'] > budget:
                    break
                    
                # Possibility 1: a '2' tile (log2 value 1) is added (90% chance)
                board_with_2 = board_after_move | (1 << (4 * pos))
                value_if_2 = expectimax_search(board_with_2, depth)

                # Check budget again
                if nodes_visited['count'] > budget:
                    break

                # Possibility 2: a '4' tile (log2 value 2) is added (10% chance)
                board_with_4 = board_after_move | (2 << (4 * pos))
                value_if_4 = expectimax_search(board_with_4, depth)
                
                total_expected_value += (0.9 * value_if_2 + 0.1 * value_if_4)
            
            # The final value is the average over all considered empty cells
            return total_expected_value / len(cells_to_consider)

        # --- Main logic for act_with_search ---
        # Evaluate each initial legal action
        action_scores = {}
        for action in legal_actions:
            if nodes_visited['count'] > budget:
                break
                
            next_board, score, _ = self.moves[action](board)
            # The value of an action is the immediate score + the expected value of the resulting state
            chance_value = calculate_chance_node_value(next_board, 1)
            action_scores[action] = score + chance_value

        # If we ran out of budget, fall back to neural network evaluation
        if not action_scores:
            return self.act(board, is_eval=True)

        # Return the action with the highest computed score
        best_action = max(action_scores, key=action_scores.get)
        return best_action

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(embed_dim=32, num_heads=2, num_layers=1)
        # [[ 2, 2, 4, 0],
        #  [ 0, 0, 0, 0],
        #  [ 0, 0, 0, 0],
        #  [ 0, 0, 0, 0]]
        self.board = (1 << 0) | (1 << 4) | (2 << 8)

    def test_board_to_tokens(self):
        tokens = self.agent._board_to_tokens(self.board)
        self.assertEqual(len(tokens), 16)
        self.assertEqual(tokens[0], 1)
        self.assertEqual(tokens[1], 1)
        self.assertEqual(tokens[2], 2)
        self.assertEqual(tokens[3], 0)
        self.assertTrue(all(t == 0 for t in tokens[4:]))

    def test_create_input_tensor(self):
        tensor = self.agent._create_input_tensor(self.board)
        self.assertEqual(tensor.shape, (80,))
        self.assertEqual(tensor.dtype, torch.long)

        # Check first 16 tokens (current board)
        self.assertEqual(tensor[0].item(), 1)
        self.assertEqual(tensor[1].item(), 1)
        self.assertEqual(tensor[2].item(), 2)

        # Move up is illegal, so the next 16 tokens should be the same as the original board
        up_tokens = tensor[16:32]
        self.assertTrue(torch.equal(up_tokens, tensor[0:16]))

        # Check next 16 tokens (move left)
        # [[4, 4, 0, 0], ...] -> tokens [2, 2, 0, 0, ...]
        left_tokens = tensor[48:64]
        self.assertEqual(left_tokens[0].item(), 2)
        self.assertEqual(left_tokens[1].item(), 2)
        self.assertTrue(torch.all(left_tokens[2:16] == 0))
    
    def test_model_forward_pass(self):
        input_tensor = self.agent._create_input_tensor(self.board).unsqueeze(0)
        self.assertEqual(input_tensor.shape, (1, 80))
        
        with torch.no_grad():
            q_values = self.agent.policy_net(input_tensor)
        
        self.assertEqual(q_values.shape, (1, 4))

    def test_get_legal_actions(self):
        # For self.board, down, left, and right are legal moves. Up is not.
        legal = self.agent.get_legal_actions(self.board)
        self.assertNotIn(0, legal) # up
        self.assertIn(1, legal) # down
        self.assertIn(2, legal) # left
        self.assertIn(3, legal) # right
        self.assertEqual(len(legal), 3)
    
    def test_expectimax_search_logic(self):
        # Mock the heuristic function to return a simple, predictable value: number of empty cells
        def mock_heuristic(board: int):
            # Count number of zero nibbles
            empty_cells = sum(1 for i in range(16) if ((board >> (4 * i)) & 0xF) == 0)
            return float(empty_cells)

        self.agent._get_heuristic_value = mock_heuristic

        # Board: [[2, 2, 0, 0], [4, 0, 0, 0], ...] -> 13 empty cells
        board = (1 << 0) | (1 << 4) | (2 << 16)
        
        # We will calculate the expected value for each move with depth=2 (1 player, 1 chance)
        
        # --- Analysis for MOVE LEFT (action 2) ---
        # Deterministic result: [[4,0,0,0],[4,0,0,0],...] -> score=4, 14 empty cells
        # Now, the game adds a random tile. Any random placement (2 or 4) will result
        # in a board with 13 empty cells.
        # So, the heuristic value of any board in the next state is 13.
        # Expected value of chance node = sum(prob * heuristic_value) = sum(prob * 13) = 13 * sum(prob) = 13.
        # Total value for LEFT = immediate_score + expected_future_value = 4 + 13 = 17.0
        
        # --- Analysis for MOVE RIGHT (action 3) ---
        # Deterministic result: [[0,0,2,2],[0,0,0,4],...] -> score=0, 13 empty cells
        # The game adds a random tile. The resulting board will have 12 empty cells.
        # Heuristic value of any next state is 12.
        # Total value for RIGHT = 0 + 12 = 12.0

        # --- Analysis for MOVE DOWN (action 1) ---
        # Deterministic result: [[0,0,0,0],[2,0,0,0],[4,2,0,0],[0,0,0,0]] -> score=0, 13 empty cells
        # The game adds a random tile. The resulting board will have 12 empty cells.
        # Heuristic value of any next state is 12.
        # Total value for DOWN = 0 + 12 = 12.0
        
        # Based on this analysis, LEFT (value 17.0) is the best move.
        
        best_action = self.agent.act_with_search(board, max_depth=2)
        self.assertEqual(best_action, 2) # 2 is move left

if __name__ == "__main__":
    unittest.main()
