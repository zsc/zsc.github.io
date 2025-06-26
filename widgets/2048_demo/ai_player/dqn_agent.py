# 2048_demo/ai_player/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import collections
import os
import math

try:
    from game import Game # For Game.ACTION_LIST if needed, though this agent uses its own
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # from game import Game # Not strictly needed here anymore

# --- Default Hyperparameters for Agent Structure and Exploration ---
INPUT_SIZE = 16  # Flattened 4x4 board
ACTION_SIZE = 4  # up, down, left, right
DEFAULT_HIDDEN_SIZE = 256 
DEFAULT_LEARNING_RATE = 0.00025 
DEFAULT_GAMMA = 0.99 # Discount factor for Bellman equation
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.01
# EPSILON_DECAY_FRAMES defines how many *agent steps* (frames processed) it takes to decay
DEFAULT_EPSILON_DECAY_FRAMES = 100000 
DEFAULT_REPLAY_BUFFER_SIZE = 30000

class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=DEFAULT_HIDDEN_SIZE):
        super(DuelingQNetwork, self).__init__()
        self.hidden_size = hidden_size # Store for saving/loading context if needed
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state_tensor, action_idx, reward_val, next_state_tensor, done_flag):
        # States are already 4x4 tensors (torch.bfloat16) from game.boards[i]
        # Preprocess them here before storing.
        state_p = self._preprocess_single_state_tensor(state_tensor)
        next_state_p = self._preprocess_single_state_tensor(next_state_tensor)
        # Store as numpy arrays to save GPU memory if buffer is huge, convert back when sampling.
        # Or keep as tensors if mostly on GPU and buffer fits. Let's keep as CPU tensors.
        self.buffer.append((state_p.cpu(), action_idx, reward_val, next_state_p.cpu(), done_flag))

    def _preprocess_single_state_tensor(self, board_tensor_2d):
        # board_tensor_2d: (size, size) tensor, e.g., from game.boards[i]
        state_flat = board_tensor_2d.flatten().float() # Convert to float32 for log2 and network
        processed_state = torch.zeros_like(state_flat)
        non_zero_indices = state_flat > 0
        # Ensure log2 is safe. Game tiles are 2, 4, 8...
        processed_state[non_zero_indices] = torch.log2(state_flat[non_zero_indices]) / 16.0 
        return processed_state # Shape (16,)

    def sample(self, batch_size, device):
        samples = random.sample(self.buffer, batch_size)
        states_p, actions, rewards, next_states_p, dones = zip(*samples)

        # states_p and next_states_p are already processed (16,) tensors
        return (torch.stack(states_p).to(device),      
                torch.tensor(actions, dtype=torch.long).to(device),             
                torch.tensor(rewards, dtype=torch.float32).to(device),            
                torch.stack(next_states_p).to(device),
                torch.tensor(dones, dtype=torch.float32).to(device)) # dones as float for (1-dones)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    # Must match Game.ACTION_LIST order if Game uses string mapping internally
    DIRECTIONS_LIST = ['up', 'down', 'left', 'right'] 
    DIRECTION_TO_IDX = {direction: i for i, direction in enumerate(DIRECTIONS_LIST)}

    def __init__(self, model_path=None, training=True, device=None,
                 hidden_size=DEFAULT_HIDDEN_SIZE, 
                 learning_rate=DEFAULT_LEARNING_RATE,
                 gamma=DEFAULT_GAMMA, 
                 epsilon_start=DEFAULT_EPSILON_START,
                 epsilon_end=DEFAULT_EPSILON_END, 
                 epsilon_decay_frames=DEFAULT_EPSILON_DECAY_FRAMES,
                 replay_buffer_size=DEFAULT_REPLAY_BUFFER_SIZE):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Moved print to be unconditional for clarity on device used per instance
        print(f"DQN Agent instance initializing on device: {self.device}")


        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        
        self.policy_net = DuelingQNetwork(INPUT_SIZE, ACTION_SIZE, self.hidden_size).to(self.device)
        self.target_net = DuelingQNetwork(INPUT_SIZE, ACTION_SIZE, self.hidden_size).to(self.device)

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                # If model was saved with DataParallel, keys might have "module." prefix
                if isinstance(self.policy_net, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                elif not isinstance(self.policy_net, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

                self.policy_net.load_state_dict(state_dict)
                # Potential: check self.policy_net.hidden_size against loaded model if saved.
                # For now, assume hidden_size matches or DuelingQNetwork init handles it.
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Initializing new model with hidden_size={self.hidden_size}.")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.training = training
        self.epsilon = self.epsilon_start if self.training else 0.001 # Minimal exploration for eval

        self.total_agent_steps = 0 # Agent steps (frames processed or decisions made)
        self.total_learning_steps = 0 # Optimizer steps

    def _boards_to_tensor_processed(self, boards_batch_tensor_NxHxW):
        # boards_batch_tensor_NxHxW: (batch_size, H, W) tensor, e.g., from game.boards
        # Output: (batch_size, 16) tensor, float32, processed for network input
        batch_size = boards_batch_tensor_NxHxW.shape[0]
        flat_boards = boards_batch_tensor_NxHxW.reshape(batch_size, -1).float() # to float32
        
        processed_state = torch.zeros_like(flat_boards) # float32
        non_zero_mask = flat_boards > 0
        
        # Ensure log2 input is > 0. Game tiles are powers of 2, so >= 2.0.
        processed_state[non_zero_mask] = torch.log2(flat_boards[non_zero_mask]) / 16.0 
        return processed_state.to(self.device)

    def get_move(self, current_boards_tensor_NxHxW, legal_moves_mask_tensor_Nx4):
        # current_boards_tensor_NxHxW: (batch, H, W) from game.boards
        # legal_moves_mask_tensor_Nx4: (batch, 4) boolean mask from game.get_legal_moves()
        batch_size = current_boards_tensor_NxHxW.shape[0]

        if self.training:
            self.total_agent_steps += batch_size # Assuming one step per board in batch
            # Linear decay for epsilon
            self.epsilon = max(self.epsilon_end, 
                               self.epsilon_start - (self.total_agent_steps / self.epsilon_decay_frames) * (self.epsilon_start - self.epsilon_end))
        
        # Epsilon-greedy action selection (batched)
        random_samples = torch.rand(batch_size, device=self.device)
        explore_mask = random_samples < self.epsilon
        
        # For exploration: choose a random legal action
        # Create a tensor of random actions, then ensure they are legal.
        # This is tricky if some boards have very few legal moves.
        # Safer: for boards in explore_mask, pick from their specific legal_moves.
        
        with torch.no_grad():
            processed_states = self._boards_to_tensor_processed(current_boards_tensor_NxHxW)
            q_values = self.policy_net(processed_states) # (batch, 4)
        
        # Apply legal moves mask (set Q-values of illegal moves to -inf)
        # Ensure legal_moves_mask_tensor_Nx4 is on the same device as q_values
        q_values_masked = q_values.masked_fill(~legal_moves_mask_tensor_Nx4.to(self.device), -float('inf'))
        
        greedy_actions = torch.argmax(q_values_masked, dim=1) # (batch,)

        # For exploration, choose a random *legal* action.
        # If explore_mask.any() is true:
        chosen_actions = greedy_actions.clone()
        if self.training and explore_mask.any(): # Only do random selection if training and explore_mask is true for some
            # Get indices of boards that need random actions
            explore_indices = explore_mask.nonzero(as_tuple=True)[0]
            for idx_in_batch in explore_indices:
                legal_for_board = legal_moves_mask_tensor_Nx4[idx_in_batch]
                if legal_for_board.any(): # If there are any legal moves
                    legal_action_indices = legal_for_board.nonzero(as_tuple=True)[0]
                    # Ensure randint is on the correct device if legal_action_indices can be on a different device
                    # However, legal_action_indices should be on self.device due to legal_moves_mask_tensor_Nx4.to(self.device) earlier
                    # or if legal_moves_mask_tensor_Nx4 input is already on self.device
                    rand_idx = torch.randint(0, len(legal_action_indices), (1,), device=self.device).item()
                    chosen_actions[idx_in_batch] = legal_action_indices[rand_idx]
                # else: If no legal moves (game over for this one?), greedy_actions might pick -inf if all masked.
                # argmax of all -inf is 0. If action 0 is illegal, this is bad.
                # However, get_legal_moves should ensure at least one if not game_over.
                # If game_over, actions don't matter much. The game.move handles it.
                # Game class move will check if the board changed. If no legal moves, board won't change.
        
        # Ensure chosen actions are valid if, e.g., greedy_actions picked an action that became illegal
        # This check is more a safeguard. q_values_masked should handle it.
        # Fallback for boards where all Q-values were -inf (no legal moves):
        # argmax(-inf, -inf, ...) is 0. If action 0 is illegal, this would be an issue.
        # The game.get_legal_moves() should handle this: if game is not over, there's a legal move.
        # If game is over, get_legal_moves() might return all False.
        # The training loop should check game_over status.
        
        return chosen_actions.to(self.device) # Ensure output is on self.device

    def learn(self, batch_size_learn, target_update_freq_steps, learn_start_frames):
        # learn_start_frames is based on total_agent_steps (frames/decisions processed)
        if len(self.replay_buffer) < batch_size_learn or self.total_agent_steps < learn_start_frames:
            return None 

        if not self.training: return None # Should not be called if not training
        self.policy_net.train() # Ensure model is in training mode

        states_p, actions, rewards, next_states_p, dones = self.replay_buffer.sample(batch_size_learn, self.device)
        
        actions = actions.unsqueeze(1) # For gather: (batch_size, 1)
        rewards = rewards.unsqueeze(1) 
        dones = dones.unsqueeze(1)     

        current_q_values = self.policy_net(states_p).gather(1, actions)

        with torch.no_grad(): 
            # Double DQN: select actions using policy_net, evaluate using target_net
            policy_next_actions = self.policy_net(next_states_p).argmax(dim=1, keepdim=True)
            target_next_q_values = self.target_net(next_states_p).gather(1, policy_next_actions)
        
        expected_q_values = rewards + (self.gamma * target_next_q_values * (1 - dones))
        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values) 

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) 
        self.optimizer.step()
        
        self.total_learning_steps += 1
        if self.total_learning_steps % target_update_freq_steps == 0:
            self.update_target_network()
            
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model's hidden_size with state_dict for more robust loading if structure varies.
        # For this project, assume hidden_size during loading will match.
        model_state = self.policy_net.state_dict()
        # If using DataParallel, self.policy_net.module.state_dict() might be preferred.
        # Or, strip "module." prefix if it exists.
        if isinstance(self.policy_net, nn.DataParallel):
             model_state = self.policy_net.module.state_dict()

        torch.save(model_state, path)
        print(f"Model saved to {path}")

    def load_model(self, path): # Primarily used by app.py for playing
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            # Handle "module." prefix if model saved from DataParallel
            if isinstance(self.policy_net, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif not isinstance(self.policy_net, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
            
            if not self.training: self.policy_net.eval()
            else: self.policy_net.train() # Should be rare if loading for play
            self.target_net.eval()
            print(f"Model loaded from {path} for agent instance.")
        else:
            print(f"No model found at {path} for agent instance. Using initial weights.")


# Self-contained tests
if __name__ == "__main__":
    print("Testing DQN Agent Components...")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using test_device: {test_device}")
    
    # Test ReplayBuffer with tensor states
    buffer_test = ReplayBuffer(DEFAULT_REPLAY_BUFFER_SIZE)
    test_board_tensor = torch.tensor([[2,4,0,8],[16,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.bfloat16)
    processed_tensor = buffer_test._preprocess_single_state_tensor(test_board_tensor)
    assert processed_tensor.shape == (16,)
    assert processed_tensor.dtype == torch.float32
    assert abs(processed_tensor[0].item() - (math.log2(2)/16.0)) < 1e-6
    assert abs(processed_tensor[3].item() - (math.log2(8)/16.0)) < 1e-6
    print("ReplayBuffer._preprocess_single_state_tensor test passed.")

    # Mock game environment for agent testing (batched)
    mock_batch_size = 2
    mock_game_boards = torch.stack([
        torch.tensor([[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.bfloat16),
        torch.tensor([[0,4,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=torch.bfloat16)
    ]).to(test_device)
    
    # up, down, left, right
    mock_legal_moves = torch.tensor([
        [False, True, False, True], # Board 0: down, right
        [False, True, True, False]  # Board 1: down, left
    ], dtype=torch.bool).to(test_device)

    # Test agent in eval mode (batched)
    agent_eval_mode = DQNAgent(training=False, device=test_device) # Uses default HPs
    actions_eval = agent_eval_mode.get_move(mock_game_boards, mock_legal_moves)
    assert actions_eval.shape == (mock_batch_size,)
    assert actions_eval.device == test_device, f"actions_eval.device ({actions_eval.device}) != test_device ({test_device})"

    # Check if chosen actions are legal
    for i in range(mock_batch_size):
        assert mock_legal_moves[i, actions_eval[i].item()].item() == True
    print(f"Agent (eval mode, batched) suggests actions: {actions_eval.tolist()}")

    # Test agent in training mode (batched)
    agent_train_mode = DQNAgent(training=True, device=test_device, epsilon_start=0.5, epsilon_decay_frames=1000)
    actions_train = agent_train_mode.get_move(mock_game_boards, mock_legal_moves)
    assert actions_train.shape == (mock_batch_size,)
    assert actions_train.device == test_device, f"actions_train.device ({actions_train.device}) != test_device ({test_device})"
    for i in range(mock_batch_size):
        assert mock_legal_moves[i, actions_train[i].item()].item() == True
    print(f"Agent (train mode, batched, eps_start=0.5) suggests actions: {actions_train.tolist()}")
    print(f"Agent epsilon after one call with batch_size {mock_batch_size}: {agent_train_mode.epsilon:.4f}")


    # Test learn (using placeholder values)
    TEST_LEARN_BATCH_SIZE = 2 # Must be <= mock_batch_size for this push setup
    TEST_TARGET_UPDATE = 10
    TEST_LEARN_START_FRAMES = mock_batch_size # Start learning immediately for test

    agent_train_mode.total_agent_steps = TEST_LEARN_START_FRAMES # Pretend we've processed enough frames
    
    # Push some experiences
    for i in range(mock_batch_size): # Pushing 2 experiences
        s_tensor = mock_game_boards[i].clone() # These are on test_device
        ns_tensor = mock_game_boards[i].clone() # Dummy next state, also on test_device
        # Ensure ns_tensor[0,0] is different if s_tensor[0,0] was used for action
        if s_tensor[0,0] > 0 : ns_tensor[0,0] = 0
        else: ns_tensor[0,0] = 2

        a_idx = actions_train[i].item() # Use actions from get_move
        r_val = random.random() * 10
        d_flag = random.choice([True, False])
        # Pushing tensors that are on test_device (e.g. CUDA). ReplayBuffer.push will move them to CPU.
        agent_train_mode.replay_buffer.push(s_tensor, a_idx, r_val, ns_tensor, d_flag)
    
    print(f"Replay buffer size: {len(agent_train_mode.replay_buffer)}")
    if len(agent_train_mode.replay_buffer) >= TEST_LEARN_BATCH_SIZE:
        loss = agent_train_mode.learn(TEST_LEARN_BATCH_SIZE, TEST_TARGET_UPDATE, TEST_LEARN_START_FRAMES)
        assert loss is not None or len(agent_train_mode.replay_buffer) < TEST_LEARN_BATCH_SIZE
        print(f"Learn step executed, loss: {loss if loss else 'None (buffer too small or learn_start not met)'}")
    else:
        print(f"Skipping learn test as buffer size {len(agent_train_mode.replay_buffer)} < batch_size {TEST_LEARN_BATCH_SIZE}")


    # Test save/load
    temp_model_path = "temp_dqn_test_model.pth"
    agent_train_mode.save_model(temp_model_path)
    assert os.path.exists(temp_model_path)
    
    agent_loaded = DQNAgent(model_path=temp_model_path, training=False, device=test_device, hidden_size=DEFAULT_HIDDEN_SIZE)
    # Compare some weights (e.g., bias of first linear layer in feature_layer)
    original_bias = list(agent_train_mode.policy_net.feature_layer[0].parameters())[1].data
    loaded_bias = list(agent_loaded.policy_net.feature_layer[0].parameters())[1].data
    assert torch.allclose(original_bias, loaded_bias), "Model weights differ after load."
    print("Model save and load test passed.")
    if os.path.exists(temp_model_path): os.remove(temp_model_path)

    print("DQN Agent component tests completed.")
