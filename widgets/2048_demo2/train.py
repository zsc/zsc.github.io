# train.py

import os
import time
import datetime
import argparse
import collections
import unittest
import shutil  # Moved import to the top
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from game import Game2048
from agent import Agent

# Initialize the game's lookup tables once for the entire process.
Game2048._init_tables()

class VectorizedEnv:
    """
    A simple vectorized environment handler that runs multiple 2048 games in parallel.
    This is designed for synchronous execution.
    """
    def __init__(self, num_envs: int):
        """
        Initializes the vectorized environment.

        Args:
            num_envs: The number of parallel game environments.
        """
        self.num_envs = num_envs
        self._action_map = {
            0: Game2048.move_up,
            1: Game2048.move_down,
            2: Game2048.move_left,
            3: Game2048.move_right,
        }
        self.reset()

    def reset(self) -> List[int]:
        """
        Resets all environments to a new game state.

        Returns:
            A list of initial board states (64-bit integers).
        """
        self.boards = [Game2048.reset_board() for _ in range(self.num_envs)]
        self.scores = np.zeros(self.num_envs, dtype=np.int32)
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        return self.boards

    def step(self, actions: List[int]) -> Tuple[List[int], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Takes a step in each environment.

        Args:
            actions: A list of actions (0-3) for each environment.

        Returns:
            A tuple containing:
            - next_boards (List[int]): The new board states after the actions.
            - rewards (np.ndarray): The rewards obtained in each environment.
            - dones (np.ndarray): Boolean flags indicating if an environment's episode is over.
            - infos (List[Dict]): A list of info dictionaries, containing terminal
                                  observation data (e.g., final score) for finished episodes.
        """
        if len(actions) != self.num_envs:
            raise ValueError("Number of actions must match number of environments.")

        next_boards = [0] * self.num_envs
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        self.steps += 1

        for i in range(self.num_envs):
            move_func = self._action_map[actions[i]]
            moved_board, score, moved = move_func(self.boards[i])

            if moved:
                rewards[i] = float(score)
                self.scores[i] += score
                next_b = Game2048.add_random_tile(moved_board)
                is_done = Game2048.is_game_over(next_b)
            else:
                # Agent should ideally not choose illegal moves.
                # If it does, state does not change, reward is zero, and game is likely over.
                rewards[i] = 0.0 
                next_b = self.boards[i]
                is_done = Game2048.is_game_over(next_b)

            next_boards[i] = next_b
            dones[i] = is_done

            if is_done:
                infos[i]['episode'] = {
                    'r': self.scores[i],
                    'l': self.steps[i],
                    'max_tile': Game2048.get_max_tile(next_b)
                }
                # Reset this specific environment for the next iteration
                next_boards[i] = Game2048.reset_board()
                self.scores[i] = 0
                self.steps[i] = 0

        self.boards = next_boards
        return self.boards, rewards, dones, infos


def train(args):
    """
    Main training loop for the Double-DQN agent.
    """
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    # Create a unique experiment name
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f"2048_transformer_edim{args.embed_dim}_h{args.num_heads}_l{args.num_layers}_lr{args.lr}_{timestamp}"
    log_dir = os.path.join("runs", exp_name)
    checkpoint_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # --- 2. Initialize Agent and Environment ---
    agent = Agent(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epsilon_decay=args.epsilon_decay,
        device=device
    )

    envs = VectorizedEnv(num_envs=args.num_envs)
    current_boards = envs.reset()

    # Buffer for 2-step returns
    # deque stores (state, action, reward) for one step
    two_step_buffers = [collections.deque(maxlen=2) for _ in range(args.num_envs)]

    print("Starting training...")
    start_time = time.time()
    num_episodes = 0
    
    # --- 3. Training Loop ---
    for global_step in range(1, args.total_steps + 1):
        # Epsilon is managed inside agent.act()
        actions = [agent.act(board) for board in current_boards]

        # Step the parallel environments
        next_boards, rewards, dones, infos = envs.step(actions)
        
        # --- 4. Handle 2-step Bellman unroll and push to replay buffer ---
        for i in range(args.num_envs):
            # Store the latest transition info
            two_step_buffers[i].append((current_boards[i], actions[i], rewards[i]))
            
            # If the buffer has two transitions, we can calculate a 2-step return
            if len(two_step_buffers[i]) == 2:
                s_t0, a_t0, r_t0 = two_step_buffers[i][0]
                s_t1, a_t1, r_t1 = two_step_buffers[i][1]
                
                # The 2-step reward: R = r_t + gamma * r_{t+1}
                reward_2_step = r_t0 + args.gamma * r_t1
                
                # The experience is (s_t, a_t, R, s_{t+2}, done_{t+1})
                # s_t is s_t0, a_t is a_t0
                # s_{t+2} is the state after a_t1, which is `next_boards[i]`
                # done_{t+1} is the `dones[i]` flag for the current step
                agent.memory.push(s_t0, a_t0, reward_2_step, next_boards[i], dones[i])
            
            if dones[i]:
                # Episode is over. An info dict is available.
                num_episodes += 1
                ep_info = infos[i]['episode']
                
                print(f"Global Step: {global_step}, Episode: {num_episodes}, "
                      f"Score: {ep_info['r']}, Max Tile: {ep_info['max_tile']}, "
                      f"Steps: {ep_info['l']}, Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-1. * agent.steps_done / agent.epsilon_decay):.3f}")

                # Log episode stats to TensorBoard
                writer.add_scalar("episode/reward", ep_info['r'], global_step)
                writer.add_scalar("episode/max_tile", ep_info['max_tile'], global_step)
                writer.add_scalar("episode/length", ep_info['l'], global_step)
                writer.add_scalar("charts/episodes", num_episodes, global_step)
                
                # Flush the buffer as a 1-step return since the episode ended.
                # If len was 2, it was just processed. If it's 1, it's the last step.
                if len(two_step_buffers[i]) == 1:
                    s_last, a_last, r_last = two_step_buffers[i][0]
                    agent.memory.push(s_last, a_last, r_last, next_boards[i], True)

                two_step_buffers[i].clear()

        current_boards = next_boards

        # --- 5. Learning Step ---
        if global_step > args.learning_starts:
            loss = agent.learn()
            if loss is not None and global_step % 100 == 0:
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("charts/epsilon", agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-1. * agent.steps_done / agent.epsilon_decay), global_step)

        # --- 6. Checkpointing ---
        if global_step % args.save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}.pth")
            agent.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    print("Training finished.")


class TestTraining(unittest.TestCase):
    def setUp(self):
        """Set up a clean environment before each test."""
        self.test_dir = "test_temp"
        # Clean up directories from any previous runs to ensure a clean slate
        if os.path.exists("runs"):
            shutil.rmtree("runs")
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        os.makedirs(self.test_dir, exist_ok=True)
        # Dummy args for testing
        self.args = argparse.Namespace(
            num_envs=4,
            total_steps=50,
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            lr=1e-4,
            gamma=0.99,
            target_update_freq=20,
            buffer_size=100,
            batch_size=8,
            learning_starts=10,
            save_freq=40,
            epsilon_decay=100,
            force_cpu=True,
        )

    def tearDown(self):
        """Clean up artifacts after each test."""
        if os.path.exists("runs"):
            shutil.rmtree("runs")
        if os.path.exists("checkpoints"):
            shutil.rmtree("checkpoints")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_vectorized_env(self):
        num_envs = 8
        env = VectorizedEnv(num_envs)
        
        # Test reset
        initial_boards = env.reset()
        self.assertEqual(len(initial_boards), num_envs)
        self.assertTrue(all(isinstance(b, int) for b in initial_boards))

        # Test step
        actions = [np.random.randint(0, 4) for _ in range(num_envs)]
        next_boards, rewards, dones, infos = env.step(actions)
        
        self.assertEqual(len(next_boards), num_envs)
        self.assertEqual(rewards.shape, (num_envs,))
        self.assertEqual(dones.shape, (num_envs,))
        self.assertEqual(len(infos), num_envs)

    def test_training_smoke_test(self):
        """
        A "smoke test" to ensure the training loop runs without crashing
        for a few steps and creates expected artifacts.
        """
        # The original try/except Exception block is an anti-pattern in tests
        # because it catches AssertionErrors and obscures the true point of failure.
        # We let the test runner handle exceptions and assertion failures.
        try:
            train(self.args)
            # Check if log and checkpoint directories were created
            self.assertTrue(os.path.exists("runs"))
            self.assertTrue(os.path.exists("checkpoints"))
            
            # Find the created subdirectories
            run_subdirs = os.listdir("runs")
            ckpt_subdirs = os.listdir("checkpoints")
            self.assertEqual(len(run_subdirs), 1)
            self.assertEqual(len(ckpt_subdirs), 1)
            
            # Check if a checkpoint file was saved
            ckpt_path = os.path.join("checkpoints", ckpt_subdirs[0])
            self.assertTrue(any(f.startswith("step_") for f in os.listdir(ckpt_path)))

        except Exception as e:
            # This will now only catch unexpected errors from train(), not assertion failures.
            self.fail(f"Training loop failed with an unexpected exception: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer-based Double-DQN agent for 2048.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total number of steps to train for.")
    
    # Model Hyperparameters
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for the Transformer.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in the Transformer's multi-head attention.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the Transformer encoder.")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the AdamW optimizer.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for learning from the replay buffer.")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Size of the experience replay buffer.")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Number of steps to fill the buffer before learning starts.")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Frequency (in steps) to update the target network.")
    parser.add_argument("--epsilon_decay", type=float, default=200000, help="Decay rate for epsilon-greedy exploration.")

    # Logging and Saving
    parser.add_argument("--save_freq", type=int, default=50000, help="Frequency (in steps) to save a model checkpoint.")
    parser.add_argument("--force_cpu", action="store_true", help="Force training on CPU even if CUDA is available.")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    # To run tests: python -m unittest train.py
    # To run training: python train.py [options]
    main()
