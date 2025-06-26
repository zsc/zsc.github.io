# 2048_demo/train_dqn.py
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from collections import deque
import argparse 

from game import Game # Now the batched PyTorch version
# Import agent and its defaults for structure/exploration
from ai_player.dqn_agent import DQNAgent, DEFAULT_HIDDEN_SIZE, DEFAULT_LEARNING_RATE, \
                                DEFAULT_GAMMA, DEFAULT_EPSILON_START, DEFAULT_EPSILON_END, \
                                DEFAULT_EPSILON_DECAY_FRAMES, DEFAULT_REPLAY_BUFFER_SIZE
from ai_player.dummy import DummyAI # For baseline comparison

# --- Default Training Configuration (can be overridden by args) ---
DEFAULT_NUM_EPISODES_TOTAL = 50000 
DEFAULT_MAX_STEPS_PER_EPISODE = 3000 
DEFAULT_ENV_BATCH_SIZE = 128 # Number of game environments to run in parallel

# Training loop specific defaults (passed to agent.learn method)
DEFAULT_LEARN_BATCH_SIZE_TRAIN = 128 # Batch size for sampling from replay buffer
DEFAULT_TARGET_UPDATE_FREQ_STEPS_TRAIN = 1000 # In terms of agent.total_learning_steps
DEFAULT_LEARN_START_FRAMES_TRAIN = 10000 # In terms of agent.total_agent_steps

MODEL_DIR = "models"
MODEL_NAME_BASE = "dqn_2048"
LOG_DIR_BASE = "runs/dqn_2048_trainer"
SAVE_MODEL_EVERY_N_EPISODES = 500
EVALUATE_EVERY_N_EPISODES = 200 # How often to run evaluation
EVALUATION_GAMES = 64 # Number of games for evaluation (should be multiple of env_batch_size for efficiency)


def calculate_reward_batch(current_scores_batch, prev_scores_batch, 
                           prev_boards_batch, current_boards_batch, 
                           moved_mask_batch, game_over_mask_batch, device):
    # All inputs are PyTorch tensors on the specified device.
    # current_scores_batch, prev_scores_batch: (env_batch_size,)
    # prev_boards_batch, current_boards_batch: (env_batch_size, size, size)
    # moved_mask_batch, game_over_mask_batch: (env_batch_size,) boolean
    
    score_increase = (current_scores_batch - prev_scores_batch).float()
    rewards = score_increase.clone()

    # Penalty for game over, scaled by how far from 2048
    # Max tile on board for scaling penalty
    max_tiles_current = torch.max(current_boards_batch.view(current_boards_batch.shape[0], -1), dim=1)[0].float()
    
    # Penalty: higher if ended with lower max tile or score.
    # Base penalty for game over, then add more if score is low.
    # Avoid division by zero if score is zero.
    score_penalty_factor = torch.ones_like(max_tiles_current)
    low_score_mask = current_scores_batch < 2048
    # A simple factor: (2048 - score) / 2048. Max 1 if score is 0.
    score_penalty_factor[low_score_mask] = (2048.0 - current_scores_batch[low_score_mask].float()) / 2048.0
    score_penalty_factor[current_scores_batch >= 2048] = 0.1 # Small penalty even if win

    rewards[game_over_mask_batch] -= (50.0 * score_penalty_factor[game_over_mask_batch])

    # Penalty for invalid moves (not moved and not game over)
    rewards[~moved_mask_batch & ~game_over_mask_batch] -= 2.0

    # Reward for creating empty cells (if score also increased)
    prev_empty_counts = torch.sum(prev_boards_batch.view(prev_boards_batch.shape[0], -1) == 0, dim=1).float()
    current_empty_counts = torch.sum(current_boards_batch.view(current_boards_batch.shape[0], -1) == 0, dim=1).float()
    
    empty_cell_increase_mask = (current_empty_counts > prev_empty_counts) & (score_increase > 0)
    rewards[empty_cell_increase_mask] += (current_empty_counts[empty_cell_increase_mask] - prev_empty_counts[empty_cell_increase_mask]) * 0.5
    
    return rewards # (env_batch_size,)


def train(args): 
    num_episodes = args.num_episodes
    max_steps_per_episode = args.max_steps
    env_batch_size = args.env_batch_size # Number of parallel game environments
    
    # Hyperparameters for DQNAgent instantiation
    hidden_size = args.hidden_size
    learning_rate = args.lr
    gamma = args.gamma
    epsilon_start = args.eps_start
    epsilon_end = args.eps_end
    epsilon_decay_frames = args.eps_decay # Based on agent steps
    replay_buffer_size = args.buffer_size

    # Hyperparameters for the training loop (passed to agent.learn)
    learn_batch_size = args.learn_batch_size 
    target_update_freq_steps = args.target_update # Based on learning steps
    learn_start_frames = args.learn_start # Based on agent steps

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_specific_tag = f"hs{hidden_size}_lr{learning_rate}_gamma{gamma}_ebs{env_batch_size}_lbs{learn_batch_size}_{timestamp}"
    RUN_LOG_DIR = os.path.join(LOG_DIR_BASE, run_specific_tag)
    RUN_MODEL_DIR = os.path.join(MODEL_DIR, f"{MODEL_NAME_BASE}_{run_specific_tag}")
    BEST_MODEL_PATH = os.path.join(RUN_MODEL_DIR, f"{MODEL_NAME_BASE}_best.pth")
    FINAL_MODEL_PATH = os.path.join(RUN_MODEL_DIR, f"{MODEL_NAME_BASE}_final.pth")

    os.makedirs(RUN_LOG_DIR, exist_ok=True); os.makedirs(RUN_MODEL_DIR, exist_ok=True)
    writer = SummaryWriter(RUN_LOG_DIR)
    hparams_dict_to_log = vars(args) # Log all parsed args
    # Convert device object to string for logging if it's in args (it's not here, but good practice)
    # if 'device' in hparams_dict_to_log and not isinstance(hparams_dict_to_log['device'], str):
    #    hparams_dict_to_log['device'] = str(hparams_dict_to_log['device'])
    writer.add_text("Hyperparameters", str(hparams_dict_to_log).replace(", ", ",\n"), 0)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(training=True, device=device,
                     hidden_size=hidden_size, learning_rate=learning_rate, gamma=gamma,
                     epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                     epsilon_decay_frames=epsilon_decay_frames, replay_buffer_size=replay_buffer_size)
    
    game = Game(size=4, batch_size=env_batch_size, device=device)

    print(f"Starting training on {device}. Logs: {RUN_LOG_DIR}, Models: {RUN_MODEL_DIR}")
    # Corrected replay buffer size reporting for DQNAgent
    # The ReplayBuffer class stores capacity in self.buffer.maxlen
    print(f"Agent HPs: hidden_size={agent.hidden_size}, lr={agent.learning_rate}, gamma={agent.gamma}, eps_decay={agent.epsilon_decay_frames}, buffer_size={agent.replay_buffer.buffer.maxlen}")
    print(f"Training Loop HPs: env_batch_size={env_batch_size}, learn_batch_size={learn_batch_size}, target_update_freq={target_update_freq_steps}, learn_start_frames={learn_start_frames}")

    recent_episode_scores_deque = deque(maxlen=100 * env_batch_size) # Store scores from individual games in batch
    best_eval_avg_score = -float('inf')
    total_steps_across_episodes = 0

    for episode in range(1, num_episodes + 1):
        game.reset() # Resets all boards in the batch
        
        # Track episode aggregates (average over batch, sum over steps)
        episode_total_reward_sum_over_batch = 0.0 
        episode_total_loss_sum = 0.0
        episode_loss_count = 0
        
        start_time_episode = time.time()

        for step_in_episode in range(max_steps_per_episode):
            current_boards_tensor = game.boards.clone() # (env_batch, size, size)
            prev_scores_tensor = game.scores.clone()    # (env_batch,)
            
            legal_moves_batch_mask = game.get_legal_moves() # (env_batch, 4)
            actions_tensor = agent.get_move(current_boards_tensor, legal_moves_batch_mask) # (env_batch,)
            
            moved_mask_tensor = game.move(actions_tensor) # (env_batch,) boolean, indicates if board changed
            
            next_boards_tensor = game.boards.clone()
            current_scores_tensor = game.scores.clone() # Scores after move
            game_over_mask_tensor = game.is_game_over() # (env_batch,) boolean
            
            # The order of current_boards_tensor (prev_state) and next_boards_tensor (current_state) is correct for this call.
            rewards_tensor = calculate_reward_batch(current_scores_tensor, prev_scores_tensor,
                                                 current_boards_tensor, next_boards_tensor, 
                                                 moved_mask_tensor, game_over_mask_tensor, device)
            
            episode_total_reward_sum_over_batch += rewards_tensor.sum().item() # Sum rewards from all envs in this step

            # Push experiences to replay buffer (one for each environment in the batch)
            for i in range(env_batch_size):
                agent.replay_buffer.push(
                    current_boards_tensor[i], # Single board tensor (size,size)
                    actions_tensor[i].item(),
                    rewards_tensor[i].item(),
                    next_boards_tensor[i],    # Single board tensor (size,size)
                    game_over_mask_tensor[i].item()
                )
            
            # Learning step for the agent
            if agent.total_agent_steps > learn_start_frames and \
               len(agent.replay_buffer) >= learn_batch_size:
                loss = agent.learn(learn_batch_size, target_update_freq_steps, learn_start_frames)
                if loss is not None:
                    episode_total_loss_sum += loss
                    episode_loss_count += 1
            
            total_steps_across_episodes += env_batch_size # Total frames processed across all episodes

            if torch.any(game_over_mask_tensor) or (step_in_episode == max_steps_per_episode - 1) :
                 # If any game in the batch is over, or max steps reached, end this "episode" for logging.
                 # The game.reset() at start of loop will handle resetting all.
                 # For individual game metrics, we'd need to track them separately.
                 # Here, we log aggregates.
                final_scores_this_episode_batch = game.scores.cpu().tolist()
                for score_val in final_scores_this_episode_batch:
                    recent_episode_scores_deque.append(score_val)
                break 
        
        # End of episode calculations
        episode_duration = time.time() - start_time_episode
        steps_this_episode_total_envs = (step_in_episode + 1) * env_batch_size
        
        avg_final_score_batch = np.mean(game.scores.cpu().numpy()) if env_batch_size > 0 else 0.0
        max_tile_batch_avg = np.mean(torch.max(game.boards.view(env_batch_size, -1), dim=1)[0].cpu().float().numpy()) if env_batch_size > 0 else 0.0
        
        avg_recent_score_overall = np.mean(recent_episode_scores_deque) if recent_episode_scores_deque else 0.0
        avg_episode_reward = episode_total_reward_sum_over_batch / env_batch_size if env_batch_size > 0 else 0.0
        avg_loss_episode = episode_total_loss_sum / episode_loss_count if episode_loss_count > 0 else 0.0

        writer.add_scalar('Reward/AvgEpisodeSummed', avg_episode_reward, episode)
        writer.add_scalar('Score/AvgFinalBatch', avg_final_score_batch, episode)
        writer.add_scalar('Score/AvgRecentOverall_100epsBatch', avg_recent_score_overall, episode)
        writer.add_scalar('MaxTile/AvgFinalBatch', max_tile_batch_avg, episode)
        writer.add_scalar('Steps/EpisodeTotalEnvs', steps_this_episode_total_envs, episode)
        writer.add_scalar('Epsilon/Value', agent.epsilon, episode) 
        writer.add_scalar('ReplayBuffer/Size', len(agent.replay_buffer), agent.total_agent_steps) # X-axis as agent steps
        writer.add_scalar('Performance/SPS_train_total_env_steps', steps_this_episode_total_envs / episode_duration if episode_duration > 0 else 0, episode)
        writer.add_scalar('Loss/AvgEpisodeLoss', avg_loss_episode, episode)

        print(f"Ep {episode}/{num_episodes} | Steps: {step_in_episode+1} (x{env_batch_size}) | AvgScoreBatch: {avg_final_score_batch:.1f} | AvgMaxTileBatch: {max_tile_batch_avg:.1f} | AvgRecentOverall: {avg_recent_score_overall:.1f} | Eps: {agent.epsilon:.3f} | Loss: {avg_loss_episode:.4f} | AvgEpReward: {avg_episode_reward:.1f} | SPS: {(steps_this_episode_total_envs)/episode_duration if episode_duration > 0 else 0:.1f}")

        if episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_model(os.path.join(RUN_MODEL_DIR, f"{MODEL_NAME_BASE}_ep{episode}.pth"))

        if episode % EVALUATE_EVERY_N_EPISODES == 0:
            print(f"\n--- Evaluating model at episode {episode} ---")
            # For evaluation, run EVALUATION_GAMES. If EVALUATION_GAMES > env_batch_size, it will run in chunks.
            eval_avg_score, eval_avg_max_tile, eval_win_rate = evaluate_model(agent, Game, EVALUATION_GAMES, device, env_batch_size_eval=args.env_batch_size) # Use train env_batch_size for eval too
            writer.add_scalar('Evaluation/AvgScore_DQN', eval_avg_score, episode)
            writer.add_scalar('Evaluation/AvgMaxTile_DQN', eval_avg_max_tile, episode)
            writer.add_scalar('Evaluation/WinRate2048_DQN', eval_win_rate, episode)
            
            # Log hparams with evaluation metrics
            # Strip device from hparams_dict_to_log if it's an object for add_hparams
            hparams_loggable = {k: (str(v) if isinstance(v, torch.device) else v) for k,v in hparams_dict_to_log.items()}
            writer.add_hparams(hparams_loggable,
                {'hparam/eval_avg_score': eval_avg_score, 
                 'hparam/eval_avg_max_tile': eval_avg_max_tile, 
                 'hparam/eval_win_rate': eval_win_rate},
                run_name='.', # Standard practice for single run hparam logging
                global_step=episode) # Associate with this episode

            if eval_avg_score > best_eval_avg_score:
                best_eval_avg_score = eval_avg_score
                agent.save_model(BEST_MODEL_PATH)
                print(f"*** New best evaluation model saved: Avg Score {best_eval_avg_score:.2f} ***\n")
            
    agent.save_model(FINAL_MODEL_PATH)
    writer.close()
    print(f"Training finished. Final model: {FINAL_MODEL_PATH}, Best eval model: {BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else 'Not Saved'}")


def evaluate_model(eval_agent, game_class, num_total_eval_games, device, env_batch_size_eval, is_dummy=False):
    if is_dummy: # Dummy AI is not batched, run sequentially
        # Ensure DummyAIAdapter is used if DummyAI needs adaptation
        eval_agent_instance = eval_agent() # This should be DummyAIAdapter() if using the adapter
        total_scores, total_max_tiles, wins_2048 = 0, 0, 0
        games_won_this_eval = 0 # Specific for 2048 counting
        for game_idx in range(num_total_eval_games):
            game_eval_single = game_class(size=4, batch_size=1, device=device) # Dummy uses single game
            game_won_flag = False
            while not game_eval_single.is_game_over()[0].item(): 
                # Adapting DummyAI.get_move:
                # board_list_of_list = game_eval_single.boards[0].tolist() # If DummyAI needs list of lists
                # For DummyAIAdapter, it expects the raw board tensor and legal moves tensor
                action_str = eval_agent_instance.get_move_adapter(game_eval_single.boards[0], game_eval_single.get_legal_moves()[0])
                if action_str is None: break # Should not happen if game not over
                action_idx = DQNAgent.DIRECTION_TO_IDX[action_str]
                game_eval_single.move(torch.tensor([action_idx], device=device))
                if not game_won_flag and torch.max(game_eval_single.boards[0]).item() >= 2048: 
                    games_won_this_eval +=1
                    game_won_flag = True # Count win once per game
                    # Some evaluation setups might stop game upon reaching 2048, others continue. Assuming continue for max score.
            total_scores += game_eval_single.scores[0].item()
            total_max_tiles += torch.max(game_eval_single.boards[0]).item()
        avg_score = total_scores/num_total_eval_games if num_total_eval_games > 0 else 0
        avg_max_tile = total_max_tiles/num_total_eval_games if num_total_eval_games > 0 else 0
        win_rate = games_won_this_eval/num_total_eval_games if num_total_eval_games > 0 else 0
        print(f"  DummyAI Eval ({num_total_eval_games} games): Avg Score: {avg_score:.1f}, Avg Max Tile: {avg_max_tile:.1f}, Win (2048): {win_rate*100:.1f}%")
        return avg_score, avg_max_tile, win_rate

    # For DQNAgent (batched evaluation)
    original_training_state = eval_agent.training
    original_epsilon = eval_agent.epsilon
    eval_agent.training = False 
    eval_agent.epsilon = 0.001 # Minimal exploration for evaluation
    if hasattr(eval_agent, 'policy_net'): eval_agent.policy_net.eval()

    all_game_scores = []
    all_game_max_tiles = []
    all_game_wins = [] # Tracks if 2048 tile was achieved in each game

    game_eval_batch = game_class(size=4, batch_size=env_batch_size_eval, device=device)
    
    num_eval_loops = (num_total_eval_games + env_batch_size_eval - 1) // env_batch_size_eval # Ceiling division
    games_processed_count = 0

    for loop_idx in range(num_eval_loops):
        if games_processed_count >= num_total_eval_games: break
        
        current_batch_actual_size = min(env_batch_size_eval, num_total_eval_games - games_processed_count)
        if current_batch_actual_size != game_eval_batch.batch_size: 
            game_eval_batch = game_class(size=4, batch_size=current_batch_actual_size, device=device)
        else:
            game_eval_batch.reset()

        active_games_mask = torch.ones(current_batch_actual_size, dtype=torch.bool, device=device)
        batch_final_scores = torch.zeros(current_batch_actual_size, dtype=torch.int32, device=device)
        batch_final_max_tiles = torch.zeros(current_batch_actual_size, dtype=torch.bfloat16, device=device) # game.boards is bfloat16
        batch_has_won_2048 = torch.zeros(current_batch_actual_size, dtype=torch.bool, device=device) # Tracks if 2048 achieved

        for _step in range(DEFAULT_MAX_STEPS_PER_EPISODE): 
            if not active_games_mask.any(): break 

            current_boards_active = game_eval_batch.boards[active_games_mask]
            legal_moves_active = game_eval_batch.get_legal_moves()[active_games_mask]
            
            if current_boards_active.shape[0] == 0: # Should be caught by active_games_mask.any()
                break

            actions_active = eval_agent.get_move(current_boards_active, legal_moves_active)
            
            full_actions = torch.zeros(current_batch_actual_size, dtype=torch.long, device=device)
            # Ensure actions_active is not empty before trying to place it in full_actions
            if actions_active.numel() > 0:
                 full_actions[active_games_mask] = actions_active
            
            game_eval_batch.move(full_actions) 

            # Check for 2048 win condition for active games that haven't won yet
            # active_indices_global refers to indices in the current batch (0 to current_batch_actual_size-1)
            active_indices_global = active_games_mask.nonzero(as_tuple=True)[0]
            if active_indices_global.numel() > 0: # If there are still active games
                not_yet_won_mask_global = ~batch_has_won_2048[active_indices_global]
                # Boards of active games that haven't won yet
                boards_to_check_win = game_eval_batch.boards[active_indices_global[not_yet_won_mask_global]]
                if boards_to_check_win.numel() > 0: # If any such boards exist
                    # achieved_2048_now is a boolean mask relative to boards_to_check_win
                    achieved_2048_now = torch.max(boards_to_check_win.view(boards_to_check_win.shape[0], -1), dim=1)[0] >= 2048
                    # Update batch_has_won_2048 for these specific games
                    batch_has_won_2048[active_indices_global[not_yet_won_mask_global][achieved_2048_now]] = True

            # Game over check for active games
            game_over_now_local_mask = game_eval_batch.is_game_over()[active_games_mask] # Relative to active_games_mask
            
            # Games that just finished in this step (game over)
            # active_indices_global are indices into the current batch
            # global_indices_just_finished are indices into current batch for games that just ended
            global_indices_just_finished = active_indices_global[game_over_now_local_mask]

            if global_indices_just_finished.any():
                batch_final_scores[global_indices_just_finished] = game_eval_batch.scores[global_indices_just_finished]
                batch_final_max_tiles[global_indices_just_finished] = torch.max(game_eval_batch.boards[global_indices_just_finished].view(global_indices_just_finished.shape[0],-1),dim=1)[0]
                # Win status already updated, just ensure these are marked inactive
                active_games_mask[global_indices_just_finished] = False 
        
        # After max steps, if any games are still active, record their current state
        if active_games_mask.any():
            batch_final_scores[active_games_mask] = game_eval_batch.scores[active_games_mask]
            batch_final_max_tiles[active_games_mask] = torch.max(game_eval_batch.boards[active_games_mask].view(active_games_mask.sum(),-1),dim=1)[0]
            # Win status already updated through the loop

        all_game_scores.extend(batch_final_scores.cpu().tolist())
        all_game_max_tiles.extend(batch_final_max_tiles.cpu().float().tolist()) # Convert bfloat16 to float for numpy
        all_game_wins.extend(batch_has_won_2048.cpu().tolist()) # Use the tracked win status
        games_processed_count += current_batch_actual_size

    avg_score = np.mean(all_game_scores[:num_total_eval_games]) if len(all_game_scores) > 0 else 0
    avg_max_tile = np.mean(all_game_max_tiles[:num_total_eval_games]) if len(all_game_max_tiles) > 0 else 0
    win_rate = np.mean(all_game_wins[:num_total_eval_games]) if len(all_game_wins) > 0 else 0
    
    print(f"  DQNAgent Eval ({num_total_eval_games} games): Avg Score: {avg_score:.1f}, Avg Max Tile: {avg_max_tile:.1f}, Win (2048): {win_rate*100:.1f}%")

    if not is_dummy and hasattr(eval_agent, 'policy_net'): 
        eval_agent.training = original_training_state
        eval_agent.epsilon = original_epsilon
        if eval_agent.training: eval_agent.policy_net.train()

    return avg_score, avg_max_tile, win_rate


# Adapter for DummyAI if its get_move is not updated
class DummyAIAdapter(DummyAI):
    def get_move_adapter(self, board_tensor_single, legal_moves_bool_tensor_for_board0):
        # board_tensor_single: (size,size) tensor for one board
        # legal_moves_bool_tensor_for_board0: (4,) boolean tensor for one board
        class MockOldGame:
            def __init__(self, board_list, legal_moves_list_str):
                self.board = board_list # Expected: list of lists
                self._legal_moves_list_str = legal_moves_list_str
            def get_board(self): return self.board
            def get_legal_moves(self): return self._legal_moves_list_str # Expected: ['up', 'down', ...]
        
        # Convert board tensor to list of lists for DummyAI's old expected format
        board_list_of_list = board_tensor_single.tolist() 
        
        # Convert boolean mask to list of action strings
        # Assuming Game.ACTION_LIST exists and matches DQNAgent.DIRECTIONS_LIST order
        action_list_str_order = DQNAgent.DIRECTIONS_LIST # Use agent's definition for consistency
        legal_strs = [action_list_str_order[i] for i, legal in enumerate(legal_moves_bool_tensor_for_board0) if legal]
        
        mock_game = MockOldGame(board_list_of_list, legal_strs)
        return super().get_move(mock_game)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for 2048 (Batched)")
    # Training setup
    parser.add_argument("--num_episodes", type=int, default=DEFAULT_NUM_EPISODES_TOTAL, help="Total training episodes")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE, help="Max steps per episode")
    parser.add_argument("--env_batch_size", type=int, default=DEFAULT_ENV_BATCH_SIZE, help="Number of game environments to run in parallel")

    # Agent structure/exploration HPs (for DQNAgent constructor)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate for Adam optimizer")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE, help="Network hidden layer size")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor gamma")
    parser.add_argument("--eps_start", type=float, default=DEFAULT_EPSILON_START, help="Epsilon start value")
    parser.add_argument("--eps_end", type=float, default=DEFAULT_EPSILON_END, help="Epsilon end value")
    parser.add_argument("--eps_decay", type=int, default=DEFAULT_EPSILON_DECAY_FRAMES, help="Agent steps over which epsilon decays")
    parser.add_argument("--buffer_size", type=int, default=DEFAULT_REPLAY_BUFFER_SIZE, help="Replay buffer capacity")

    # Training loop HPs (for agent.learn method and loop control)
    parser.add_argument("--learn_batch_size", type=int, default=DEFAULT_LEARN_BATCH_SIZE_TRAIN, help="Batch size for learning from replay buffer")
    parser.add_argument("--target_update", type=int, default=DEFAULT_TARGET_UPDATE_FREQ_STEPS_TRAIN, help="Frequency (in learning steps) to update target network")
    parser.add_argument("--learn_start", type=int, default=DEFAULT_LEARN_START_FRAMES_TRAIN, help="Agent steps before learning starts")
    
    cli_args = parser.parse_args()
    
    # Ensure evaluation games is reasonable with env_batch_size
    if EVALUATION_GAMES < cli_args.env_batch_size :
        print(f"Warning: EVALUATION_GAMES ({EVALUATION_GAMES}) is less than env_batch_size ({cli_args.env_batch_size}). Adjusting EVALUATION_GAMES to {cli_args.env_batch_size}.")
        EVALUATION_GAMES = cli_args.env_batch_size
    
    train(cli_args)

    print("\n--- Final Evaluation of Dummy AI for Comparison ---")
    # DummyAI evaluation is sequential (not batched internally)
    evaluate_model(DummyAIAdapter, game_class=Game, num_total_eval_games=64, # Use EVALUATION_GAMES
                   device=torch.device("cpu"), # DummyAI likely not using GPU, safer on CPU
                   env_batch_size_eval=1, is_dummy=True)
