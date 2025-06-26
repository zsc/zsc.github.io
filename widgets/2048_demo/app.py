from flask import Flask, jsonify, request, send_from_directory
from game import Game # Now the PyTorch version
from ai_player.dummy import DummyAI
# Import DQNAgent and its defaults
from ai_player.dqn_agent import DQNAgent, DEFAULT_HIDDEN_SIZE, DEFAULT_LEARNING_RATE, \
                                DEFAULT_GAMMA, DEFAULT_EPSILON_START, DEFAULT_EPSILON_END, \
                                DEFAULT_EPSILON_DECAY_FRAMES, DEFAULT_REPLAY_BUFFER_SIZE
# Import training loop defaults from train_dqn.py
from train_dqn import DEFAULT_ENV_BATCH_SIZE, DEFAULT_LEARN_BATCH_SIZE_TRAIN, \
                      DEFAULT_TARGET_UPDATE_FREQ_STEPS_TRAIN, \
                      DEFAULT_LEARN_START_FRAMES_TRAIN, DEFAULT_NUM_EPISODES_TOTAL

import os
import subprocess 
import sys 
import torch 

app = Flask(__name__, static_folder='static')

# For the Flask app, we run a single game instance on CPU
game_instance_device = "cpu" 
game_instance = Game(size=4, batch_size=1, device=game_instance_device) 
current_player_mode = "human"  
dummy_ai_player = DummyAI()
dqn_ai_player = None 
current_dqn_model_path_relative = None # Stores relative path of loaded DQN model

DQN_MODEL_DIR = "models"
MODEL_NAME_BASE = "dqn_2048" 
DEFAULT_DQN_PLAY_MODEL_NAME = "dqn_2048_best.pth" 
active_ai_agent_type = "dummy" 
current_ai_instance = dummy_ai_player


def find_latest_best_model(base_model_dir=DQN_MODEL_DIR):
    if not os.path.exists(base_model_dir): return None
    candidate_models = [] 
    for item_name in os.listdir(base_model_dir): 
        item_path = os.path.join(base_model_dir, item_name)
        if os.path.isdir(item_path) and item_name.startswith(MODEL_NAME_BASE):
            parts = item_name.split('_')
            if len(parts) > 1:
                dir_timestamp_candidate = parts[-1] 
                if '-' in dir_timestamp_candidate and len(dir_timestamp_candidate.split('-')[0]) == 8:
                    timestamp_str = dir_timestamp_candidate 
                    potential_best_model_path = os.path.join(item_path, DEFAULT_DQN_PLAY_MODEL_NAME)
                    if os.path.exists(potential_best_model_path):
                        candidate_models.append((timestamp_str, potential_best_model_path))
    
    if candidate_models:
        candidate_models.sort(key=lambda x: x[0], reverse=True) 
        latest_model_path = candidate_models[0][1]
        print(f"[FIND_MODEL] Selected latest timestamped model: {latest_model_path}")
        return latest_model_path

    legacy_path = os.path.join(base_model_dir, DEFAULT_DQN_PLAY_MODEL_NAME)
    if os.path.exists(legacy_path):
        print(f"[FIND_MODEL] Selected legacy model: {legacy_path}")
        return legacy_path
    
    print(f"[FIND_MODEL] No suitable model found in {base_model_dir}")
    return None

def list_all_dqn_models(base_model_dir=DQN_MODEL_DIR):
    found_models = []
    if not os.path.exists(base_model_dir):
        return found_models

    # Check for models directly in base_model_dir (legacy)
    for item_name in os.listdir(base_model_dir):
        if item_name.endswith(".pth") and os.path.isfile(os.path.join(base_model_dir, item_name)):
            # Store path relative to base_model_dir
            found_models.append({'name': item_name, 'path': item_name})

    # Check for models in timestamped subdirectories
    for item_name in os.listdir(base_model_dir):
        item_path = os.path.join(base_model_dir, item_name)
        # Check if it's a directory and seems like one of our model run folders
        if os.path.isdir(item_path) and item_name.startswith(MODEL_NAME_BASE) and "_" in item_name:
            # Look for .pth files inside this subdir
            for sub_item_name in os.listdir(item_path):
                if sub_item_name.endswith(".pth") and os.path.isfile(os.path.join(item_path, sub_item_name)):
                    relative_path = os.path.join(item_name, sub_item_name)
                    # Use a more descriptive name if it's the standard best model name
                    display_sub_name = sub_item_name
                    if sub_item_name == DEFAULT_DQN_PLAY_MODEL_NAME:
                         display_sub_name = f"Best ({sub_item_name})"
                    
                    display_name = f"{display_sub_name} in /{item_name}"
                    found_models.append({'name': display_name, 'path': relative_path})
    
    found_models.sort(key=lambda x: x['path'])
    return found_models


def load_dqn_for_play(specific_model_path_full=None):
    global dqn_ai_player, current_ai_instance, active_ai_agent_type, current_dqn_model_path_relative
    
    actual_path_used_for_loading = None
    source_description = ""

    if specific_model_path_full:
        if os.path.exists(specific_model_path_full):
            actual_path_used_for_loading = specific_model_path_full
            source_description = f"specified path ({os.path.basename(specific_model_path_full)})"
        else:
            print(f"[LOAD_DQN] Specified model path '{specific_model_path_full}' does not exist.")
            # If a specific path is given and it's bad, it's an error for this load attempt.
            # Do not fall back to latest. Clear current DQN player if any.
            dqn_ai_player = None 
            current_dqn_model_path_relative = None
            # If current AI mode was DQN, fall back to dummy instance for safety
            if active_ai_agent_type == "dqn": current_ai_instance = dummy_ai_player
            return False 
    else: # specific_model_path_full is None, so find the latest model
        actual_path_used_for_loading = find_latest_best_model()
        if actual_path_used_for_loading:
            source_description = f"latest model ({os.path.basename(actual_path_used_for_loading)})"
        else:
            source_description = "latest model (none found)"

    if actual_path_used_for_loading: # This implies os.path.exists(actual_path_used_for_loading) is true
        try:
            play_device = torch.device("cpu")
            loaded_agent = DQNAgent(model_path=actual_path_used_for_loading, training=False, device=play_device)
            dqn_ai_player = loaded_agent 
            
            if not (hasattr(dqn_ai_player, 'policy_net') and dqn_ai_player.policy_net is not None):
                 print(f"[LOAD_DQN] policy_net not found after DQNAgent instantiation from {source_description}.")
                 dqn_ai_player = None; current_dqn_model_path_relative = None
                 if active_ai_agent_type == "dqn": current_ai_instance = dummy_ai_player
                 return False
            
            print(f"[LOAD_DQN] DQNAgent successfully instantiated for play from {source_description}.")
            current_dqn_model_path_relative = os.path.relpath(actual_path_used_for_loading, DQN_MODEL_DIR)
            
            if active_ai_agent_type == "dqn": current_ai_instance = dqn_ai_player # Ensure AI instance is updated if mode is DQN
            return True 
        except Exception as e:
            print(f"[LOAD_DQN] ERROR loading '{actual_path_used_for_loading}' (from {source_description}): {e}")
            dqn_ai_player = None 
            current_dqn_model_path_relative = None
            if active_ai_agent_type == "dqn": current_ai_instance = dummy_ai_player 
            return False 
    else: # No model to load
        print(f"[LOAD_DQN] No model path to load. Attempted source: {source_description}.")
        dqn_ai_player = None 
        current_dqn_model_path_relative = None
        if active_ai_agent_type == "dqn": current_ai_instance = dummy_ai_player
        return False 

load_dqn_for_play() # Attempt to load latest model at startup
training_process_handle = None
LOG_DIR_BASE = "runs/dqn_2048_trainer"

@app.route('/')
def index(): return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/game_state', methods=['GET'])
def game_state():
    global current_dqn_model_path_relative
    return jsonify({
        'board': game_instance.get_board(),
        'score': game_instance.get_score(),
        'game_over': game_instance.is_game_over()[0].item(),
        'current_player': current_player_mode,
        'active_ai_agent_type': active_ai_agent_type,
        'dqn_model_available': dqn_ai_player is not None,
        'loaded_dqn_model_path': current_dqn_model_path_relative if dqn_ai_player is not None else None
    })

@app.route('/api/move', methods=['POST'])
def move():
    global current_player_mode
    if current_player_mode != "human": return jsonify({'error': 'Not human player turn'}), 400
    
    data = request.json
    direction_str = data.get('direction')
    if not direction_str or direction_str not in Game.DIRECTIONS_MAP:
        return jsonify({'error': f'Direction not provided or invalid: {direction_str}'}), 400
    
    action_idx = Game.DIRECTIONS_MAP[direction_str]
    action_tensor = torch.tensor([action_idx], device=game_instance.device, dtype=torch.long)
    
    moved_mask = game_instance.move(action_tensor)
    action_resulted_in_change = moved_mask[0].item()

    return jsonify({
        'board': game_instance.get_board(), 
        'score': game_instance.get_score(),
        'game_over': game_instance.is_game_over()[0].item(), 
        'current_player': current_player_mode,
        'action_taken': action_resulted_in_change, 
        'active_ai_agent_type': active_ai_agent_type, 
        'dqn_model_available': dqn_ai_player is not None,
        'loaded_dqn_model_path': current_dqn_model_path_relative if dqn_ai_player is not None else None
    })


@app.route('/api/ai_move', methods=['POST'])
def ai_move():
    global current_player_mode, current_ai_instance 
    if current_player_mode != "ai": return jsonify({'error': 'Not AI player turn'}), 400
    
    if current_ai_instance is None: 
        print("[AI_MOVE_ERROR] current_ai_instance is None. Defaulting to dummy and attempting reload if DQN was active.")
        current_ai_instance = dummy_ai_player 
        if active_ai_agent_type == "dqn": 
            load_dqn_for_play() # Try to reload DQN (latest)
            if dqn_ai_player: current_ai_instance = dqn_ai_player # if successful, use it

    ai_decision_str = None
    ai_moved_board = False

    current_board_tensor_for_ai = game_instance.boards.clone() 
    legal_moves_mask_for_ai = game_instance.get_legal_moves()

    if active_ai_agent_type == "dqn" and dqn_ai_player and current_ai_instance == dqn_ai_player :
        action_idx_tensor = dqn_ai_player.get_move(current_board_tensor_for_ai, legal_moves_mask_for_ai)
        action_idx = action_idx_tensor[0].item()
        ai_decision_str = Game.ACTION_LIST[action_idx]
    elif active_ai_agent_type == "dummy" or current_ai_instance == dummy_ai_player: 
        board_list = game_instance.get_board()
        legal_strs = [Game.ACTION_LIST[i] for i, legal in enumerate(legal_moves_mask_for_ai[0]) if legal]
        class MockOldGameForDummy:
            def __init__(self, b, l): self.board=b; self._legal_moves=l
            def get_board(self): return self.board
            def get_legal_moves(self): return self._legal_moves
        mock_game_dummy = MockOldGameForDummy(board_list, legal_strs)
        ai_decision_str = dummy_ai_player.get_move(mock_game_dummy)
    else: 
        print(f"[AI_MOVE_WARN] AI type {active_ai_agent_type} logic issue or dqn_player mismatch. Using random legal move.")
        if legal_moves_mask_for_ai[0].any():
            legal_indices = legal_moves_mask_for_ai[0].nonzero(as_tuple=True)[0]
            action_idx = legal_indices[torch.randint(0, len(legal_indices), (1,))[0]].item()
            ai_decision_str = Game.ACTION_LIST[action_idx]

    if ai_decision_str:
        action_idx_to_move = Game.DIRECTIONS_MAP[ai_decision_str]
        action_tensor = torch.tensor([action_idx_to_move], device=game_instance.device, dtype=torch.long)
        moved_mask = game_instance.move(action_tensor)
        ai_moved_board = moved_mask[0].item()

    return jsonify({
        'board': game_instance.get_board(), 
        'score': game_instance.get_score(),
        'game_over': game_instance.is_game_over()[0].item(), 
        'current_player': current_player_mode,
        'ai_decision': ai_decision_str, 'ai_moved': ai_moved_board,
        'active_ai_agent_type': active_ai_agent_type,
        'dqn_model_available': dqn_ai_player is not None,
        'loaded_dqn_model_path': current_dqn_model_path_relative if dqn_ai_player is not None else None
    })

@app.route('/api/undo', methods=['POST'])
def undo():
    print("Undo endpoint called. Current Game.undo is a stub.")
    game_instance.undo() 
    return jsonify({
        'board': game_instance.get_board(), 
        'score': game_instance.get_score(),
        'game_over': game_instance.is_game_over()[0].item(), 
        'current_player': current_player_mode,
        'active_ai_agent_type': active_ai_agent_type,
        'dqn_model_available': dqn_ai_player is not None,
        'loaded_dqn_model_path': current_dqn_model_path_relative if dqn_ai_player is not None else None
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    global game_instance 
    game_instance.reset() 
    return jsonify({
        'board': game_instance.get_board(), 
        'score': game_instance.get_score(),
        'game_over': game_instance.is_game_over()[0].item(), 
        'current_player': current_player_mode,
        'active_ai_agent_type': active_ai_agent_type,
        'dqn_model_available': dqn_ai_player is not None,
        'loaded_dqn_model_path': current_dqn_model_path_relative if dqn_ai_player is not None else None
    })

@app.route('/api/toggle_player', methods=['POST'])
def toggle_player():
    global current_player_mode
    data = request.json
    new_player = data.get('player')
    if new_player in ["human", "ai"]:
        current_player_mode = new_player
        return jsonify({'status': 'ok', 'current_player': current_player_mode})
    return jsonify({'error': 'Invalid player mode'}), 400

@app.route('/api/models/list_dqn', methods=['GET'])
def get_dqn_models():
    models = list_all_dqn_models()
    return jsonify(models)

@app.route('/api/set_ai_agent', methods=['POST'])
def set_ai_agent():
    global active_ai_agent_type, current_ai_instance, dqn_ai_player, current_dqn_model_path_relative
    data = request.json
    requested_type = data.get('agent_type')

    if requested_type == "dummy":
        active_ai_agent_type = "dummy"
        current_ai_instance = dummy_ai_player
        # dqn_ai_player and current_dqn_model_path_relative are not cleared when switching to dummy.
        # This means a DQN model might still be "loaded" in memory but not active.
        return jsonify({
            'status': 'ok', 
            'active_ai_agent_type': active_ai_agent_type, 
            'message': 'Switched to Dummy AI.',
            'dqn_model_available': dqn_ai_player is not None, # Reflects if a model is in dqn_ai_player
            'loaded_dqn_model_path': current_dqn_model_path_relative # Reflects path if dqn_ai_player not None
            })
    
    elif requested_type == "dqn":
        model_file_relative_path = data.get('model_file') # Relative to DQN_MODEL_DIR or None
        
        full_path_to_load = None
        load_attempt_description = "Latest/Default"

        if model_file_relative_path: # User specified a model
            full_path_to_load = os.path.join(DQN_MODEL_DIR, model_file_relative_path)
            load_attempt_description = model_file_relative_path
            # load_dqn_for_play will check existence, so we don't need to duplicate it here
            # if not os.path.exists(full_path_to_load): ...
        
        if load_dqn_for_play(specific_model_path_full=full_path_to_load):
            active_ai_agent_type = "dqn"
            current_ai_instance = dqn_ai_player # dqn_ai_player is set by load_dqn_for_play
            return jsonify({
                'status': 'ok', 
                'active_ai_agent_type': active_ai_agent_type, 
                'message': f'Switched to DQN AI. Model: {current_dqn_model_path_relative}.',
                'dqn_model_available': True, # dqn_ai_player is now set
                'loaded_dqn_model_path': current_dqn_model_path_relative
            })
        else:
            # Load failed. load_dqn_for_play already set dqn_ai_player = None, current_dqn_model_path_relative = None.
            # It also sets current_ai_instance = dummy_ai_player if active_ai_agent_type (global) was "dqn".
            # Here, we ensure active_ai_agent_type is set to "dummy" because the DQN load attempt failed.
            active_ai_agent_type = "dummy" 
            current_ai_instance = dummy_ai_player 
            
            return jsonify({
                'error': f'DQN model ({load_attempt_description}) failed to load. Switched to Dummy AI.',
                'active_ai_agent_type': active_ai_agent_type, 
                'dqn_model_available': False, # dqn_ai_player is None
                'loaded_dqn_model_path': None
            }), 400
    else: 
        return jsonify({'error': 'Invalid AI agent type.'}), 400


@app.route('/api/training/start', methods=['POST'])
def start_training_route():
    global training_process_handle
    if training_process_handle and training_process_handle.poll() is None:
        return jsonify({'status': 'error', 'message': 'Training already in progress.'}), 400
    try:
        data = request.json
        hyperparams = data.get('hyperparameters', {})
        train_script_path = os.path.join(os.path.dirname(__file__), 'train_dqn.py')
        cmd = [sys.executable, train_script_path]
        
        cmd.extend([f"--num_episodes", str(hyperparams.get('num_episodes', DEFAULT_NUM_EPISODES_TOTAL))])
        cmd.extend([f"--env_batch_size", str(hyperparams.get('env_batch_size', DEFAULT_ENV_BATCH_SIZE))])
        cmd.extend([f"--lr", str(hyperparams.get('lr', DEFAULT_LEARNING_RATE))])
        cmd.extend([f"--hidden_size", str(hyperparams.get('hidden_size', DEFAULT_HIDDEN_SIZE))])
        cmd.extend([f"--gamma", str(hyperparams.get('gamma', DEFAULT_GAMMA))])
        cmd.extend([f"--eps_start", str(hyperparams.get('eps_start', DEFAULT_EPSILON_START))])
        cmd.extend([f"--eps_end", str(hyperparams.get('eps_end', DEFAULT_EPSILON_END))])
        cmd.extend([f"--eps_decay", str(hyperparams.get('eps_decay', DEFAULT_EPSILON_DECAY_FRAMES))])
        cmd.extend([f"--buffer_size", str(hyperparams.get('buffer_size', DEFAULT_REPLAY_BUFFER_SIZE))])
        cmd.extend([f"--learn_batch_size", str(hyperparams.get('learn_batch_size', DEFAULT_LEARN_BATCH_SIZE_TRAIN))])
        cmd.extend([f"--target_update", str(hyperparams.get('target_update', DEFAULT_TARGET_UPDATE_FREQ_STEPS_TRAIN))])
        cmd.extend([f"--learn_start", str(hyperparams.get('learn_start', DEFAULT_LEARN_START_FRAMES_TRAIN))])
        
        print(f"[TRAINING_START] CMD: {' '.join(cmd)}")
        cwd = os.path.dirname(os.path.abspath(__file__)) 
        training_env = os.environ.copy()
        training_process_handle = subprocess.Popen(cmd, cwd=cwd, env=training_env)
        return jsonify({'status': 'ok', 'message': f'Training started (PID: {training_process_handle.pid}).'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to start training: {str(e)}'}), 500

@app.route('/api/training/status', methods=['GET'])
def training_status_route():
    global training_process_handle, current_dqn_model_path_relative
    if training_process_handle:
        if training_process_handle.poll() is None: 
            return jsonify({'status': 'running', 'pid': training_process_handle.pid, 
                            'dqn_model_available': dqn_ai_player is not None,
                            'loaded_dqn_model_path': current_dqn_model_path_relative})
        else: 
            return_code = training_process_handle.returncode
            training_process_handle = None 
            print(f"Training process finished (code {return_code}). Attempting to reload DQN model...")
            dqn_reloaded = load_dqn_for_play() # Loads latest
            msg = f'Training finished (code {return_code}). Model reload: {"OK" if dqn_reloaded else "Failed or no new model"}.'
            if dqn_reloaded: msg += f' Loaded: {current_dqn_model_path_relative}'
            print(msg)
            return jsonify({'status': 'finished', 'return_code': return_code, 'message': msg, 
                            'dqn_model_available': dqn_ai_player is not None,
                            'loaded_dqn_model_path': current_dqn_model_path_relative})
    return jsonify({'status': 'idle', 
                    'dqn_model_available': dqn_ai_player is not None,
                    'loaded_dqn_model_path': current_dqn_model_path_relative})

@app.route('/api/training/stop', methods=['POST'])
def stop_training_route():
    global training_process_handle, current_dqn_model_path_relative
    if training_process_handle and training_process_handle.poll() is None:
        try:
            print("Terminating training process...")
            training_process_handle.terminate() 
            training_process_handle.wait(timeout=10) 
            msg = 'Training process terminated.'
            print(msg)
        except subprocess.TimeoutExpired:
            print("Training process did not terminate gracefully, killing...")
            training_process_handle.kill() 
            training_process_handle.wait() 
            msg = 'Training process force-killed.'
            print(msg)
        except Exception as e:
            msg = f'Error stopping training process: {str(e)}'
            print(msg)
        
        training_process_handle = None
        print("Attempting to reload DQN model after stopping training...")
        load_dqn_for_play() # Reload best model available (latest)
        return jsonify({'status': 'ok', 'message': msg, 
                        'dqn_model_available': dqn_ai_player is not None,
                        'loaded_dqn_model_path': current_dqn_model_path_relative})
    
    return jsonify({'status': 'idle', 'message': 'No active training process.', 
                    'dqn_model_available': dqn_ai_player is not None,
                    'loaded_dqn_model_path': current_dqn_model_path_relative})

if __name__ == '__main__':
    static_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_dir_path): os.makedirs(static_dir_path)
    
    model_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DQN_MODEL_DIR)
    runs_base_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_DIR_BASE.split('/')[0])
    
    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(runs_base_dir_path, exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)
