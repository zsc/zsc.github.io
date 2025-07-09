#!/usr/bin/env python3
# app.py
import os
import sys
import threading
import subprocess
import time
import glob
import re
import argparse
import collections
from flask import Flask, render_template, jsonify, request

import torch
import numpy as np

# Add the current directory to sys.path to ensure modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your custom modules
from game import Game2048
from agent import Agent
from train import train as run_training_process

# --- Global State Management ---

# Flask App
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 # Disable caching for development

# Training State
training_thread = None
training_status = {
    'status': 'idle', # idle, running, finished, error
    'progress': 0,
    'total_steps': 0,
    'message': 'Training has not started.',
    'tensorboard_url': None,
    'exp_name': None,
    'log_lines': collections.deque(maxlen=100) # Store recent log lines
}
tensorboard_process = None

# Testing State
test_agent = None
# Initialize Game2048 tables once
Game2048._init_tables()


# --- Background Training Wrapper ---

def train_runner(args_dict):
    """
    A wrapper function to run the training process in a background thread
    and update the global status dictionary.
    """
    global training_status, tensorboard_process

    original_stdout = sys.stdout
    try:
        # Convert dict to argparse.Namespace for the training script
        args = argparse.Namespace(**args_dict)

        # --- Reset and Update Status ---
        training_status['status'] = 'running'
        training_status['message'] = 'Initializing training...'
        training_status['progress'] = 0
        training_status['total_steps'] = args.total_steps
        training_status['log_lines'].clear()
        
        # --- Create Experiment Name and Dirs (mirrors train.py logic) ---
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        exp_name = f"2048_transformer_edim{args.embed_dim}_h{args.num_heads}_l{args.num_layers}_lr{args.lr}_{timestamp}"
        training_status['exp_name'] = exp_name
        log_dir = os.path.join("runs", exp_name)
        
        # --- Launch TensorBoard ---
        # Kill previous process if it exists
        if tensorboard_process:
            tensorboard_process.kill()
            
        tb_command = [
            sys.executable, '-m', 'tensorboard.main', 
            '--logdir', 'runs', '--port', '6006', '--host', '0.0.0.0'
        ]
        # Use Popen to run in the background without blocking
        tensorboard_process = subprocess.Popen(tb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5) # Give it a moment to start
        
        # --- Redirect stdout to capture logs ---
        # This is a simple way to capture print statements from the training script
        class LogCatcher:
            def write(self, message):
                if message.strip():
                    training_status['log_lines'].append(message.strip())
                sys.__stdout__.write(message) # Also print to console
            def flush(self):
                sys.__stdout__.flush()

        sys.stdout = LogCatcher()

        # --- Run Training ---
        # The main call to your training script
        run_training_process(args)

        # --- Finalize ---
        training_status['status'] = 'finished'
        training_status['message'] = 'Training completed successfully.'
        training_status['progress'] = args.total_steps

    except Exception as e:
        training_status['status'] = 'error'
        training_status['message'] = f"An error occurred: {str(e)}"
        import traceback
        training_status['log_lines'].append(traceback.format_exc())
    finally:
        # Restore stdout
        sys.stdout = original_stdout


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Starts the training process in a background thread."""
    global training_thread
    if training_thread and training_thread.is_alive():
        return jsonify({'status': 'error', 'message': 'Training is already in progress.'}), 400

    params = request.json
    # Start the training runner in a new thread
    training_thread = threading.Thread(target=train_runner, args=(params,), daemon=True)
    training_thread.start()

    return jsonify({'status': 'success', 'message': 'Training started.'})

@app.route('/api/get_status', methods=['GET'])
def get_status():
    """Returns the current status of the training process."""
    # Convert deque to list for JSON serialization
    status_copy = training_status.copy()
    status_copy['log_lines'] = list(status_copy['log_lines'])
    return jsonify(status_copy)

@app.route('/api/get_checkpoints', methods=['GET'])
def get_checkpoints():
    """Scans the checkpoints directory and returns available models."""
    base_path = 'checkpoints'
    if not os.path.exists(base_path):
        return jsonify([])
    
    checkpoints = []
    exp_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for exp_name in sorted(exp_dirs, reverse=True):
        # Parse hyperparams from the directory name
        match_dim = re.search(r'edim(\d+)', exp_name)
        match_heads = re.search(r'h(\d+)', exp_name)
        match_layers = re.search(r'l(\d+)', exp_name)
        
        if not (match_dim and match_heads and match_layers):
            continue

        hyperparams = {
            'embed_dim': int(match_dim.group(1)),
            'num_heads': int(match_heads.group(1)),
            'num_layers': int(match_layers.group(1))
        }

        # Find all .pth files in the directory
        ckpt_dir_path = os.path.join(base_path, exp_name)
        files = sorted(glob.glob(os.path.join(ckpt_dir_path, '*.pth')))
        if files:
            checkpoints.append({
                'name': exp_name,
                'path': files[-1], # Use the latest checkpoint file
                'hyperparams': hyperparams
            })
            
    return jsonify(checkpoints)

@app.route('/api/start_test', methods=['POST'])
def start_test():
    """Loads a model and starts a new test game."""
    global test_agent
    data = request.json
    checkpoint_path = data.get('path')
    hyperparams = data.get('hyperparams')

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return jsonify({'status': 'error', 'message': 'Checkpoint not found.'}), 404

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_agent = Agent(
            embed_dim=hyperparams['embed_dim'],
            num_heads=hyperparams['num_heads'],
            num_layers=hyperparams['num_layers'],
            device=device
        )
        test_agent.load_checkpoint(checkpoint_path)
        test_agent.policy_net.eval()
        
        board_int = Game2048.reset_board()
        board_array = Game2048.get_board_array(board_int)
        
        return jsonify({
            'status': 'success',
            'board': board_array.tolist(),
            # FIX: Convert large integer to string to prevent precision loss in JavaScript
            'board_int': str(board_int),
            'max_tile': Game2048.get_max_tile(board_int)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to load agent: {e}'}), 500

@app.route('/api/test_move', methods=['POST'])
def test_move():
    """Performs one AI move in the test game."""
    if not test_agent:
        return jsonify({'status': 'error', 'message': 'Test agent not loaded.'}), 400

    data = request.json
    # Note: `int()` correctly handles string-to-integer conversion from JSON
    board_int = int(data.get('board_int'))
    budget = int(data.get('search_budget'))

    action = test_agent.act_with_search(board_int, budget=budget)
    
    if action == -1: # No legal moves
        return jsonify({
            'is_done': True,
            'message': 'Game Over. No legal moves.'
        })

    move_func = test_agent.moves[action]
    new_board_int, score_gained, moved = move_func(board_int)

    final_board_int = new_board_int
    if moved:
        final_board_int = Game2048.add_random_tile(new_board_int)

    is_done = Game2048.is_game_over(final_board_int)
    board_array = Game2048.get_board_array(final_board_int)
    
    return jsonify({
        'board': board_array.tolist(),
        # FIX: Convert large integer to string to prevent precision loss in JavaScript
        'board_int': str(final_board_int),
        'score_gained': score_gained,
        'is_done': is_done,
        'action': ['Up', 'Down', 'Left', 'Right'][action],
        'max_tile': Game2048.get_max_tile(final_board_int)
    })


if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
