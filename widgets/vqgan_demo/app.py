# app.py
import os
import io
import json
import base64
import threading
import queue
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import torch
from torchvision.utils import save_image
import numpy as np

# 假设 model.py 和 train.py 在同一目录下
from model import VQGAN, VQGANTransformer
from train import get_data_loader, train_stage1, train_stage2

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key' # 用于 Flask session 等，虽然此 demo 没用到
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('static/test_outputs', exist_ok=True) # 用于存放测试图片

# --- Global State for Training Management ---
# 使用一个字典来封装，以便在函数间传递引用
training_state = {
    'thread': None,
    'log_queue': queue.Queue(),
    'controls': {'stop': False},
    'is_running': False,
    'current_stage': None
}

DATA_FILE_PATH = 'celeba_cache_64x64.npz'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions ---
def to_base64(image_tensor):
    """Converts a single image tensor to a base64 string."""
    buffer = io.BytesIO()
    # Normalize from [-1, 1] to [0, 1] for saving
    save_image(image_tensor, buffer, format='PNG', normalize=True, value_range=(-1, 1))
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Returns the current training status."""
    return jsonify({
        'is_running': training_state['is_running'],
        'current_stage': training_state['current_stage']
    })

@app.route('/logs')
def stream_logs():
    """Streams training logs using Server-Sent Events (SSE)."""
    def generate():
        while True:
            try:
                message = training_state['log_queue'].get(timeout=1)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # If training is not running, stop sending keep-alive packets
                if not training_state['is_running']:
                    break
                # Send a keep-alive comment to prevent the connection from closing
                yield ": keep-alive\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/get_checkpoints')
def get_checkpoints():
    """Scans the checkpoints directory and returns available .pt files."""
    checkpoints = {'stage1': [], 'stage2': []}
    if not os.path.exists('checkpoints'):
        return jsonify(checkpoints)
    
    for root, _, files in os.walk('checkpoints'):
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                if 'vqgan' in file.lower() or 'stage1' in root.lower():
                    checkpoints['stage1'].append(full_path)
                elif 'transformer' in file.lower() or 'stage2' in root.lower():
                    checkpoints['stage2'].append(full_path)
    # Sort them for consistency
    checkpoints['stage1'].sort()
    checkpoints['stage2'].sort()
    return jsonify(checkpoints)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Starts a training process in a background thread."""
    if training_state['is_running']:
        return jsonify({'status': 'error', 'message': 'A training process is already running.'}), 400

    data = request.json
    stage = data.get('stage')

    # Reset state
    training_state['controls']['stop'] = False
    training_state['is_running'] = True
    training_state['current_stage'] = stage
    
    # Clear queue
    while not training_state['log_queue'].empty():
        training_state['log_queue'].get()

    def training_wrapper(config, stage_func):
        try:
            stage_func(config, training_state['controls'], training_state['log_queue'], DEVICE)
        except Exception as e:
            import traceback
            error_msg = f"Unhandled exception in training thread: {e}\n{traceback.format_exc()}"
            training_state['log_queue'].put(error_msg)
        finally:
            # Signal completion
            training_state['is_running'] = False
            training_state['current_stage'] = None
            training_state['log_queue'].put("---TRAINING-COMPLETE---")


    if stage == 1:
        config = {
            'data_path': DATA_FILE_PATH,
            'epochs': data.get('s1_epochs', 100),
            'batch_size': 32, # Hardcoded for 4090 memory
            'lr': 1e-4,
            'n_embed': data.get('s1_n_embed', 512),
            'commitment_cost': data.get('s1_commitment_cost', 0.25),
            'gan_weight': 0.03,
            'save_epoch_freq': data.get('s1_save_epoch_freq', 1),
            'checkpoint_path': data.get('s1_checkpoint_path') or None
        }
        target_func = train_stage1
    elif stage == 2:
        config = {
            'data_path': DATA_FILE_PATH,
            'vqgan_checkpoint_path': data.get('s2_vqgan_checkpoint_path'),
            'epochs': data.get('s2_epochs', 100),
            'batch_size': 32, # Hardcoded
            'lr': 1e-4,
            'n_layer': data.get('s2_n_layer', 8),
            'n_head': data.get('s2_n_head', 8),
            'n_embd': data.get('s2_n_embd', 512),
            'save_epoch_freq': data.get('s2_save_epoch_freq', 5),
            'checkpoint_path': data.get('s2_checkpoint_path') or None
        }
        if not config['vqgan_checkpoint_path']:
             return jsonify({'status': 'error', 'message': 'Stage 2 training requires a VQGAN checkpoint.'}), 400
        target_func = train_stage2
    else:
        return jsonify({'status': 'error', 'message': 'Invalid stage specified.'}), 400

    training_state['thread'] = threading.Thread(target=training_wrapper, args=(config, target_func))
    training_state['thread'].start()

    return jsonify({'status': 'success', 'message': f'Stage {stage} training started.'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Signals the training thread to stop."""
    if not training_state['is_running']:
        return jsonify({'status': 'error', 'message': 'No training process is running.'}), 400

    training_state['log_queue'].put("Stop signal received. Finishing current step...")
    training_state['controls']['stop'] = True
    # The thread will clean up itself. We don't join here to keep the UI responsive.
    # The wrapper function handles setting 'is_running' to False.
    return jsonify({'status': 'success', 'message': 'Stop signal sent.'})

@app.route('/test_reconstruction', methods=['POST'])
def test_reconstruction():
    """Tests VQGAN reconstruction."""
    data = request.json
    ckpt_path = data.get('checkpoint_path')
    if not ckpt_path:
        return jsonify({'status': 'error', 'message': 'No checkpoint path provided.'}), 400

    try:
        if not os.path.exists(DATA_FILE_PATH):
            return jsonify({'status': 'error', 'message': f'Data file not found: {DATA_FILE_PATH}'}), 400

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model = VQGAN(checkpoint['config']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get data
        dataloader = get_data_loader(DATA_FILE_PATH, batch_size=6, shuffle=True)
        images = next(iter(dataloader)).to(DEVICE)

        # Reconstruct
        with torch.no_grad():
            reconstructions, _, _ = model(images)
        
        # Convert to base64
        results = []
        for i in range(images.size(0)):
            results.append({
                'original': to_base64(images[i].cpu()),
                'reconstructed': to_base64(reconstructions[i].cpu())
            })

        return jsonify({'status': 'success', 'images': results})

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': f"An error occurred: {e}\n{traceback.format_exc()}"}), 500

@app.route('/test_generation', methods=['POST'])
def test_generation():
    """Tests Transformer generation."""
    data = request.json
    vqgan_ckpt_path = data.get('vqgan_checkpoint_path')
    transformer_ckpt_path = data.get('transformer_checkpoint_path')

    if not vqgan_ckpt_path or not transformer_ckpt_path:
        return jsonify({'status': 'error', 'message': 'Both VQGAN and Transformer checkpoints are required.'}), 400

    try:
        # Load VQGAN
        vqgan_ckpt = torch.load(vqgan_ckpt_path, map_location=DEVICE)
        vqgan_model = VQGAN(vqgan_ckpt['config']).to(DEVICE)
        vqgan_model.load_state_dict(vqgan_ckpt['model_state_dict'])
        vqgan_model.eval()

        # Load Transformer
        transformer_ckpt = torch.load(transformer_ckpt_path, map_location=DEVICE)
        transformer_model = VQGANTransformer(transformer_ckpt['config']).to(DEVICE)
        transformer_model.load_state_dict(transformer_ckpt['model_state_dict'])
        transformer_model.eval()

        # Generate
        with torch.no_grad():
            n_samples = 6
            seq_len = transformer_ckpt['config']['block_size']
            start_token = transformer_ckpt['config']['vocab_size']
            
            sampled_indices = transformer_model.sample(
                n_samples=n_samples, seq_len=seq_len, start_token=start_token, device=DEVICE
            )
            
            # Reshape indices to be 2D latent map
            f = 2**(len(vqgan_ckpt['config']['ch_mult'])-1)
            latent_h = vqgan_ckpt['config']['resolution'] // f
            latent_w = latent_h
            sampled_indices = sampled_indices.view(n_samples, latent_h, latent_w)

            generated_images = vqgan_model.decode_from_indices(sampled_indices)

        # Convert to base64
        results = [to_base64(img.cpu()) for img in generated_images]
        return jsonify({'status': 'success', 'images': results})

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': f"An error occurred: {e}\n{traceback.format_exc()}"}), 500


if __name__ == '__main__':
    print(f"Starting server... PyTorch is using device: {DEVICE}")
    if not os.path.exists(DATA_FILE_PATH):
        print(f"\nWARNING: Data file '{DATA_FILE_PATH}' not found.")
        print("Please ensure the preprocessed CelebA cache is in the root directory.\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
