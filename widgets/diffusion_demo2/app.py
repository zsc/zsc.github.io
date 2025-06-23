from flask import Flask, request, jsonify, render_template
import threading
import os
import traceback
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision.utils import make_grid

# Import training functions and models
from training.train_vae import train_vae_model
from training.train_dit import train_dit_model, get_ddpm_params, p_sample_loop
from models.vae import VAE, VQVAE
from models.dit import DiT
from data_loader import IMAGE_SIZE # For VAE/VQVAE input H,W during generation

app = Flask(__name__)

# --- Global state (simple tracking for demo) ---
training_threads = {} # To keep track of running training threads
VAE_MODEL_CONFIG_CACHE = {} # To store VAE config for DiT/Gen
DIT_MODEL_CONFIG_CACHE = {} # To store DiT config for Gen

def run_training_async(training_function, config, job_id, result_cache_key_prefix=""):
    global training_threads
    try:
        print(f"Starting job {job_id} with config: {config}")
        model_path = training_function(config)
        if model_path:
            # Store config used for this model path for later use (e.g., generation)
            if result_cache_key_prefix == "vae_":
                VAE_MODEL_CONFIG_CACHE[model_path] = config
            elif result_cache_key_prefix == "dit_":
                DIT_MODEL_CONFIG_CACHE[model_path] = config
            print(f"Job {job_id} completed. Model saved at: {model_path}")
            # Could update a global status dict here if needed for polling
        else:
            print(f"Job {job_id} completed but no model path returned (possibly an error).")

    except Exception as e:
        print(f"Error in {job_id} thread: {e}")
        traceback.print_exc()
    finally:
        if job_id in training_threads:
            del training_threads[job_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-training-vae', methods=['POST'])
def handle_start_training_vae():
    if 'vae_training' in training_threads and training_threads['vae_training'].is_alive():
        return jsonify({"message": "VAE training is already in progress."}), 400
    
    config = request.json
    print("Received VAE training config:", config)
    
    # Simple validation for required fields (can be more extensive)
    required_keys = ['lr', 'batch_size', 'epochs']
    if not all(key in config for key in required_keys):
        return jsonify({"message": "Missing required VAE parameters."}), 400

    thread = threading.Thread(target=run_training_async, args=(train_vae_model, config, 'vae_training', "vae_"))
    training_threads['vae_training'] = thread
    thread.start()
    
    # The actual model path will be known when training finishes.
    # For now, construct an expected name based on config for UI feedback.
    model_type = "VQVAE" if config.get('use_vq') else "VAE"
    run_name_stem = f"{model_type}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}_ld{config.get('latent_dim', 'vq')}"
    # This is a simplification; actual name might have more parts.
    # The `train_vae_model` function will return the precise path.
    # For now, the JS will get the actual path from the response when training finishes (if we implement status polling or websockets)
    # Or, the UI can just show "training started" and user checks TensorBoard/console logs.
    # Let's modify run_training_async to return path, but Flask needs a way to get it back.
    # For simplicity, let the training function print the path and user can copy it.
    # Or, we can modify the training function to return the path, and a callback updates UI (more complex)
    # The current `run_training_async` is a fire-and-forget.
    # A simple improvement: the training function could write the path to a known file or update a global var.
    # Let's make the endpoint return the *expected* path for now.
    
    # A better approach for this demo: trainer writes path to a temp file, another endpoint reads it.
    # Or simply have the trainer return it and user manually copies it.
    # For now, let's assume the trainer will print the path to console.
    # The UI will fill this path if/when it gets it (e.g., from a future status update endpoint)
    # Modified JS to accept model_path from response if server sends it back.
    # But `run_training_async` runs in a separate thread.
    # The immediate response indicates training start. User has to get path from logs or a future "get_status" endpoint.
    # For the UI, let's just make `train_X_model` return the path and in `app.py`, we can then pass it
    # But that blocks the request.
    # For simplicity, the `run_training_async` now has a basic mechanism to associate config with path.
    # The client-side will rely on the printed path or manually finding it in `checkpoints`.
    # A robust solution needs a proper job queue or status update mechanism.

    # Let's try to make the thread update a global var that app.py can pass back (still hacky)
    # Instead, let's just let the thread save it and the user find it via logs / checkpoints dir.
    # The UI path fields are for user input or to be filled by future 'status' calls.

    return jsonify({"message": f"{model_type} training started. Check console/TensorBoard. Model path will be printed upon completion.", "status": "started"})


@app.route('/start-training-dit', methods=['POST'])
def handle_start_training_dit():
    if 'dit_training' in training_threads and training_threads['dit_training'].is_alive():
        return jsonify({"message": "DiT training is already in progress."}), 400
        
    config = request.json
    print("Received DiT training config:", config)

    required_keys = ['vae_model_path', 'lr', 'batch_size', 'epochs', 'ddpm_timesteps', 
                     'dit_patch_size', 'dit_hidden_size', 'dit_depth', 'dit_num_heads',
                     'vae_latent_channels', 'vae_latent_spatial_dim']
    if not all(key in config for key in required_keys):
        missing = [key for key in required_keys if key not in config]
        return jsonify({"message": f"Missing required DiT parameters: {missing}"}), 400
    
    if not os.path.exists(config['vae_model_path']):
        return jsonify({"message": f"VAE model path {config['vae_model_path']} not found."}), 400

    thread = threading.Thread(target=run_training_async, args=(train_dit_model, config, 'dit_training', "dit_"))
    training_threads['dit_training'] = thread
    thread.start()
    
    return jsonify({"message": "DiT training started. Check console/TensorBoard. Model path will be printed upon completion.", "status": "started"})


@app.route('/generate-samples', methods=['POST'])
def handle_generate_samples():
    if 'generation' in training_threads and training_threads['generation'].is_alive():
        return jsonify({"error": "Another generation task is in progress."}), 400

    config = request.json
    print("Received generation config:", config)
    
    # --- Parameter Validation (Simplified) ---
    required_gen_keys = [
        'vae_model_path', 'dit_model_path', 'ddpm_timesteps', 
        'dit_patch_size', 'dit_hidden_size', 'dit_depth', 'dit_num_heads',
        'vae_latent_channels', 'vae_latent_spatial_dim', 'num_samples'
    ]
    if not all(key in config for key in required_gen_keys):
        missing = [key for key in required_gen_keys if key not in config]
        return jsonify({"error": f"Missing required generation parameters: {missing}"}), 400

    if not os.path.exists(config['vae_model_path']):
        return jsonify({"error": f"VAE model {config['vae_model_path']} not found."}), 400
    if not os.path.exists(config['dit_model_path']):
        return jsonify({"error": f"DiT model {config['dit_model_path']} not found."}), 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bfloat16 = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()

    try:
        # --- Load VAE/VQVAE Model ---
        # Infer type and params (similar to train_dit.py, but needs to be more robust for generation)
        # For simplicity, assume UI provides enough info, or we retrieve from VAE_MODEL_CONFIG_CACHE
        vae_config_used = VAE_MODEL_CONFIG_CACHE.get(config['vae_model_path'], {}) # Get cached config if available
        is_vqvae = "VQVAE" in config['vae_model_path'] # Basic inference

        vae_latent_channels = config['vae_latent_channels']
        vae_latent_spatial_dim = config['vae_latent_spatial_dim']
        
        # Determine VAE's latent characteristics for DiT input
        if is_vqvae:
            vq_embedding_dim = config.get('vq_embedding_dim_for_dit', vae_config_used.get('embedding_dim', 64))
            vq_num_embeddings = config.get('vq_num_embeddings_for_dit', vae_config_used.get('num_embeddings', 512))
            commitment_cost = vae_config_used.get('commitment_cost', 0.25)
            # These hidden_dims should ideally come from saved VAE config
            vq_hidden_dims_enc = vae_config_used.get('vq_hidden_dims_enc', [128, vq_embedding_dim])
            vq_hidden_dims_dec = vae_config_used.get('vq_hidden_dims_dec', [128])

            autoencoder = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                                num_embeddings=vq_num_embeddings, commitment_cost=commitment_cost,
                                hidden_dims_enc=vq_hidden_dims_enc, hidden_dims_dec=vq_hidden_dims_dec
                               ).to(device)
            vae_latent_channels = vq_embedding_dim # DiT operates on VQ embedding dim
            # VQVAE spatial dim: probe or get from config
            if 'vae_latent_spatial_dim' not in config: # Try to infer if not given
                dummy_vq_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
                z_e_dummy = autoencoder.encoder(dummy_vq_input.to(next(autoencoder.parameters()).dtype)) # Match dtype
                _, _, vae_latent_spatial_dim, _ = z_e_dummy.shape
                del dummy_vq_input, z_e_dummy
                print(f"Inferred VQVAE latent spatial dim for generation: {vae_latent_spatial_dim}")
            else:
                 vae_latent_spatial_dim = config['vae_latent_spatial_dim']


        else: # Standard VAE
            vae_flat_latent_dim = config.get('vae_flat_latent_dim', vae_config_used.get('latent_dim', 256))
            autoencoder = VAE(input_channels=3, latent_dim=vae_flat_latent_dim).to(device)
        
        autoencoder.load_state_dict(torch.load(config['vae_model_path'], map_location=device))
        autoencoder.eval()
        print(f"Loaded {'VQVAE' if is_vqvae else 'VAE'} model from {config['vae_model_path']}")

        # --- Load DiT Model ---
        dit_model = DiT(
            latent_shape=(vae_latent_channels, vae_latent_spatial_dim, vae_latent_spatial_dim),
            patch_size=config['dit_patch_size'],
            in_channels=vae_latent_channels,
            hidden_size=config['dit_hidden_size'],
            depth=config['dit_depth'],
            num_heads=config['dit_num_heads']
        ).to(device)
        dit_model.load_state_dict(torch.load(config['dit_model_path'], map_location=device))
        dit_model.eval()
        print(f"Loaded DiT model from {config['dit_model_path']}")

        # --- DDPM Params ---
        timesteps = config['ddpm_timesteps']
        ddpm_params = get_ddpm_params("linear", timesteps, device) # Assuming linear schedule

        # --- Generate ---
        num_samples = config['num_samples']
        latent_sample_shape = (num_samples, vae_latent_channels, vae_latent_spatial_dim, vae_latent_spatial_dim)
        
        vae_decoder_fn = autoencoder.decode
        
        # Autocast for DiT and VAE models during inference
        # The p_sample_loop handles autocast for vae_decoder internally if its params are bf16
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
            generated_output, _ = p_sample_loop(dit_model, latent_sample_shape, timesteps,
                                                ddpm_params, device, vae_decoder=vae_decoder_fn)
        
        # generated_output is now image pixels if vae_decoder_fn was used
        generated_images_unnorm = (generated_output.clamp(-1, 1) + 1) / 2 # Unnormalize
        
        # Convert to base64 PNGs
        base64_images = []
        for i in range(num_samples):
            img_tensor = generated_images_unnorm[i].cpu()
            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(img_tensor)
            buffered = BytesIO()
            img_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(img_str)
        
        print(f"Generated {len(base64_images)} samples.")
        return jsonify({"images": base64_images})

    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if 'generation' in training_threads: # Should not happen with current sync generation
            del training_threads['generation']


if __name__ == '__main__':
    # Add torchvision.transforms to global scope for PIL conversion if not already done
    from torchvision import transforms 
    # Ensure necessary directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("results/vae_samples", exist_ok=True)
    os.makedirs("results/dit_samples", exist_ok=True)
    os.makedirs("cache", exist_ok=True) # For data_loader cache
    
    app.run(debug=True, host='0.0.0.0', port=5000)
