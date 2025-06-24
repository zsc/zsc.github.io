from flask import Flask, request, jsonify, render_template
import threading
import os
import traceback
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision import transforms # Added import

# Import training functions and models
from training.train_vae import train_vae_model
from training.train_dit import train_dit_model, get_ddpm_params, p_sample_loop
from models.vae import VAE, VQVAE # Discriminator is imported within train_dit if needed
from models.dit import DiT
from data_loader import IMAGE_SIZE # Default IMAGE_SIZE

app = Flask(__name__)

# --- Global state (simple tracking for demo) ---
training_threads = {} # To keep track of running training threads
VAE_MODEL_CONFIG_CACHE = {} # To store VAE config for DiT/Gen
DIT_MODEL_CONFIG_CACHE = {} # To store DiT config for Gen

def parse_hidden_dims(dims_str):
    if not dims_str:
        return []
    try:
        return [int(d.strip()) for d in dims_str.split(',')]
    except ValueError:
        return [] # Or raise error/log warning

def run_training_async(training_function, config, job_id, result_cache_key_prefix=""):
    global training_threads
    try:
        print(f"Starting job {job_id} with config: {config}")
        model_path = training_function(config)
        if model_path:
            config_to_cache = config.copy()
            # Ensure image_size used for training is in the cache.
            # It might come from config directly, or be a global default.
            # The training scripts use IMAGE_SIZE from data_loader if not overridden.
            config_to_cache['image_size'] = config.get('image_size', IMAGE_SIZE) 
            
            if result_cache_key_prefix == "vae_":
                VAE_MODEL_CONFIG_CACHE[model_path] = config_to_cache
            elif result_cache_key_prefix == "dit_":
                DIT_MODEL_CONFIG_CACHE[model_path] = config_to_cache
            print(f"Job {job_id} completed. Model saved at: {model_path}. Config cached.")
            
            # For VAE+GAN, also cache discriminator path if available for DiT reference
            if result_cache_key_prefix == "vae_" and config.get('use_gan', False) and "_generator_final.pth" in model_path:
                discriminator_path = model_path.replace("_generator_final.pth", "_discriminator_final.pth")
                if os.path.exists(discriminator_path):
                     # Add it to the main VAE model's cache entry or a separate system
                    VAE_MODEL_CONFIG_CACHE[model_path]['discriminator_path_trained_with_vae'] = discriminator_path
                    print(f"Associated discriminator path {discriminator_path} cached with VAE model {model_path}")

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
    
    required_keys = ['lr', 'batch_size', 'epochs']
    if not all(key in config for key in required_keys):
        return jsonify({"message": "Missing required VAE parameters (lr, batch_size, epochs)."}), 400

    if config.get('use_vq', False):
        if 'vq_hidden_dims_enc_str' in config: # Comes from JS if user entered text
            config['vq_hidden_dims_enc'] = parse_hidden_dims(config.pop('vq_hidden_dims_enc_str', ""))
        if 'vq_hidden_dims_dec_str' in config:
            config['vq_hidden_dims_dec'] = parse_hidden_dims(config.pop('vq_hidden_dims_dec_str', ""))
    
    thread = threading.Thread(target=run_training_async, args=(train_vae_model, config, 'vae_training', "vae_"))
    training_threads['vae_training'] = thread
    thread.start()
    
    model_type = "VQVAE" if config.get('use_vq') else "VAE"
    if config.get('use_gan', False):
        model_type += "+GAN"
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

    # --- New: Handle GAN parameters for DiT training ---
    if config.get('use_gan_for_dit', False):
        if not config.get('vae_discriminator_path'):
            return jsonify({"message": "If 'use_gan_for_dit' is true, 'vae_discriminator_path' must be provided."}), 400
        if not os.path.exists(config['vae_discriminator_path']):
            return jsonify({"message": f"VAE Discriminator path for DiT GAN loss {config['vae_discriminator_path']} not found."}), 400
        if 'gan_loss_weight_dit' not in config: # Ensure weight is present if GAN is used
            # Could default here or require from client
            return jsonify({"message": "If 'use_gan_for_dit' is true, 'gan_loss_weight_dit' must be provided."}), 400
    # --- End New ---
            
    thread = threading.Thread(target=run_training_async, args=(train_dit_model, config, 'dit_training', "dit_"))
    training_threads['dit_training'] = thread
    thread.start()
    
    dit_train_type = "DiT"
    if config.get('use_gan_for_dit', False):
        dit_train_type += "+GAN"

    return jsonify({"message": f"{dit_train_type} training started. Check console/TensorBoard. Model path will be printed upon completion.", "status": "started"})


@app.route('/generate-samples', methods=['POST'])
def handle_generate_samples():
    if 'generation' in training_threads and training_threads['generation'].is_alive():
        return jsonify({"error": "Another generation task is in progress."}), 400

    config = request.json
    print("Received generation config:", config)
    
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
    # Determine bfloat16 use for generation (can be independent of training)
    use_bfloat16_gen = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()


    try:
        # --- Load VAE/VQVAE Model ---
        vae_model_path = config['vae_model_path']
        # Try to get VAE's training config from cache to infer its architecture
        vae_config_from_cache = VAE_MODEL_CONFIG_CACHE.get(vae_model_path, {})
        
        is_vqvae = vae_config_from_cache.get('use_vq', "VQVAE" in vae_model_path.upper())
        
        # Determine image_size: Cache -> Gen Form (if added) -> Global Default
        vae_image_size = vae_config_from_cache.get('image_size', config.get('generation_image_size', IMAGE_SIZE))


        if is_vqvae:
            # For VQVAE, parameters from its training are most reliable
            vq_embedding_dim = vae_config_from_cache.get('embedding_dim', config.get('vq_embedding_dim', 64)) # gen form 'vq_embedding_dim' for VQVAE
            vq_num_embeddings = vae_config_from_cache.get('num_embeddings', config.get('vq_num_embeddings', 512)) # gen form 'vq_num_embeddings'
            commitment_cost = vae_config_from_cache.get('commitment_cost', 0.25)
            
            vq_hidden_dims_enc_default = [128, vq_embedding_dim] if vq_embedding_dim else [128, 64]
            vq_hidden_dims_dec_default = [vq_embedding_dim // 2] if vq_embedding_dim else [32] # Simpler default
            
            vq_hidden_dims_enc = vae_config_from_cache.get('vq_hidden_dims_enc', vq_hidden_dims_enc_default)
            vq_hidden_dims_dec = vae_config_from_cache.get('vq_hidden_dims_dec', vq_hidden_dims_dec_default)

            autoencoder = VQVAE(input_channels=3, 
                                embedding_dim=vq_embedding_dim, 
                                num_embeddings=vq_num_embeddings, 
                                commitment_cost=commitment_cost,
                                hidden_dims_enc=vq_hidden_dims_enc, 
                                hidden_dims_dec=vq_hidden_dims_dec,
                                image_size=vae_image_size
                               ).to(device)
            vae_latent_channels_for_dit = vq_embedding_dim # For DiT, this is VQ's embedding_dim
            # VAE Latent Spatial Dim for DiT is what DiT was trained with. VQVAE output must match this.
            # User provides this in generation form: config['vae_latent_spatial_dim']

        else: # Standard VAE
            vae_flat_latent_dim = vae_config_from_cache.get('latent_dim', config.get('vae_flat_latent_dim', 256))
            vae_hidden_dims_enc = vae_config_from_cache.get('hidden_dims_enc', None) # VAE class defaults
            vae_hidden_dims_dec = vae_config_from_cache.get('hidden_dims_dec', None) # VAE class defaults
            autoencoder = VAE(input_channels=3, 
                              latent_dim=vae_flat_latent_dim,
                              hidden_dims_enc=vae_hidden_dims_enc,
                              hidden_dims_dec=vae_hidden_dims_dec,
                              image_size=vae_image_size
                             ).to(device)
            # For DiT, VAE latent is reshaped. Channels and Spatial Dim from Gen Form.
            vae_latent_channels_for_dit = config['vae_latent_channels'] 
            # vae_latent_spatial_dim_for_dit = config['vae_latent_spatial_dim'] (used below for DiT init)

        # Load state dict for autoencoder
        autoencoder_state_dict = torch.load(vae_model_path, map_location=device)
        autoencoder.load_state_dict(autoencoder_state_dict)
        autoencoder.eval()
        print(f"Loaded {'VQVAE' if is_vqvae else 'VAE'} model from {vae_model_path} (Image Size: {vae_image_size})")

        # --- Load DiT Model ---
        # DiT architecture is defined by parameters from the generation form,
        # which should match how the DiT model was trained.
        dit_model = DiT(
            latent_shape=(config['vae_latent_channels'], config['vae_latent_spatial_dim'], config['vae_latent_spatial_dim']),
            patch_size=config['dit_patch_size'],
            in_channels=config['vae_latent_channels'],
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
        latent_sample_shape_for_dit = (num_samples, config['vae_latent_channels'], config['vae_latent_spatial_dim'], config['vae_latent_spatial_dim'])
        
        vae_decoder_fn = autoencoder.decode
        # Determine if VAE's latents are flat for p_sample_loop logic during decoding
        # This primarily applies to standard VAEs where DiT output (spatial) needs flattening before VAE decode.
        # VQVAE decoder typically expects spatial latents.
        vae_latent_is_flat_for_decode = (not is_vqvae) and vae_config_from_cache.get('vae_latent_is_flat', True) 
                                        # Also consider if 'vae_latent_is_flat' is sent from generation form if needed
        
        # Autocast for fixed models (VAE decoder) during generation if they were trained with bfloat16
        autocast_for_fixed_models_gen = False
        try:
            if next(iter(autoencoder.parameters())).dtype == torch.bfloat16:
                autocast_for_fixed_models_gen = True
        except StopIteration: pass


        # Main generation loop with autocast for DiT and VAE decoder
        with torch.no_grad(): # Overall no_grad for inference
            # Autocast for DiT model's execution
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_gen):
                generated_output, _ = p_sample_loop(
                    dit_model, 
                    latent_sample_shape_for_dit, 
                    timesteps,
                    ddpm_params, 
                    device, 
                    vae_decoder=vae_decoder_fn,
                    vae_latent_is_flat=vae_latent_is_flat_for_decode,
                    autocast_setting_for_vae=autocast_for_fixed_models_gen # Pass specific autocast for VAE
                )
        
        generated_images_unnorm = (generated_output.clamp(-1, 1) + 1) / 2 # To [0,1] range
        
        base64_images = []
        for i in range(num_samples):
            img_tensor = generated_images_unnorm[i].cpu()
            # Ensure img_tensor is 3D (C, H, W) for ToPILImage
            if img_tensor.ndim == 4 and img_tensor.shape[0] == 1: # Squeezed if num_samples was 1 earlier
                img_tensor = img_tensor.squeeze(0)
            
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
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    finally:
        if 'generation' in training_threads: 
            del training_threads['generation'] # Should not be needed for sync op, but safe

if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("results/vae_samples", exist_ok=True)
    os.makedirs("results/dit_samples", exist_ok=True)
    # os.makedirs("cache", exist_ok=True) # Not explicitly used for file caching in this version
    
    app.run(debug=True, host='0.0.0.0', port=5000)
