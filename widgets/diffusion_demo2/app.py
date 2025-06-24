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
from models.vae import VAE, VQVAE # Discriminator is also in models.vae
from models.dit import DiT
from data_loader import IMAGE_SIZE # Default IMAGE_SIZE

app = Flask(__name__, template_folder='templates')

# --- Global state (simple tracking for demo) ---
training_threads = {} # To keep track of running training threads
VAE_MODEL_CONFIG_CACHE = {} # To store VAE config for DiT/Gen
DIT_MODEL_CONFIG_CACHE = {} # To store DiT config for Gen

def parse_hidden_dims(dims_str):
    if not dims_str:
        return []
    try:
        return [int(d.strip()) for d in dims_str.split(',') if d.strip()]
    except ValueError:
        print(f"Warning: Could not parse hidden dimensions string: {dims_str}")
        return []

def run_training_async(training_function, config, job_id, result_cache_key_prefix=""):
    global training_threads
    try:
        print(f"Starting job {job_id} with config: {config}")
        model_path = training_function(config) # This is expected to be the generator/main model path
        
        # Construct full config that was effectively used, including IMAGE_SIZE
        # and potentially paths to auxiliary models like discriminators.
        effective_config = config.copy()
        effective_config['image_size'] = IMAGE_SIZE # Assume global IMAGE_SIZE was used

        if model_path:
            print(f"Job {job_id} completed. Main model saved at: {model_path}.")
            if result_cache_key_prefix == "vae_":
                # If VAE was trained with GAN, its discriminator path needs to be known
                # by DiT if DiT is also using GAN.
                # The train_vae_model currently returns only the VAE (generator) path.
                # The discriminator path is generated predictively in train_vae.py
                # For simplicity, if VAE used GAN, we'll require DiT to be explicitly given the D path.
                VAE_MODEL_CONFIG_CACHE[model_path] = effective_config
                print(f"VAE model config cached for path: {model_path}")

            elif result_cache_key_prefix == "dit_":
                DIT_MODEL_CONFIG_CACHE[model_path] = effective_config
                print(f"DiT model config cached for path: {model_path}")
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

    # Parse hidden dims if provided for VQVAE
    if config.get('use_vq', False):
        # These keys come from the HTML form's input names
        config['vq_hidden_dims_enc'] = config.get('vq_hidden_dims_enc', []) # Already parsed by JS
        config['vq_hidden_dims_dec'] = config.get('vq_hidden_dims_dec', []) # Already parsed by JS

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

    required_keys_base = ['vae_model_path', 'lr', 'batch_size', 'epochs', 'ddpm_timesteps',
                     'dit_patch_size', 'dit_hidden_size', 'dit_depth', 'dit_num_heads',
                     'vae_latent_channels', 'vae_latent_spatial_dim']
    
    current_required_keys = list(required_keys_base) # Make a mutable copy

    if config.get('use_gan', False): # DiT specific GAN usage
        required_keys_gan_dit = ['discriminator_model_path', 'gan_loss_weight', 'lr_d']
        current_required_keys.extend(required_keys_gan_dit)
        if not os.path.exists(config.get('discriminator_model_path','')):
             return jsonify({"message": f"Discriminator model path {config.get('discriminator_model_path')} not found for DiT GAN."}), 400


    if not all(key in config for key in current_required_keys):
        missing = [key for key in current_required_keys if key not in config]
        return jsonify({"message": f"Missing required DiT parameters: {missing}"}), 400

    if not os.path.exists(config['vae_model_path']):
        return jsonify({"message": f"VAE model path {config['vae_model_path']} not found."}), 400

    thread = threading.Thread(target=run_training_async, args=(train_dit_model, config, 'dit_training', "dit_"))
    training_threads['dit_training'] = thread
    thread.start()

    dit_mode = "DiT"
    if config.get('use_gan', False):
        dit_mode += "+GAN"
    return jsonify({"message": f"{dit_mode} training started. Check console/TensorBoard. Model path will be printed upon completion.", "status": "started"})


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
    use_bfloat16 = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()

    try:
        # --- Load VAE/VQVAE Model ---
        vae_model_path = config['vae_model_path']
        # Try to get VAE's original training config from cache
        vae_train_config_from_cache = VAE_MODEL_CONFIG_CACHE.get(vae_model_path, {})
        print(f"VAE training config from cache for {vae_model_path}: {vae_train_config_from_cache}")


        # Determine VAE type and parameters
        is_vqvae = vae_train_config_from_cache.get('use_vq', "VQVAE" in vae_model_path.upper())
        vae_image_size = vae_train_config_from_cache.get('image_size', IMAGE_SIZE) # Important!

        if is_vqvae:
            vq_embedding_dim = vae_train_config_from_cache.get('embedding_dim', config.get('vq_embedding_dim_for_dit', 64))
            vq_num_embeddings = vae_train_config_from_cache.get('num_embeddings', 512) # Default if not in gen form or cache
            commitment_cost = vae_train_config_from_cache.get('commitment_cost', 0.25)
            
            vq_hidden_dims_enc_default = [128, vq_embedding_dim] if vq_embedding_dim else [128, 64]
            vq_hidden_dims_dec_default = [vq_embedding_dim, 128] if vq_embedding_dim else [64,128] # Symmetric to common enc
            
            # Use cached hidden_dims if available, else use default. These are for VQVAE model init.
            vq_hidden_dims_enc = vae_train_config_from_cache.get('vq_hidden_dims_enc', vq_hidden_dims_enc_default)
            vq_hidden_dims_dec = vae_train_config_from_cache.get('vq_hidden_dims_dec', vq_hidden_dims_dec_default)


            autoencoder = VQVAE(input_channels=3,
                                embedding_dim=vq_embedding_dim,
                                num_embeddings=vq_num_embeddings,
                                commitment_cost=commitment_cost,
                                hidden_dims_enc=vq_hidden_dims_enc,
                                hidden_dims_dec=vq_hidden_dims_dec,
                                image_size=vae_image_size
                               ).to(device)
            vae_latent_channels_for_dit = vq_embedding_dim # For VQVAE, DiT channel is embedding_dim
            # The vae_latent_spatial_dim for DiT comes from the generation form, must match DiT training.
            vae_latent_spatial_dim_for_dit = config['vae_latent_spatial_dim']

        else: # Standard VAE
            vae_flat_latent_dim = vae_train_config_from_cache.get('latent_dim', config.get('vae_flat_latent_dim', 256))
            # Standard VAE in this project doesn't take hidden_dims in constructor in the same way as VQVAE
            # So, we don't try to pass them unless the VAE class is updated to accept them.
            autoencoder = VAE(input_channels=3,
                              latent_dim=vae_flat_latent_dim,
                              image_size=vae_image_size
                             ).to(device)
            # For DiT, VAE latent is reshaped. Channels and Spatial Dim from Gen Form.
            vae_latent_channels_for_dit = config['vae_latent_channels']
            vae_latent_spatial_dim_for_dit = config['vae_latent_spatial_dim']

        autoencoder.load_state_dict(torch.load(vae_model_path, map_location=device))
        autoencoder.eval()
        print(f"Loaded {'VQVAE' if is_vqvae else 'VAE'} model from {vae_model_path} (Image Size: {vae_image_size}, Latent Channels for DiT: {vae_latent_channels_for_dit}, Spatial Dim for DiT: {vae_latent_spatial_dim_for_dit})")

        # --- Load DiT Model ---
        dit_model = DiT(
            latent_shape=(vae_latent_channels_for_dit, vae_latent_spatial_dim_for_dit, vae_latent_spatial_dim_for_dit),
            patch_size=config['dit_patch_size'],
            in_channels=vae_latent_channels_for_dit,
            hidden_size=config['dit_hidden_size'],
            depth=config['dit_depth'],
            num_heads=config['dit_num_heads']
        ).to(device)
        dit_model.load_state_dict(torch.load(config['dit_model_path'], map_location=device))
        dit_model.eval()
        print(f"Loaded DiT model from {config['dit_model_path']}")

        # --- DDPM Params ---
        timesteps = config['ddpm_timesteps']
        ddpm_params = get_ddpm_params("linear", timesteps, device)

        # --- Generate ---
        num_samples = config['num_samples']
        latent_sample_shape_for_dit = (num_samples, vae_latent_channels_for_dit, vae_latent_spatial_dim_for_dit, vae_latent_spatial_dim_for_dit)

        vae_decoder_fn = autoencoder.decode
        # VAE might have been trained with bfloat16, p_sample_loop needs to know for its internal autocast
        vae_decode_uses_bfloat16 = False
        if use_bfloat16: # Only relevant if main generation uses bfloat16
            try:
                if next(autoencoder.parameters()).dtype == torch.bfloat16:
                    vae_decode_uses_bfloat16 = True
            except StopIteration: pass # No parameters
        
        # Determine if VAE latents are flat for p_sample_loop logic during decoding
        # This must match how the VAE decoder expects its input *after* DiT produces spatial latents
        # Standard VAE decoder usually takes flat latents. VQVAE decoder takes spatial.
        vae_latent_is_flat_for_decode = (not is_vqvae) and vae_train_config_from_cache.get('vae_latent_is_flat', True)


        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16): # For DiT model itself
            generated_output, _ = p_sample_loop(
                model_dit=dit_model, 
                shape=latent_sample_shape_for_dit, 
                timesteps=timesteps,
                ddpm_params=ddpm_params, 
                device=device, 
                vae_decoder=vae_decoder_fn,
                vae_latent_is_flat=vae_latent_is_flat_for_decode, # How VAE decoder expects its input
                vae_decode_uses_bfloat16=vae_decode_uses_bfloat16 # If VAE itself uses bfloat16
            )

        generated_images_unnorm = (generated_output.clamp(-1, 1) + 1) / 2

        base64_images = []
        for i in range(num_samples):
            img_tensor = generated_images_unnorm[i].cpu()
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
        if 'generation' in training_threads:
            del training_threads['generation']


if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    os.makedirs("results/vae_samples", exist_ok=True)
    os.makedirs("results/dit_samples", exist_ok=True)
    # os.makedirs("cache", exist_ok=True) # Not explicitly used for file cache in this version

    app.run(debug=True, host='0.0.0.0', port=5000)
