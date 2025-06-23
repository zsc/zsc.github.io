import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time
from tqdm import tqdm # Ensure tqdm is imported globally

from data_loader import get_dataloader, IMAGE_SIZE
from models.vae import VAE, VQVAE # To load trained VAE/VQVAE
from models.dit import DiT

# Ensure checkpoints and tensorboard run directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("results/dit_samples", exist_ok=True)


# --- DDPM Utilities ---
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_ddpm_params(schedule_name, timesteps, device):
    if schedule_name == "linear":
        betas = linear_beta_schedule(timesteps)
    else: # Add cosine or other schedules if needed
        raise NotImplementedError(f"schedule {schedule_name} not implemented")

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "betas": betas.to(device),
        "alphas_cumprod": alphas_cumprod.to(device),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.to(device),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod.to(device),
        "posterior_variance": posterior_variance.to(device)
    }

def q_sample(x_start, t, ddpm_params, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"][t, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t, None, None, None]
    
    noisy_latents = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_latents

@torch.no_grad()
def p_sample(model_dit, x_t, t_tensor, t_idx, ddpm_params):
    betas_t = ddpm_params["betas"][t_idx, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t_idx, None, None, None]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))
    
    predicted_noise = model_dit(x_t, t_tensor)
    
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_idx == 0:
        return model_mean
    else:
        posterior_variance_t = ddpm_params["posterior_variance"][t_idx, None, None, None]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model_dit, shape, timesteps, ddpm_params, device, 
                  vae_decoder=None, vae_latent_is_flat=False, autocast_setting_for_vae=False):
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc="DiT Sampling loop", total=timesteps):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model_dit, img, t_tensor, i, ddpm_params)
        if i % (timesteps//10) == 0 or i < 10 :
            imgs.append(img.cpu())
    
    if vae_decoder:
        with torch.no_grad():
            # If VAE decoder expects a flat latent vector (standard VAE), reshape it.
            # VQVAE decoder typically expects spatial latents (B, embedding_dim, H', W').
            if vae_latent_is_flat:
                img = img.view(img.size(0), -1) # Flatten to (B, num_features)
            
            # Autocast for VAE decoder if its parameters suggest it (e.g., trained with bfloat16)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_setting_for_vae):
                img = vae_decoder(img) # img is the final x_0 latent, now becomes decoded image
    
    return img, imgs


def train_dit_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for DiT training")

    vae_model_path = config['vae_model_path']
    if not os.path.exists(vae_model_path):
        print(f"VAE model path {vae_model_path} not found. Please train VAE first.")
        return None

    is_vqvae = "VQVAE" in vae_model_path 
    
    # Load VAE state dict with weights_only=True for security if PyTorch version supports it well.
    # For now, following the original trace, keeping weights_only=False (default).
    try:
        vae_state_dict = torch.load(vae_model_path, map_location=device) # Original
    except AttributeError: # Fallback for older torch or specific pickle issues
        vae_state_dict = torch.load(vae_model_path, map_location=device, weights_only=True)


    vae_latent_channels = config.get('vae_latent_channels', 16)
    vae_latent_spatial_dim = config.get('vae_latent_spatial_dim', 12)
    vae_flat_latent_dim = config.get('vae_flat_latent_dim', None)

    if is_vqvae:
        vq_embedding_dim = config.get('vq_embedding_dim_for_dit', 64)
        vq_num_embeddings = config.get('vq_num_embeddings_for_dit', 128)
        
        # Infer spatial dim for VQVAE
        # Note: Ensure VQVAE's hidden_dims_enc match the trained model for correct shape inference.
        # This part requires the VQVAE architecture to be consistent.
        temp_vqvae_for_shape = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                                     num_embeddings=vq_num_embeddings,
                                     hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, vq_embedding_dim]),
                                     hidden_dims_dec=config.get('vq_hidden_dims_dec', [vq_embedding_dim, 128]) # Ensure these match
                                     ).to(device) # Needs to be on device for dummy pass
        dummy_vq_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad(): # Ensure no grads for dummy pass
            z_e_dummy = temp_vqvae_for_shape.encoder(dummy_vq_input)
        _, _, vae_latent_spatial_dim, _ = z_e_dummy.shape
        del temp_vqvae_for_shape, z_e_dummy, dummy_vq_input
        print(f"Inferred VQVAE latent spatial dim: {vae_latent_spatial_dim} (for H' and W')")

        vae_latent_channels = vq_embedding_dim # For VQVAE, latent channels are embedding_dim

        autoencoder = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                            num_embeddings=vq_num_embeddings,
                            commitment_cost=config.get('vq_commitment_cost_for_dit', 0.25),
                            hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, vq_embedding_dim]),
                            hidden_dims_dec=config.get('vq_hidden_dims_dec', [vq_embedding_dim, 128])
                           ).to(device)
        autoencoder.load_state_dict(vae_state_dict)
        print(f"Loaded VQVAE model from {vae_model_path}")
    else: # Standard VAE
        if vae_flat_latent_dim is None:
            if 'encoder.fc_mu.bias' in vae_state_dict:
                vae_flat_latent_dim = vae_state_dict['encoder.fc_mu.bias'].shape[0]
            elif 'encoder.fc_mu.weight' in vae_state_dict:
                 vae_flat_latent_dim = vae_state_dict['encoder.fc_mu.weight'].shape[0]
            else:
                # Try inferring from decoder_input_fc.weight if fc_mu is not present (e.g. different VAE structure)
                # This depends on VAE architecture details. For the VAE in models/vae.py:
                # decoder.decoder_input_fc.weight has shape (output_features, latent_dim)
                if 'decoder.decoder_input_fc.weight' in vae_state_dict:
                    vae_flat_latent_dim = vae_state_dict['decoder.decoder_input_fc.weight'].shape[1]
                else:
                    raise ValueError("Cannot infer vae_flat_latent_dim from VAE state_dict. Please provide it in config.")
            print(f"Inferred VAE flat latent_dim: {vae_flat_latent_dim}")
        
        # Ensure VAE class init matches the loaded state_dict structure (latent_dim, image_size, hidden_dims etc.)
        autoencoder = VAE(input_channels=3, latent_dim=vae_flat_latent_dim).to(device)
        autoencoder.load_state_dict(vae_state_dict)
        print(f"Loaded VAE model from {vae_model_path}")

        if config.get('vae_latent_is_flat', True):
            total_elements = vae_latent_channels * vae_latent_spatial_dim * vae_latent_spatial_dim
            if vae_flat_latent_dim != total_elements:
                raise ValueError(f"Product of vae_latent_channels ({vae_latent_channels}), "
                                 f"vae_latent_spatial_dim^2 ({vae_latent_spatial_dim}^2) "
                                 f"which is {total_elements} does not match vae_flat_latent_dim ({vae_flat_latent_dim}). "
                                 "Adjust config for reshaping.")
            print(f"DiT will operate on latents reshaped from {vae_flat_latent_dim} to "
                  f"({vae_latent_channels}, {vae_latent_spatial_dim}, {vae_latent_spatial_dim})")

    autoencoder.eval()
    
    dit_model = DiT(
        latent_shape=(vae_latent_channels, vae_latent_spatial_dim, vae_latent_spatial_dim),
        patch_size=config['dit_patch_size'],
        in_channels=vae_latent_channels,
        hidden_size=config['dit_hidden_size'],
        depth=config['dit_depth'],
        num_heads=config['dit_num_heads']
    ).to(device)
    
    optimizer = optim.AdamW(dit_model.parameters(), lr=config['lr'])

    run_name = f"DiT_VAE_lsd{vae_latent_spatial_dim}_lc{vae_latent_channels}_ps{config['dit_patch_size']}_hs{config['dit_hidden_size']}_d{config['dit_depth']}_lr{config['lr']}_bs{config['batch_size']}_epochs{config['epochs']}"
    if is_vqvae:
        run_name = run_name.replace("VAE", "VQVAE")
    writer = SummaryWriter(log_dir=os.path.join("runs", "dit", run_name))

    timesteps = config['ddpm_timesteps']
    ddpm_params = get_ddpm_params(config['ddpm_schedule'], timesteps, device)
    dataloader = get_dataloader(batch_size=config['batch_size'], data_limit=config.get('data_limit'))

    use_bfloat16 = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bfloat16:
        print("Using bfloat16 for DiT training.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_bfloat16) # Initialize scaler, enabled flag controls it.

    print("Starting DiT training...")
    global_step = 0
    for epoch in range(config['epochs']):
        dit_model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            optimizer.zero_grad(set_to_none=True) # More memory efficient

            # Determine VAE autocast setting for encoding
            autocast_vae_encode = False
            if autoencoder is not None: # Should always be true here
                try:
                    if next(autoencoder.parameters()).dtype == torch.bfloat16:
                        autocast_vae_encode = True
                except StopIteration: # Should not happen if autoencoder is loaded
                    pass

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_vae_encode):
                    if is_vqvae:
                        clean_latents, _ = autoencoder.encode(real_images)
                    else:
                        mu, logvar = autoencoder.encode(real_images)
                        clean_latents = autoencoder.reparameterize(mu, logvar)
                        if config.get('vae_latent_is_flat', True):
                            clean_latents = clean_latents.view(real_images.size(0), 
                                                               vae_latent_channels, 
                                                               vae_latent_spatial_dim, 
                                                               vae_latent_spatial_dim)
            
            t = torch.randint(0, timesteps, (real_images.size(0),), device=device).long()
            noise = torch.randn_like(clean_latents)
            noisy_latents = q_sample(clean_latents, t, ddpm_params, noise=noise)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                predicted_noise = dit_model(noisy_latents, t)
                loss = F.mse_loss(predicted_noise, noise)

            if use_bfloat16: # If using bfloat16 (or fp16), GradScaler is recommended
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # Standard full-precision training
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * real_images.size(0)
            global_step += 1

            if batch_idx % config.get('log_interval', 100) == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                writer.add_scalar('DiT/batch_loss', loss.item(), global_step)
        
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(dataloader.dataset) # Adjusted for data_limit if used
        writer.add_scalar('DiT/epoch_loss', avg_epoch_loss, epoch + 1)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        if (epoch + 1) % config.get('sample_interval', 10) == 0:
            dit_model.eval()
            with torch.no_grad():
                num_samples_to_gen = 5
                latent_sample_shape = (num_samples_to_gen, vae_latent_channels, 
                                       vae_latent_spatial_dim, vae_latent_spatial_dim)
                
                vae_decoder_fn = autoencoder.decode
                
                # Determine VAE autocast setting for decoding during sampling
                autocast_setting_for_vae_sampling = False
                if autoencoder is not None:
                    try:
                        if next(autoencoder.parameters()).dtype == torch.bfloat16:
                            autocast_setting_for_vae_sampling = True
                    except StopIteration:
                        pass
                
                # Determine if VAE latents are flat for p_sample_loop logic
                current_vae_is_flat_for_sampling = (not is_vqvae) and config.get('vae_latent_is_flat', True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16): # For DiT model
                    generated_images_final, _ = p_sample_loop(
                        dit_model, latent_sample_shape, timesteps, 
                        ddpm_params, device, 
                        vae_decoder=vae_decoder_fn,
                        vae_latent_is_flat=current_vae_is_flat_for_sampling,
                        autocast_setting_for_vae=autocast_setting_for_vae_sampling
                    )
                
                generated_images = (generated_images_final.clamp(-1, 1) + 1) / 2
                grid = make_grid(generated_images)
                writer.add_image('DiT/generated_samples', grid, epoch + 1)
                save_image(grid, f"results/dit_samples/{run_name}_epoch_{epoch+1}.png") # Use run_name for unique sample filenames
            dit_model.train()

    model_save_path = os.path.join("checkpoints", f"{run_name}_final.pth")
    torch.save(dit_model.state_dict(), model_save_path)
    print(f"DiT training finished. Model saved to {model_save_path}")
    writer.close()
    return model_save_path

if __name__ == "__main__":
    # from tqdm import tqdm # Already imported globally
    print("Testing DiT training script...")
    
    vae_flat_latent_dim_for_test = 64
    vae_latent_channels_for_test = 16 
    vae_latent_spatial_dim_for_test = 2
    
    dummy_vae_config = {
        'lr': 1e-5, 'batch_size': 4, 'epochs': 1, 'latent_dim': vae_flat_latent_dim_for_test,
        'log_interval': 1, 'sample_interval': 1, 'data_limit': 16, 'use_bfloat16': False, 'use_vq': False,
        'image_size': IMAGE_SIZE # Ensure VAE is trained with consistent image_size
    }
    dummy_vae_run_name = f"VAE_img{IMAGE_SIZE}_lr{dummy_vae_config['lr']}_bs{dummy_vae_config['batch_size']}_epochs{dummy_vae_config['epochs']}_ld{dummy_vae_config['latent_dim']}"
    test_vae_model_path = os.path.join("checkpoints", f"{dummy_vae_run_name}_final.pth")

    if not os.path.exists(test_vae_model_path):
        print(f"Dummy VAE model not found at {test_vae_model_path}. Training a minimal one for DiT test...")
        # Ensure train_vae_model is available and uses IMAGE_SIZE correctly.
        # This assumes train_vae.py is in training/ and train_vae_model accepts image_size.
        # For simplicity, if train_vae is complex, this step might need manual running or a simpler VAE creation.
        try:
            from training.train_vae import train_vae_model
            train_vae_model(dummy_vae_config) # Pass full config
            print("Dummy VAE trained.")
        except ImportError:
            print("Could not import train_vae_model. Please ensure a VAE model exists or train one manually.")
            print(f"Expected VAE at: {test_vae_model_path}")
            # Exit or skip DiT training if VAE is essential and not found/trainable
            exit(1) 
        except Exception as e:
            print(f"Error training dummy VAE: {e}")
            exit(1)
            
    else:
        print(f"Using existing dummy VAE model: {test_vae_model_path}")

    test_config_dit = {
        'vae_model_path': test_vae_model_path,
        'vae_flat_latent_dim': vae_flat_latent_dim_for_test,
        'vae_latent_is_flat': True,
        'vae_latent_channels': vae_latent_channels_for_test,
        'vae_latent_spatial_dim': vae_latent_spatial_dim_for_test,   
        
        'lr': 1e-4, 'batch_size': 4, 'epochs': 1, 'data_limit': 16,
        'ddpm_timesteps': 50, 'ddpm_schedule': 'linear',
        
        'dit_patch_size': 1, 'dit_hidden_size': 64, 'dit_depth': 1, 'dit_num_heads': 2,
        
        'log_interval': 1, 'sample_interval': 1, 'use_bfloat16': False,
    }
    
    print("\n--- Training DiT (test) ---")
    dit_model_path = train_dit_model(test_config_dit)
    if dit_model_path:
        print(f"DiT test training completed. Model saved to {dit_model_path}")
    else:
        print("DiT test training failed or was skipped.")

    print("\nDiT training script tests finished.")
