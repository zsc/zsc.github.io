import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn # Added for BCEWithLogitsLoss
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import time
from tqdm import tqdm # Ensure tqdm is imported globally

from data_loader import get_dataloader, IMAGE_SIZE
from models.vae import VAE, VQVAE, Discriminator # To load trained VAE/VQVAE and Discriminator
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
        "alphas_cumprod": alphas_cumprod.to(device), # Will be used for x0 prediction
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
    sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t)) # Corresponds to 1/sqrt(alpha_t)
    
    predicted_noise = model_dit(x_t, t_tensor)
    
    # Equation 11 from DDPM paper:
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
    img = torch.randn(shape, device=device) # Start with random noise in latent space
    # imgs_intermediate = [] # To store intermediate latents if needed

    for i in tqdm(reversed(range(0, timesteps)), desc="DiT Sampling loop", total=timesteps, leave=False):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model_dit, img, t_tensor, i, ddpm_params)
        # if i % (timesteps//10) == 0 or i < 10 :
        #     imgs_intermediate.append(img.cpu())
    
    # img is now the predicted x_0 latent
    if vae_decoder:
        final_latents = img
        with torch.no_grad():
            if vae_latent_is_flat:
                final_latents = final_latents.view(final_latents.size(0), -1)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_setting_for_vae):
                img = vae_decoder(final_latents) # Decode to pixel space
    
    return img, None # Second return can be intermediate images/latents if needed


def train_dit_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for DiT training")

    vae_model_path = config['vae_model_path']
    if not os.path.exists(vae_model_path):
        print(f"VAE model path {vae_model_path} not found. Please train VAE first.")
        return None

    is_vqvae = "VQVAE" in vae_model_path.upper() # Check VQVAE in path name (case-insensitive)
    
    try:
        vae_state_dict = torch.load(vae_model_path, map_location=device)
    except AttributeError: 
        vae_state_dict = torch.load(vae_model_path, map_location=device, weights_only=True)


    vae_latent_channels = config.get('vae_latent_channels', 16)
    vae_latent_spatial_dim = config.get('vae_latent_spatial_dim', 12)
    vae_flat_latent_dim = config.get('vae_flat_latent_dim', None)

    if is_vqvae:
        vq_embedding_dim = config.get('vq_embedding_dim_for_dit', 64) # This should match the loaded VQVAE
        vq_num_embeddings = config.get('vq_num_embeddings_for_dit', 128) # This too
        
        # Instantiate to infer shape AND to load weights
        # Ensure VQVAE's hidden_dims_enc/dec match the trained model for correct shape inference & loading.
        # These should be part of the config that trained the VQVAE.
        vq_hidden_dims_enc = config.get('vq_hidden_dims_enc', [128, vq_embedding_dim])
        vq_hidden_dims_dec = config.get('vq_hidden_dims_dec', [vq_embedding_dim // 2]) # Example, match actual VQVAE

        temp_vqvae_for_shape = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                                     num_embeddings=vq_num_embeddings,
                                     hidden_dims_enc=vq_hidden_dims_enc,
                                     hidden_dims_dec=vq_hidden_dims_dec, # Ensure these match
                                     image_size=IMAGE_SIZE 
                                     ).to(device)
        dummy_vq_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad():
            z_e_dummy = temp_vqvae_for_shape.encoder(dummy_vq_input) # encoded before quantization
        _, current_vq_channels, vae_latent_spatial_dim, _ = z_e_dummy.shape
        # For VQVAE, the channels fed to DiT are usually the embedding_dim after quantization.
        # If DiT works on pre-quantized latents z_e, then channels are z_e.shape[1].
        # If DiT works on post-quantized latents z_q, then channels are embedding_dim.
        # The provided code uses autoencoder.encode which returns z_q for VQVAE.
        vae_latent_channels = vq_embedding_dim 
        del temp_vqvae_for_shape, z_e_dummy, dummy_vq_input
        print(f"Inferred VQVAE latent spatial dim: {vae_latent_spatial_dim} (for H' and W'). Latent channels (embedding_dim): {vae_latent_channels}")

        autoencoder = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                            num_embeddings=vq_num_embeddings,
                            commitment_cost=config.get('vq_commitment_cost_for_dit', 0.25), # Param for VQVAE init
                            hidden_dims_enc=vq_hidden_dims_enc, # Must match loaded model
                            hidden_dims_dec=vq_hidden_dims_dec, # Must match loaded model
                            image_size=IMAGE_SIZE
                           ).to(device)
        autoencoder.load_state_dict(vae_state_dict)
        print(f"Loaded VQVAE model from {vae_model_path}")
    else: # Standard VAE
        if vae_flat_latent_dim is None: # Infer if not provided
            if 'encoder.fc_mu.bias' in vae_state_dict:
                vae_flat_latent_dim = vae_state_dict['encoder.fc_mu.bias'].shape[0]
            elif 'decoder.decoder_input_fc.weight' in vae_state_dict: # (out_features, latent_dim)
                vae_flat_latent_dim = vae_state_dict['decoder.decoder_input_fc.weight'].shape[1]
            else:
                raise ValueError("Cannot infer vae_flat_latent_dim from VAE state_dict. Please provide it in config.")
            print(f"Inferred VAE flat latent_dim: {vae_flat_latent_dim}")
        
        autoencoder = VAE(input_channels=3, latent_dim=vae_flat_latent_dim, image_size=IMAGE_SIZE).to(device)
        autoencoder.load_state_dict(vae_state_dict)
        print(f"Loaded VAE model from {vae_model_path}")

        if config.get('vae_latent_is_flat', True): # DiT expects spatial latents
            total_elements = vae_latent_channels * vae_latent_spatial_dim * vae_latent_spatial_dim
            if vae_flat_latent_dim != total_elements:
                raise ValueError(f"Product of vae_latent_channels ({vae_latent_channels}), "
                                 f"vae_latent_spatial_dim^2 ({vae_latent_spatial_dim}^2) = {total_elements} "
                                 f"does not match vae_flat_latent_dim ({vae_flat_latent_dim}). Adjust config.")
            print(f"DiT will operate on latents reshaped from {vae_flat_latent_dim} to "
                  f"({vae_latent_channels}, {vae_latent_spatial_dim}, {vae_latent_spatial_dim})")

    autoencoder.eval() # VAE is fixed
    
    # --- GAN Discriminator Setup (if used) ---
    discriminator = None
    adversarial_loss_fn = None
    if config.get('use_gan_for_dit', False):
        disc_path = config.get('vae_discriminator_path')
        if not disc_path:
            raise ValueError("vae_discriminator_path must be provided if use_gan_for_dit is True.")
        if not os.path.exists(disc_path):
            raise FileNotFoundError(f"Discriminator model path {disc_path} not found.")
        
        discriminator = Discriminator(input_channels=3, image_size=IMAGE_SIZE).to(device)
        try:
            disc_state_dict = torch.load(disc_path, map_location=device)
        except AttributeError:
            disc_state_dict = torch.load(disc_path, map_location=device, weights_only=True)
        discriminator.load_state_dict(disc_state_dict)
        discriminator.eval() # Discriminator is fixed
        adversarial_loss_fn = nn.BCEWithLogitsLoss()
        print(f"Loaded VAE's Discriminator from {disc_path} for DiT GAN loss.")

    # --- DiT Model ---
    dit_model = DiT(
        latent_shape=(vae_latent_channels, vae_latent_spatial_dim, vae_latent_spatial_dim),
        patch_size=config['dit_patch_size'],
        in_channels=vae_latent_channels,
        hidden_size=config['dit_hidden_size'],
        depth=config['dit_depth'],
        num_heads=config['dit_num_heads']
    ).to(device)
    
    optimizer = optim.AdamW(dit_model.parameters(), lr=config['lr'])

    run_name_parts = [
        "DiT",
        "VQVAE" if is_vqvae else "VAE",
        f"lsd{vae_latent_spatial_dim}", f"lc{vae_latent_channels}",
        f"ps{config['dit_patch_size']}", f"hs{config['dit_hidden_size']}",
        f"d{config['dit_depth']}", f"lr{config['lr']}",
        f"bs{config['batch_size']}", f"epochs{config['epochs']}"
    ]
    if config.get('use_gan_for_dit', False):
        run_name_parts.append("GAN")
        run_name_parts.append(f"ganW{config.get('gan_loss_weight_dit', 0.01)}")
    run_name = "_".join(run_name_parts)
    
    writer = SummaryWriter(log_dir=os.path.join("runs", "dit", run_name))

    timesteps = config['ddpm_timesteps']
    ddpm_params = get_ddpm_params(config['ddpm_schedule'], timesteps, device)
    dataloader = get_dataloader(batch_size=config['batch_size'], data_limit=config.get('data_limit'))

    use_bfloat16_dit = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bfloat16_dit:
        print("Using bfloat16 for DiT model training.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_bfloat16_dit)

    # Determine autocast for fixed VAE/Discriminator based on their parameters
    autocast_for_fixed_models = False
    if autoencoder is not None:
        try:
            if next(iter(autoencoder.parameters())).dtype == torch.bfloat16:
                autocast_for_fixed_models = True
                print("INFO: VAE parameters appear to be bfloat16. Enabling autocast for its usage.")
        except StopIteration: pass
    if discriminator is not None and not autocast_for_fixed_models: # If VAE not bf16, check D
         try:
            if next(iter(discriminator.parameters())).dtype == torch.bfloat16:
                autocast_for_fixed_models = True # If D is bf16, then fixed models need autocast
                print("INFO: Discriminator parameters appear to be bfloat16. Enabling autocast for its usage.")
         except StopIteration: pass


    print(f"Starting DiT training ({run_name})...")
    global_step = 0
    for epoch in range(config['epochs']):
        dit_model.train()
        epoch_total_loss = 0
        epoch_mse_loss = 0
        epoch_gan_loss_g = 0
        start_time = time.time()

        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad(): # VAE encoding is fixed
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_for_fixed_models):
                    if is_vqvae:
                        # VQVAE.encode returns z_q (quantized), vq_loss, perplexity, encodings
                        clean_latents, _, _, _ = autoencoder.encode(real_images) 
                    else: # Standard VAE
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
            
            # --- DiT forward and loss calculation ---
            current_total_loss = None
            current_mse_loss = None
            current_gan_loss_g = torch.tensor(0.0, device=device) # Default for logging

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_dit):
                predicted_noise = dit_model(noisy_latents, t)
                loss_mse = F.mse_loss(predicted_noise, noise)
                current_mse_loss = loss_mse
                current_total_loss = loss_mse

                if discriminator is not None and adversarial_loss_fn is not None and config.get('use_gan_for_dit', False):
                    # Predict x0 based on DiT's noise prediction
                    # x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
                    sqrt_alphas_cumprod_t = ddpm_params["alphas_cumprod"][t].sqrt()[:, None, None, None]
                    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t, None, None, None]
                    
                    # Avoid division by zero if sqrt_alphas_cumprod_t can be zero (e.g. at t near T for some schedules)
                    # For linear schedule, alphas_cumprod[T-1] is small but non-zero.
                    pred_x0_latents = (noisy_latents - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / (sqrt_alphas_cumprod_t + 1e-8) # Add epsilon

                    # Prepare latents for VAE decoder
                    pred_x0_latents_for_decode = pred_x0_latents
                    if not is_vqvae and config.get('vae_latent_is_flat', True):
                        pred_x0_latents_for_decode = pred_x0_latents.view(pred_x0_latents.size(0), -1)
                    
                    reconstructed_images_from_dit = None
                    with torch.no_grad(): # VAE decoder is fixed
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_for_fixed_models):
                             reconstructed_images_from_dit = autoencoder.decode(pred_x0_latents_for_decode)
                    
                    # Discriminator forward pass (fixed D, but gradients flow for DiT)
                    disc_output_on_g_fake = None
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_for_fixed_models):
                         disc_output_on_g_fake = discriminator(reconstructed_images_from_dit)
                    
                    loss_g_adv = adversarial_loss_fn(disc_output_on_g_fake, torch.ones_like(disc_output_on_g_fake).to(device))
                    current_gan_loss_g = loss_g_adv
                    current_total_loss = current_total_loss + config.get('gan_loss_weight_dit', 0.01) * loss_g_adv
            
            # --- Backward and Optimize ---
            if use_bfloat16_dit:
                scaler.scale(current_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                current_total_loss.backward()
                optimizer.step()

            epoch_total_loss += current_total_loss.item() * real_images.size(0)
            epoch_mse_loss += current_mse_loss.item() * real_images.size(0)
            if config.get('use_gan_for_dit', False):
                 epoch_gan_loss_g += current_gan_loss_g.item() * real_images.size(0)
            
            global_step += 1

            if batch_idx % config.get('log_interval', 100) == 0:
                log_msg = (f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(dataloader)}, "
                           f"Total Loss: {current_total_loss.item():.4f}, MSE Loss: {current_mse_loss.item():.4f}")
                writer.add_scalar('DiT/batch_total_loss', current_total_loss.item(), global_step)
                writer.add_scalar('DiT/batch_mse_loss', current_mse_loss.item(), global_step)
                if config.get('use_gan_for_dit', False):
                    log_msg += f", G_Adv_Loss: {current_gan_loss_g.item():.4f}"
                    writer.add_scalar('DiT/batch_g_adv_loss', current_gan_loss_g.item(), global_step)
                print(log_msg)
        
        epoch_time = time.time() - start_time
        num_samples_epoch = len(dataloader.dataset) # Assumes dataloader.dataset reflects data_limit

        avg_epoch_total_loss = epoch_total_loss / num_samples_epoch
        avg_epoch_mse_loss = epoch_mse_loss / num_samples_epoch
        
        writer.add_scalar('DiT/epoch_total_loss', avg_epoch_total_loss, epoch + 1)
        writer.add_scalar('DiT/epoch_mse_loss', avg_epoch_mse_loss, epoch + 1)
        
        summary_msg = (f"Epoch {epoch+1} finished. Avg Total Loss: {avg_epoch_total_loss:.4f}, "
                       f"Avg MSE Loss: {avg_epoch_mse_loss:.4f}")
        
        if config.get('use_gan_for_dit', False):
            avg_epoch_gan_loss_g = epoch_gan_loss_g / num_samples_epoch
            writer.add_scalar('DiT/epoch_g_adv_loss', avg_epoch_gan_loss_g, epoch + 1)
            summary_msg += f", Avg G_Adv Loss: {avg_epoch_gan_loss_g:.4f}"
            
        summary_msg += f", Time: {epoch_time:.2f}s"
        print(summary_msg)

        if (epoch + 1) % config.get('sample_interval', 10) == 0:
            dit_model.eval()
            with torch.no_grad():
                num_samples_to_gen = config.get('num_samples_gen', 8)
                latent_sample_shape = (num_samples_to_gen, vae_latent_channels, 
                                       vae_latent_spatial_dim, vae_latent_spatial_dim)
                
                vae_decoder_fn = autoencoder.decode
                current_vae_is_flat_for_sampling = (not is_vqvae) and config.get('vae_latent_is_flat', True)

                # DiT model sampling loop with its own autocast
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_dit):
                    generated_images_final, _ = p_sample_loop(
                        dit_model, latent_sample_shape, timesteps, 
                        ddpm_params, device, 
                        vae_decoder=vae_decoder_fn,
                        vae_latent_is_flat=current_vae_is_flat_for_sampling,
                        autocast_setting_for_vae=autocast_for_fixed_models # Use inferred for VAE
                    )
                
                generated_images_display = (generated_images_final.clamp(-1, 1) + 1) / 2 # Normalize to [0,1]
                grid = make_grid(generated_images_display.cpu())
                writer.add_image('DiT/generated_samples', grid, epoch + 1)
                save_image(grid, f"results/dit_samples/{run_name}_epoch_{epoch+1}.png")
            dit_model.train()

    model_save_path = os.path.join("checkpoints", f"{run_name}_final.pth")
    torch.save(dit_model.state_dict(), model_save_path)
    print(f"DiT training finished. Model saved to {model_save_path}")
    writer.close()
    return model_save_path

if __name__ == "__main__":
    print("Testing DiT training script...")
    
    # --- VAE (and Discriminator) Setup for DiT Test ---
    # These parameters should match a VAE previously trained or trained now
    # For this test, DiT will use a VAE that was trained with GAN.
    vae_flat_latent_dim_for_test = 64  # Example: 16 * 2 * 2 = 64
    vae_latent_channels_for_test = 16 
    vae_latent_spatial_dim_for_test = 2 # So latent is 16x2x2
    
    # Config for training a VAE+GAN model (if it doesn't exist)
    # This VAE will serve as the autoencoder for DiT.
    # Its discriminator will be used for DiT's GAN loss.
    dummy_vae_gan_config = {
        'lr': 1e-4, 'batch_size': 4, 'epochs': 1, # Minimal epochs for test
        'latent_dim': vae_flat_latent_dim_for_test,
        'log_interval': 1, 'sample_interval': 1, 'data_limit': 16, # Minimal data
        'use_bfloat16': False, 'use_vq': False,
        'kld_beta': 1.0,
        'use_gan': True, # CRITICAL: Ensure VAE is trained with GAN
        'gan_loss_weight': 0.01,
        'lr_d': 1e-4, # Discriminator LR for VAE training
        # 'image_size': IMAGE_SIZE # Implicitly uses global IMAGE_SIZE from data_loader
    }

    # Construct VAE+GAN model run name based on train_vae.py logic
    # model_type_full_vae = "VAE_GAN"
    vae_run_name_parts = [
        "VAE_GAN", # model_type_full
        f"lr{dummy_vae_gan_config['lr']}",
        f"bs{dummy_vae_gan_config['batch_size']}",
        f"epochs{dummy_vae_gan_config['epochs']}",
        f"ld{dummy_vae_gan_config['latent_dim']}",
        f"ganW{dummy_vae_gan_config.get('gan_loss_weight', 1.0)}",
        f"lrD{dummy_vae_gan_config.get('lr_d', dummy_vae_gan_config['lr'])}"
    ]
    dummy_vae_gan_run_name = "_".join(vae_run_name_parts)

    test_vae_generator_path = os.path.join("checkpoints", f"{dummy_vae_gan_run_name}_generator_final.pth")
    test_vae_discriminator_path = os.path.join("checkpoints", f"{dummy_vae_gan_run_name}_discriminator_final.pth")

    if not os.path.exists(test_vae_generator_path) or not os.path.exists(test_vae_discriminator_path):
        print(f"Dummy VAE+GAN generator or discriminator not found. Training for DiT test...")
        try:
            from training.train_vae import train_vae_model # Assumes train_vae.py is in training/
            # This will create both generator and discriminator .pth files
            train_vae_model(dummy_vae_gan_config) 
            print("Dummy VAE+GAN training completed.")
        except ImportError:
            print("Could not import train_vae_model from training.train_vae.")
            print(f"Please ensure VAE generator at: {test_vae_generator_path}")
            print(f"And VAE discriminator at: {test_vae_discriminator_path} exist, or run VAE+GAN training.")
            exit(1) 
        except Exception as e:
            print(f"Error training dummy VAE+GAN: {e}")
            exit(1)
    else:
        print(f"Using existing VAE generator: {test_vae_generator_path}")
        print(f"Using existing VAE discriminator: {test_vae_discriminator_path}")

    # --- DiT Test Configuration (using the VAE+GAN above) ---
    test_config_dit_gan = {
        'vae_model_path': test_vae_generator_path,
        'vae_flat_latent_dim': vae_flat_latent_dim_for_test, # For standard VAE, to reshape
        'vae_latent_is_flat': True, # Standard VAE output is flat
        'vae_latent_channels': vae_latent_channels_for_test, # Target C for DiT
        'vae_latent_spatial_dim': vae_latent_spatial_dim_for_test, # Target H, W for DiT   
        
        'lr': 1e-4, 'batch_size': 4, 'epochs': 1, 'data_limit': 16,
        'ddpm_timesteps': 20, # Reduced for faster test
        'ddpm_schedule': 'linear',
        
        'dit_patch_size': 1, # Latent spatial dim is 2x2, patch size 1 means 4 patches
        'dit_hidden_size': 32, # Small for test
        'dit_depth': 1,        # Small for test
        'dit_num_heads': 2,    # Small for test
        
        'log_interval': 1, 'sample_interval': 1, 'use_bfloat16': False,
        'num_samples_gen': 2, # Generate fewer samples for test

        'use_gan_for_dit': True, # Enable GAN loss for DiT
        'vae_discriminator_path': test_vae_discriminator_path,
        'gan_loss_weight_dit': 0.01, # Weight for DiT's GAN loss component
    }
    
    print("\n--- Training DiT with GAN loss (test) ---")
    dit_model_path = train_dit_model(test_config_dit_gan)
    if dit_model_path:
        print(f"DiT (with GAN loss) test training completed. Model saved to {dit_model_path}")
    else:
        print("DiT (with GAN loss) test training failed or was skipped.")

    print("\nDiT training script tests finished.")
    print("Check 'runs/dit/' for TensorBoard logs and 'checkpoints/' for models.")
    print("Check 'results/dit_samples/' for image samples.")
