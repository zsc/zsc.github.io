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
    
    # For q_posterior -> p_sample
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "betas": betas.to(device),
        "alphas": alphas.to(device), # Will need for x0 prediction if not using direct formula
        "alphas_cumprod": alphas_cumprod.to(device),
        "alphas_cumprod_prev": alphas_cumprod_prev.to(device),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.to(device),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod.to(device),
        "posterior_variance": posterior_variance.to(device)
    }

def q_sample(x_start, t, ddpm_params, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Ensure t is correctly shaped for indexing
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"].gather(0, t).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"].gather(0, t).reshape(-1, 1, 1, 1)
        
    noisy_latents = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_latents

# Helper to predict x0 from x_t and noise (epsilon)
def predict_x0_from_xt_noise(x_t, t, noise, ddpm_params):
    sqrt_alphas_cumprod_t = ddpm_params["sqrt_alphas_cumprod"].gather(0, t).reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"].gather(0, t).reshape(-1, 1, 1, 1)
    # x_0 = (x_t - sqrt(1-alpha_cumprod_t) * eps) / sqrt(alpha_cumprod_t)
    pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    return pred_x0


@torch.no_grad()
def p_sample(model_dit, x_t, t_tensor, t_idx, ddpm_params): # t_tensor is batch of timesteps, t_idx is scalar for indexing params
    betas_t = ddpm_params["betas"][t_idx].reshape(-1, 1, 1, 1) # Ensure correct shape for broadcasting
    sqrt_one_minus_alphas_cumprod_t = ddpm_params["sqrt_one_minus_alphas_cumprod"][t_idx].reshape(-1, 1, 1, 1)
    
    # Calculate 1/sqrt(alpha_t) for the model mean calculation
    # Using .gather for t_tensor which contains batch of timesteps (all same value i in p_sample_loop)
    # However, ddpm_params are already indexed by t_idx (scalar current time)
    sqrt_alphas_t = torch.sqrt(ddpm_params["alphas"][t_idx]).reshape(-1, 1, 1, 1)
    sqrt_recip_alphas_t = 1.0 / sqrt_alphas_t
        
    predicted_noise = model_dit(x_t, t_tensor) # t_tensor is (B,)
    
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_idx == 0:
        return model_mean
    else:
        posterior_variance_t = ddpm_params["posterior_variance"][t_idx].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model_dit, shape, timesteps, ddpm_params, device, 
                  vae_decoder=None, vae_latent_is_flat=False, vae_decode_uses_bfloat16=False):
    img = torch.randn(shape, device=device) # This is x_T
    imgs_progress = [] # To store intermediate generated latents (optional)

    for i in tqdm(reversed(range(0, timesteps)), desc="DiT Sampling loop", total=timesteps, leave=False):
        t_tensor = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model_dit, img, t_tensor, i, ddpm_params) # img becomes x_{t-1}
        if i % (timesteps//10) == 0 or i < 10 or i == timesteps -1 : # Store some progress steps
            imgs_progress.append(img.cpu())
    
    # img is now the final x_0 latent
    if vae_decoder:
        # If VAE decoder expects a flat latent vector (standard VAE), reshape it.
        # VQVAE decoder typically expects spatial latents (B, embedding_dim, H', W').
        if vae_latent_is_flat:
            img_for_decode = img.view(img.size(0), -1) # Flatten to (B, num_features)
        else:
            img_for_decode = img
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=vae_decode_uses_bfloat16):
            img = vae_decoder(img_for_decode) # img is the final x_0 latent, now becomes decoded image
    
    return img, imgs_progress


def train_dit_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for DiT training")

    vae_model_path = config['vae_model_path']
    if not os.path.exists(vae_model_path):
        print(f"VAE model path {vae_model_path} not found. Please train VAE first.")
        return None

    is_vqvae = "VQVAE" in vae_model_path.upper() # More robust check
    
    vae_state_dict = torch.load(vae_model_path, map_location=device)

    vae_latent_channels = config.get('vae_latent_channels', 16)
    vae_latent_spatial_dim = config.get('vae_latent_spatial_dim', 12)
    vae_flat_latent_dim = config.get('vae_flat_latent_dim', None)

    if is_vqvae:
        vq_embedding_dim = config.get('vq_embedding_dim_for_dit', 64) # Should match trained VQVAE
        vq_num_embeddings = config.get('vq_num_embeddings_for_dit', 128) # Should match
        
        # This inference of spatial dim might be tricky if VQVAE architecture varied a lot.
        # It's better if these are known from VAE training config.
        # For now, let's keep it but ideally config should provide these directly if known.
        temp_vqvae_for_shape = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                                     num_embeddings=vq_num_embeddings,
                                     hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, vq_embedding_dim]),
                                     #hidden_dims_dec relevant for full shape consistency but encoder output shape is key
                                     image_size=IMAGE_SIZE # Important for encoder structure
                                     ).to(device) 
        dummy_vq_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        with torch.no_grad(): 
            z_e_dummy = temp_vqvae_for_shape.encoder(dummy_vq_input) # Output before quantization
        _, _, vae_latent_spatial_dim_h, vae_latent_spatial_dim_w = z_e_dummy.shape
        if vae_latent_spatial_dim_h != vae_latent_spatial_dim_w :
            print(f"Warning: VQVAE latent spatial dims are not square: H={vae_latent_spatial_dim_h}, W={vae_latent_spatial_dim_w}. Using H for DiT.")
        vae_latent_spatial_dim = vae_latent_spatial_dim_h # Assuming square for DiT
        del temp_vqvae_for_shape, z_e_dummy, dummy_vq_input
        print(f"Inferred VQVAE latent spatial dim for DiT: {vae_latent_spatial_dim} (H' and W')")
        vae_latent_channels = vq_embedding_dim

        autoencoder = VQVAE(input_channels=3, embedding_dim=vq_embedding_dim, 
                            num_embeddings=vq_num_embeddings,
                            commitment_cost=config.get('vq_commitment_cost_for_dit', 0.25), # From VQVAE training
                            hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, vq_embedding_dim]),
                            hidden_dims_dec=config.get('vq_hidden_dims_dec', [vq_embedding_dim, 128]), # Match VQVAE training
                            image_size=IMAGE_SIZE
                           ).to(device)
        autoencoder.load_state_dict(vae_state_dict)
        print(f"Loaded VQVAE model from {vae_model_path}")
    else: # Standard VAE
        if vae_flat_latent_dim is None: # Infer from state_dict
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
                                 f"does not match vae_flat_latent_dim ({vae_flat_latent_dim}). Adjust config for reshaping.")
            print(f"DiT will operate on latents reshaped from {vae_flat_latent_dim} to "
                  f"({vae_latent_channels}, {vae_latent_spatial_dim}, {vae_latent_spatial_dim})")

    autoencoder.eval() # VAE is frozen
    
    # Determine if VAE parameters are bfloat16
    vae_uses_bfloat16 = False
    try:
        if next(autoencoder.parameters()).dtype == torch.bfloat16:
            vae_uses_bfloat16 = True
            print("VAE model parameters are bfloat16.")
    except StopIteration:
        pass


    dit_model = DiT(
        latent_shape=(vae_latent_channels, vae_latent_spatial_dim, vae_latent_spatial_dim),
        patch_size=config['dit_patch_size'],
        in_channels=vae_latent_channels,
        hidden_size=config['dit_hidden_size'],
        depth=config['dit_depth'],
        num_heads=config['dit_num_heads']
    ).to(device)
    
    optimizer_dit = optim.AdamW(dit_model.parameters(), lr=config['lr'])

    # --- GAN Setup (if used) ---
    use_gan = config.get('use_gan', False)
    discriminator = None
    optimizer_d = None
    adversarial_loss_fn = None
    discriminator_uses_bfloat16 = False

    if use_gan:
        print("DiT training with GAN loss.")
        discriminator_model_path = config.get('discriminator_model_path')
        if not discriminator_model_path or not os.path.exists(discriminator_model_path):
            raise FileNotFoundError(f"Discriminator model path {discriminator_model_path} not found. Required if use_gan=True.")
        
        discriminator = Discriminator(input_channels=3, image_size=IMAGE_SIZE).to(device)
        discriminator.load_state_dict(torch.load(discriminator_model_path, map_location=device))
        print(f"Loaded Discriminator model from {discriminator_model_path}")
        
        optimizer_d = optim.AdamW(discriminator.parameters(), lr=config.get('lr_d', config['lr']))
        adversarial_loss_fn = nn.BCEWithLogitsLoss()
        discriminator.train() # Set to train mode if it's being updated

        try:
            if next(discriminator.parameters()).dtype == torch.bfloat16:
                discriminator_uses_bfloat16 = True
                print("Discriminator model parameters are bfloat16.")
        except StopIteration:
            pass
    # --- End GAN Setup ---

    run_name_parts = [
        f"DiT_{'VQVAE' if is_vqvae else 'VAE'}",
        f"lsd{vae_latent_spatial_dim}",
        f"lc{vae_latent_channels}",
        f"ps{config['dit_patch_size']}",
        f"hs{config['dit_hidden_size']}",
        f"d{config['dit_depth']}",
        f"lr{config['lr']}",
        f"bs{config['batch_size']}",
        f"epochs{config['epochs']}"
    ]
    if use_gan:
        run_name_parts.append(f"GANlw{config.get('gan_loss_weight', 0.01)}")
        run_name_parts.append(f"lrD{config.get('lr_d', config['lr'])}")
    run_name = "_".join(run_name_parts)
    writer = SummaryWriter(log_dir=os.path.join("runs", "dit", run_name))

    timesteps = config['ddpm_timesteps']
    ddpm_params = get_ddpm_params(config['ddpm_schedule'], timesteps, device)
    dataloader = get_dataloader(batch_size=config['batch_size'], data_limit=config.get('data_limit'))

    use_bfloat16_dit = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bfloat16_dit:
        print("Using bfloat16 for DiT model operations.")
    # Scaler for DiT model, always initialize, 'enabled' controls it
    scaler_dit = torch.cuda.amp.GradScaler(enabled=use_bfloat16_dit)
    # Scaler for Discriminator, if used and if DiT uses bfloat16 (D might run in fp32 still)
    # If D uses bfloat16 params or main training is bf16, better to use scaler for D too.
    scaler_d = torch.cuda.amp.GradScaler(enabled=use_bfloat16_dit and use_gan)


    print("Starting DiT training...")
    global_step = 0
    for epoch in range(config['epochs']):
        dit_model.train()
        if use_gan:
            discriminator.train()

        epoch_loss_mse = 0
        epoch_loss_g_adv = 0
        epoch_loss_d = 0
        
        start_time = time.time()
        
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for batch_idx, real_images in enumerate(batch_iterator):
            real_images = real_images.to(device)
            
            # --- Get clean latents from VAE ---
            with torch.no_grad(): # VAE is frozen
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=vae_uses_bfloat16):
                    if is_vqvae:
                        clean_latents, _ = autoencoder.encode(real_images) # z_q
                    else: # Standard VAE
                        mu, logvar = autoencoder.encode(real_images)
                        clean_latents = autoencoder.reparameterize(mu, logvar)
                        if config.get('vae_latent_is_flat', True): # Reshape if VAE latent is flat
                            clean_latents = clean_latents.view(real_images.size(0), 
                                                               vae_latent_channels, 
                                                               vae_latent_spatial_dim, 
                                                               vae_latent_spatial_dim)
            
            t = torch.randint(0, timesteps, (real_images.size(0),), device=device).long()
            noise_true = torch.randn_like(clean_latents)
            noisy_latents = q_sample(clean_latents, t, ddpm_params, noise=noise_true) # x_t
            
            # --- Train DiT (Generator part) ---
            optimizer_dit.zero_grad(set_to_none=True)
            loss_g_combined = torch.tensor(0.0, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_dit):
                predicted_noise = dit_model(noisy_latents, t) # epsilon_theta(x_t, t)
                loss_mse = F.mse_loss(predicted_noise, noise_true)
                loss_g_combined += loss_mse

                if use_gan:
                    # Predict x0 based on DiT's noise prediction
                    x0_pred_latents = predict_x0_from_xt_noise(noisy_latents, t, predicted_noise, ddpm_params)
                    # Clamp latents if specified, e.g. for standard VAEs to approx N(0,1) range
                    x0_pred_latents = x0_pred_latents.clamp(*config.get('vae_latent_clamp_range', (-2.0, 2.0)))


                    # Decode predicted x0 latents to images
                    if (not is_vqvae) and config.get('vae_latent_is_flat', True):
                        x0_pred_latents_for_decode = x0_pred_latents.view(x0_pred_latents.size(0), -1)
                    else:
                        x0_pred_latents_for_decode = x0_pred_latents
                    
                    # VAE decode and Discriminator forward should respect their own precision contexts
                    # but here they are within the DiT's autocast block.
                    # If VAE/D params are bf16, they work fine. If fp32, autocast converts inputs.
                    decoded_x0_images = autoencoder.decode(x0_pred_latents_for_decode)
                    
                    d_fake_output = discriminator(decoded_x0_images)
                    loss_g_adv = adversarial_loss_fn(d_fake_output, torch.ones_like(d_fake_output))
                    loss_g_combined += config.get('gan_loss_weight', 0.01) * loss_g_adv
            
            scaler_dit.scale(loss_g_combined).backward()
            scaler_dit.step(optimizer_dit)
            scaler_dit.update()

            epoch_loss_mse += loss_mse.item() * real_images.size(0)
            if use_gan:
                epoch_loss_g_adv += loss_g_adv.item() * real_images.size(0)

            # --- Train Discriminator ---
            if use_gan:
                optimizer_d.zero_grad(set_to_none=True)
                current_d_loss_val = 0
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_dit): # D also in DiT's autocast
                    # Real samples for D: VAE reconstructions of real images
                    if (not is_vqvae) and config.get('vae_latent_is_flat', True):
                        clean_latents_for_decode = clean_latents.view(clean_latents.size(0), -1)
                    else:
                        clean_latents_for_decode = clean_latents
                    
                    # Note: decoded_clean_images are VAE's best reconstruction, not "perfectly real" images.
                    # This aligns with VAE-GAN where D distinguishes VAE reconstructions from other generated ones.
                    decoded_clean_images = autoencoder.decode(clean_latents_for_decode) 
                                        
                    d_real_output = discriminator(decoded_clean_images.detach()) # Detach as D doesn't train VAE
                    d_loss_real = adversarial_loss_fn(d_real_output, torch.ones_like(d_real_output))
                    
                    # Fake samples for D: DiT's x0 predictions, decoded by VAE
                    d_fake_output_for_d = discriminator(decoded_x0_images.detach()) # Detach from DiT graph
                    d_loss_fake = adversarial_loss_fn(d_fake_output_for_d, torch.zeros_like(d_fake_output_for_d))
                    
                    loss_d = (d_loss_real + d_loss_fake) / 2
                    current_d_loss_val = loss_d.item()
                
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
                epoch_loss_d += current_d_loss_val * real_images.size(0)


            global_step += 1
            if batch_idx % config.get('log_interval', 100) == 0:
                log_msg = (f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(dataloader)}, "
                           f"MSE_Loss: {loss_mse.item():.4f}")
                writer.add_scalar('DiT/batch_loss_mse', loss_mse.item(), global_step)
                if use_gan:
                    log_msg += f", G_Adv_Loss: {loss_g_adv.item():.4f}, D_Loss: {current_d_loss_val:.4f}"
                    writer.add_scalar('DiT/batch_loss_g_adv', loss_g_adv.item(), global_step)
                    writer.add_scalar('DiT/batch_loss_d', current_d_loss_val, global_step)
                batch_iterator.set_postfix_str(log_msg)
                if batch_idx == 0 : print(log_msg) # Print first batch log
        
        epoch_time = time.time() - start_time
        num_processed_samples = len(dataloader.dataset) if config.get('data_limit') is None else min(len(dataloader.dataset), config.get('data_limit'))

        avg_epoch_loss_mse = epoch_loss_mse / num_processed_samples
        writer.add_scalar('DiT/epoch_loss_mse', avg_epoch_loss_mse, epoch + 1)
        epoch_summary_msg = f"Epoch {epoch+1} finished. Avg MSE Loss: {avg_epoch_loss_mse:.4f}"
        
        if use_gan:
            avg_epoch_loss_g_adv = epoch_loss_g_adv / num_processed_samples
            avg_epoch_loss_d = epoch_loss_d / num_processed_samples
            writer.add_scalar('DiT/epoch_loss_g_adv', avg_epoch_loss_g_adv, epoch + 1)
            writer.add_scalar('DiT/epoch_loss_d', avg_epoch_loss_d, epoch + 1)
            epoch_summary_msg += f", Avg G_Adv_Loss: {avg_epoch_loss_g_adv:.4f}, Avg D_Loss: {avg_epoch_loss_d:.4f}"
        
        epoch_summary_msg += f", Time: {epoch_time:.2f}s"
        print(epoch_summary_msg)


        if (epoch + 1) % config.get('sample_interval', 10) == 0:
            dit_model.eval()
            if use_gan:
                discriminator.eval() # Not strictly needed for sampling but good practice

            with torch.no_grad():
                num_samples_to_gen = config.get('num_samples_preview', 8)
                latent_sample_shape = (num_samples_to_gen, vae_latent_channels, 
                                       vae_latent_spatial_dim, vae_latent_spatial_dim)
                
                vae_decoder_fn = autoencoder.decode
                current_vae_is_flat_for_sampling = (not is_vqvae) and config.get('vae_latent_is_flat', True)
                
                # DiT model sampling (generates latents)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16_dit):
                    # p_sample_loop handles VAE decoding internally, including its own autocast via vae_uses_bfloat16
                    generated_images_final, _ = p_sample_loop(
                        dit_model, latent_sample_shape, timesteps, 
                        ddpm_params, device, 
                        vae_decoder=vae_decoder_fn,
                        vae_latent_is_flat=current_vae_is_flat_for_sampling,
                        vae_decode_uses_bfloat16=vae_uses_bfloat16 # Pass VAE's bfloat16 status
                    )
                
                generated_images_final = (generated_images_final.clamp(-1, 1) + 1) / 2 # Normalize to [0,1]
                grid = make_grid(generated_images_final)
                writer.add_image('DiT/generated_samples', grid, epoch + 1)
                save_image(grid, f"results/dit_samples/{run_name}_epoch_{epoch+1}.png")
            
            # Restore train mode after sampling
            dit_model.train()
            if use_gan:
                discriminator.train()


    # --- Save final models ---
    dit_model_save_path = os.path.join("checkpoints", f"{run_name}_DiT_final.pth")
    torch.save(dit_model.state_dict(), dit_model_save_path)
    print(f"DiT training finished. DiT Model saved to {dit_model_save_path}")
    
    if use_gan:
        d_model_save_path = os.path.join("checkpoints", f"{run_name}_Discriminator_final.pth")
        torch.save(discriminator.state_dict(), d_model_save_path)
        print(f"Discriminator model (trained with DiT) saved to {d_model_save_path}")

    writer.close()
    return dit_model_save_path


if __name__ == "__main__":
    print("Testing DiT training script with GAN option...")
    
    # --- Dummy VAE and Discriminator Setup (Mimicking VAE+GAN training output) ---
    # These paths would point to models trained by train_vae.py
    # For this test, we might need to generate placeholder .pth files if they don't exist.
    
    IMAGE_SIZE_FOR_TEST = IMAGE_SIZE # Use actual IMAGE_SIZE from data_loader
    
    # Config for a hypothetical VAE that was trained (used to load its structure)
    dummy_vae_latent_dim = 64 
    # DiT will reshape this flat latent for its spatial processing
    dit_latent_channels = 16 
    dit_latent_spatial_dim = int((dummy_vae_latent_dim / dit_latent_channels)**0.5) # e.g. 64/16 = 4 -> 2x2
    if dit_latent_spatial_dim**2 * dit_latent_channels != dummy_vae_latent_dim:
        print(f"Warning: dummy_vae_latent_dim {dummy_vae_latent_dim} cannot be perfectly reshaped to "
              f"{dit_latent_channels} channels and square spatial. Adjusting spatial dim for test.")
        # Example: if latent_dim=64, channels=16 -> 4 elements per channel. sqrt(4)=2. So 16x2x2
        # If latent_dim=128, channels=8 -> 16 elements per channel. sqrt(16)=4. So 8x4x4
        # For simplicity, ensure dummy_vae_latent_dim is compatible or hardcode.
        # Let's use dummy_vae_latent_dim = 16 * 2 * 2 = 64
        dit_latent_spatial_dim = 2 # Then dummy_vae_latent_dim should be 16*2*2=64
        if dummy_vae_latent_dim != dit_latent_channels * dit_latent_spatial_dim**2:
             dummy_vae_latent_dim = dit_latent_channels * dit_latent_spatial_dim**2
             print(f"Adjusted dummy_vae_latent_dim to {dummy_vae_latent_dim} for test consistency.")


    # Paths for dummy VAE and Discriminator
    base_checkpoint_dir = "checkpoints_train_dit_test" # Use a subdir to avoid conflict
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    test_vae_model_path = os.path.join(base_checkpoint_dir, "dummy_vae_for_dit_test.pth")
    test_discriminator_path = os.path.join(base_checkpoint_dir, "dummy_discriminator_for_dit_test.pth")

    # Create dummy VAE and Discriminator if they don't exist
    if not os.path.exists(test_vae_model_path):
        print(f"Creating dummy VAE model at {test_vae_model_path}")
        dummy_vae = VAE(input_channels=3, latent_dim=dummy_vae_latent_dim, image_size=IMAGE_SIZE_FOR_TEST)
        torch.save(dummy_vae.state_dict(), test_vae_model_path)
        del dummy_vae
    
    if not os.path.exists(test_discriminator_path):
        print(f"Creating dummy Discriminator model at {test_discriminator_path}")
        dummy_d = Discriminator(input_channels=3, image_size=IMAGE_SIZE_FOR_TEST)
        torch.save(dummy_d.state_dict(), test_discriminator_path)
        del dummy_d
    # --- End Dummy VAE/Discriminator Setup ---


    # --- Test Config for DiT ---
    test_config_dit_gan = {
        'vae_model_path': test_vae_model_path,
        'vae_flat_latent_dim': dummy_vae_latent_dim, # For standard VAE
        'vae_latent_is_flat': True,                 # True for standard VAE
        'vae_latent_channels': dit_latent_channels,         # DiT's channel view of VAE latent
        'vae_latent_spatial_dim': dit_latent_spatial_dim,    # DiT's spatial view
        'vae_latent_clamp_range': (-2.0, 2.0), # Example clamping for predicted x0 latents

        # VQVAE specific params (not used if vae_model_path points to std VAE)
        # 'vq_embedding_dim_for_dit': 64, 
        # 'vq_num_embeddings_for_dit': 128,
        # 'vq_hidden_dims_enc': [64, 64], 
        # 'vq_hidden_dims_dec': [64, 64],
        
        'lr': 1e-5, 'batch_size': 2, 'epochs': 1, 'data_limit': 8, # Small for quick test
        'ddpm_timesteps': 10, 'ddpm_schedule': 'linear', # Fast sampling
        
        'dit_patch_size': 1, # Patch size for DiT (operates on latent map)
        'dit_hidden_size': 32, 
        'dit_depth': 1, 
        'dit_num_heads': 2,
        
        'log_interval': 1, 'sample_interval': 1, 'use_bfloat16': False, # Keep False for CPU test

        # GAN specific config
        'use_gan': True,
        'discriminator_model_path': test_discriminator_path,
        'gan_loss_weight': 0.01,
        'lr_d': 1e-5,
    }
    
    print("\n--- Training DiT with GAN (test) ---")
    # Make sure IMAGE_SIZE is correctly propagated if VAE/Discriminator init depends on it globally.
    # In this script, IMAGE_SIZE is imported from data_loader.
    
    dit_model_path = train_dit_model(test_config_dit_gan)
    if dit_model_path:
        print(f"DiT (GAN) test training completed. DiT Model saved to {dit_model_path}")
    else:
        print("DiT (GAN) test training failed or was skipped.")

    # Test without GAN
    test_config_dit_no_gan = test_config_dit_gan.copy()
    test_config_dit_no_gan['use_gan'] = False
    del test_config_dit_no_gan['discriminator_model_path']
    del test_config_dit_no_gan['gan_loss_weight']
    del test_config_dit_no_gan['lr_d']

    print("\n--- Training DiT without GAN (test) ---")
    dit_model_path_no_gan = train_dit_model(test_config_dit_no_gan)
    if dit_model_path_no_gan:
        print(f"DiT (no GAN) test training completed. DiT Model saved to {dit_model_path_no_gan}")
    else:
        print("DiT (no GAN) test training failed or was skipped.")


    print("\nDiT training script tests finished.")
    print(f"Check '{base_checkpoint_dir}/' for dummy models if created.")
    print("Check 'runs/dit/' for TensorBoard logs and 'checkpoints/' for DiT models.")
    print("Check 'results/dit_samples/' for image samples.")
