import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn # Added for BCEWithLogitsLoss
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import os
import time

from data_loader import get_dataloader, IMAGE_SIZE
from models.vae import VAE, VQVAE, Discriminator # Import Discriminator

# Ensure checkpoints and tensorboard run directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("results/vae_samples", exist_ok=True)


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0] 
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kld_loss, recon_loss, kld_loss

def vqvae_loss_function(recon_x, x, vq_loss, vq_beta=1.0):
    if recon_x.shape != x.shape:
        # This might indicate an issue with model architecture if sizes don't match input
        # Forcing interpolate can hide bugs but can be a temporary fix.
        # recon_x = F.interpolate(recon_x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        print(f"Warning: recon_x shape {recon_x.shape} and x shape {x.shape} mismatch. Check model architecture.")
        # Attempting to pad/crop if minor difference, common with VQVAE due to discrete latents and upsampling.
        # This is a simplified handling. A robust solution needs careful architecture design.
        diff_h = recon_x.shape[2] - x.shape[2]
        diff_w = recon_x.shape[3] - x.shape[3]
        if diff_h > 0: recon_x = recon_x[:, :, diff_h//2:x.shape[2]+diff_h//2, :]
        if diff_w > 0: recon_x = recon_x[:, :, :, diff_w//2:x.shape[3]+diff_w//2]
        if diff_h < 0: recon_x = F.pad(recon_x, (0,0, -diff_h//2, (-diff_h -diff_h//2)), mode='replicate')
        if diff_w < 0: recon_x = F.pad(recon_x, (-diff_w//2, (-diff_w -diff_w//2), 0,0), mode='replicate')


    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    total_loss = recon_loss + vq_beta * vq_loss
    return total_loss, recon_loss


def train_vae_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_description = "VQVAE" if config.get('use_vq', False) else "VAE"
    if config.get('use_gan', False):
        model_description += "+GAN"
    print(f"Using device: {device} for {model_description} training")

    dataloader = get_dataloader(batch_size=config['batch_size'], data_limit=config.get('data_limit'))

    use_vq = config.get('use_vq', False)
    use_gan = config.get('use_gan', False)
    model_type_base = "VQVAE" if use_vq else "VAE"
    model_type_full = model_type_base + ("_GAN" if use_gan else "")


    run_name_parts = [
        model_type_full,
        f"lr{config['lr']}",
        f"bs{config['batch_size']}",
        f"epochs{config['epochs']}"
    ]
    if use_vq:
        run_name_parts.extend([
            f"ne{config['num_embeddings']}",
            f"ed{config['embedding_dim']}",
            f"cc{config['commitment_cost']}",
            f"encHD{'_'.join(map(str, config.get('vq_hidden_dims_enc', [])))}",
            f"decHD{'_'.join(map(str, config.get('vq_hidden_dims_dec', [])))}"
        ])
    else: # For standard VAE
        run_name_parts.append(f"ld{config['latent_dim']}")
    
    if use_gan:
        run_name_parts.append(f"ganW{config.get('gan_loss_weight', 1.0)}")
        run_name_parts.append(f"lrD{config.get('lr_d', config['lr'])}")

    run_name = "_".join(run_name_parts)

    writer = SummaryWriter(log_dir=os.path.join("runs", "vae", run_name)) # Changed "vae" to model_type_base for clarity if needed

    if use_vq:
        model = VQVAE(input_channels=3,
                      embedding_dim=config['embedding_dim'],
                      num_embeddings=config['num_embeddings'],
                      commitment_cost=config['commitment_cost'],
                      hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, config['embedding_dim']]), # Ensure last is emb_dim
                      hidden_dims_dec=config.get('vq_hidden_dims_dec', [config['embedding_dim'] // 2]), # Adjust based on structure
                      image_size=IMAGE_SIZE
                     ).to(device)
    else:
        model = VAE(input_channels=3, latent_dim=config['latent_dim'], image_size=IMAGE_SIZE).to(device)

    optimizer_g = optim.AdamW(model.parameters(), lr=config['lr']) # Renamed to optimizer_g for clarity

    discriminator = None
    optimizer_d = None
    adversarial_loss_fn = None
    if use_gan:
        discriminator = Discriminator(input_channels=3, image_size=IMAGE_SIZE).to(device)
        optimizer_d = optim.AdamW(discriminator.parameters(), lr=config.get('lr_d', config['lr']))
        adversarial_loss_fn = nn.BCEWithLogitsLoss()

    use_bfloat16 = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bfloat16:
        print(f"Using bfloat16 for {model_type_full} training.")

    print(f"Starting {model_type_full} training...")
    global_step = 0
    for epoch in range(config['epochs']):
        model.train()
        if use_gan:
            discriminator.train()

        epoch_loss_g = 0 # Generator's total loss
        epoch_recon_loss = 0
        epoch_kld_loss = 0 
        epoch_vq_commitment_loss = 0
        epoch_loss_g_adv = 0 # Generator's adversarial loss part
        epoch_loss_d = 0 # Discriminator's loss

        start_time = time.time()
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            # --- Train Generator (VAE/VQ-VAE) ---
            optimizer_g.zero_grad(set_to_none=True)
            loss_g_combined = None # Define to ensure it's assigned
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                if use_vq:
                    recon_batch, vq_loss_val, _, _ = model(images)
                    loss_g_orig, recon_l = vqvae_loss_function(recon_batch, images, vq_loss_val, vq_beta=config.get('vq_beta', 1.0))
                    epoch_vq_commitment_loss += vq_loss_val.item() * images.size(0)
                else: # Standard VAE
                    recon_batch, mu, logvar, _ = model(images)
                    loss_g_orig, recon_l, kld_l = vae_loss_function(recon_batch, images, mu, logvar, beta=config.get('kld_beta', 0.3))
                    epoch_kld_loss += kld_l.item() * images.size(0)

                if use_gan:
                    # Discriminator's output for fake images (from G's perspective, G wants D to think they are real)
                    # D's weights are fixed here (no optimizer_d.step())
                    gen_fake_output = discriminator(recon_batch)
                    # Target for G is to make D output "real" (1s)
                    loss_g_adv = adversarial_loss_fn(gen_fake_output, torch.ones_like(gen_fake_output).to(device))
                    loss_g_combined = loss_g_orig + config.get('gan_loss_weight', 1.0) * loss_g_adv
                    epoch_loss_g_adv += loss_g_adv.item() * images.size(0)
                else:
                    loss_g_combined = loss_g_orig
            
            loss_g_combined.backward()
            optimizer_g.step()

            epoch_loss_g += loss_g_combined.item() * images.size(0)
            epoch_recon_loss += recon_l.item() * images.size(0)


            # --- Train Discriminator ---
            if use_gan:
                optimizer_d.zero_grad(set_to_none=True)
                current_d_loss_val = 0
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    # Real images
                    real_output = discriminator(images)
                    # Target for D is "real" (1s) for real images
                    d_loss_real = adversarial_loss_fn(real_output, torch.ones_like(real_output).to(device))
                    
                    # Fake images (detached, as D is not training G here)
                    fake_output = discriminator(recon_batch.detach())
                    # Target for D is "fake" (0s) for fake images
                    d_loss_fake = adversarial_loss_fn(fake_output, torch.zeros_like(fake_output).to(device))
                    
                    loss_d = (d_loss_real + d_loss_fake) / 2
                    current_d_loss_val = loss_d.item() # store for logging before backward
                
                loss_d.backward()
                optimizer_d.step()
                epoch_loss_d += current_d_loss_val * images.size(0)


            global_step += 1
            if batch_idx % config.get('log_interval', 100) == 0:
                log_msg = (f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(dataloader)}, "
                           f"G_Loss: {loss_g_combined.item():.4f}, Recon_Loss: {recon_l.item():.4f}")
                writer.add_scalar(f'{model_type_full}/batch_loss_g', loss_g_combined.item(), global_step)
                writer.add_scalar(f'{model_type_full}/batch_recon_loss', recon_l.item(), global_step)

                if use_vq:
                    writer.add_scalar(f'{model_type_full}/batch_vq_loss', vq_loss_val.item(), global_step)
                    log_msg += f", VQ_Loss: {vq_loss_val.item():.4f}"
                else: # Standard VAE
                    writer.add_scalar(f'{model_type_full}/batch_kld_loss', kld_l.item(), global_step)
                    log_msg += f", KLD_Loss: {kld_l.item():.4f}"
                
                if use_gan:
                    writer.add_scalar(f'{model_type_full}/batch_loss_g_adv', loss_g_adv.item(), global_step)
                    writer.add_scalar(f'{model_type_full}/batch_loss_d', current_d_loss_val, global_step) # current_d_loss_val
                    log_msg += f", G_Adv_Loss: {loss_g_adv.item():.4f}, D_Loss: {current_d_loss_val:.4f}" # current_d_loss_val
                print(log_msg)

        epoch_time = time.time() - start_time
        avg_epoch_loss_g = epoch_loss_g / len(dataloader.dataset)
        avg_epoch_recon_loss = epoch_recon_loss / len(dataloader.dataset)

        writer.add_scalar(f'{model_type_full}/epoch_loss_g', avg_epoch_loss_g, epoch + 1)
        writer.add_scalar(f'{model_type_full}/epoch_recon_loss', avg_epoch_recon_loss, epoch + 1)
        
        epoch_summary_msg = (f"Epoch {epoch+1} finished. Avg G_Loss: {avg_epoch_loss_g:.4f}, "
                             f"Avg Recon Loss: {avg_epoch_recon_loss:.4f}")

        if use_vq:
            avg_epoch_vq_loss = epoch_vq_commitment_loss / len(dataloader.dataset)
            writer.add_scalar(f'{model_type_full}/epoch_vq_loss', avg_epoch_vq_loss, epoch + 1)
            epoch_summary_msg += f", Avg VQ Loss: {avg_epoch_vq_loss:.4f}"
        else: # Standard VAE
            avg_epoch_kld_loss = epoch_kld_loss / len(dataloader.dataset)
            writer.add_scalar(f'{model_type_full}/epoch_kld_loss', avg_epoch_kld_loss, epoch + 1)
            epoch_summary_msg += f", Avg KLD Loss: {avg_epoch_kld_loss:.4f}"
        
        if use_gan:
            avg_epoch_loss_g_adv = epoch_loss_g_adv / len(dataloader.dataset)
            avg_epoch_loss_d = epoch_loss_d / len(dataloader.dataset)
            writer.add_scalar(f'{model_type_full}/epoch_loss_g_adv', avg_epoch_loss_g_adv, epoch + 1)
            writer.add_scalar(f'{model_type_full}/epoch_loss_d', avg_epoch_loss_d, epoch + 1)
            epoch_summary_msg += (f", Avg G_Adv_Loss: {avg_epoch_loss_g_adv:.4f}, "
                                  f"Avg D_Loss: {avg_epoch_loss_d:.4f}")
        
        epoch_summary_msg += f", Time: {epoch_time:.2f}s"
        print(epoch_summary_msg)


        if (epoch + 1) % config.get('sample_interval', 1) == 0:
            model.eval() # Generator to eval mode
            # Discriminator is not used for sampling, so its mode doesn't matter here
            with torch.no_grad():
                try:
                    fixed_test_batch = getattr(train_vae_model, 'fixed_test_batch', next(iter(dataloader)))
                    train_vae_model.fixed_test_batch = fixed_test_batch 
                except StopIteration: 
                    dataloader_iter_for_test = iter(get_dataloader(batch_size=8, data_limit=config.get('data_limit')))
                    fixed_test_batch = next(dataloader_iter_for_test)
                    train_vae_model.fixed_test_batch = fixed_test_batch

                test_batch = fixed_test_batch[:8].to(device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    if use_vq:
                        recon_samples, _, _, _ = model(test_batch)
                    else: # Standard VAE
                        recon_samples, _, _, _ = model(test_batch)
                
                # Check for shape mismatch before cat
                if recon_samples.shape != test_batch.shape:
                    print(f"Shape mismatch during sampling: test_batch {test_batch.shape}, recon_samples {recon_samples.shape}. Attempting to resize recon_samples.")
                    recon_samples = F.interpolate(recon_samples, size=(test_batch.shape[2], test_batch.shape[3]), mode='bilinear', align_corners=False)


                comparison = torch.cat([test_batch.cpu(), recon_samples.cpu()])
                comparison = (comparison.clamp(-1, 1) + 1) / 2 
                grid = make_grid(comparison)
                writer.add_image(f'{model_type_full}/reconstructions', grid, epoch + 1)
                save_image(grid, f"results/vae_samples/{model_type_full}_reconstruction_epoch_{epoch+1}_{run_name}.png")

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    if use_vq:
                        generated_samples = model.sample(num_samples=8, device=device)
                    else: # Standard VAE
                        generated_samples = model.sample(num_samples=8, device=device)

                generated_samples = (generated_samples.cpu().clamp(-1, 1) + 1) / 2
                grid_generated = make_grid(generated_samples)
                writer.add_image(f'{model_type_full}/generated_samples', grid_generated, epoch + 1)
                save_image(grid_generated, f"results/vae_samples/{model_type_full}_generated_epoch_{epoch+1}_{run_name}.png")

    # Save model(s)
    g_model_save_path = os.path.join("checkpoints", f"{run_name}_generator_final.pth")
    torch.save(model.state_dict(), g_model_save_path)
    print(f"{model_type_base} (Generator) training finished. Model saved to {g_model_save_path}")
    
    final_save_path = g_model_save_path # Default return

    if use_gan:
        d_model_save_path = os.path.join("checkpoints", f"{run_name}_discriminator_final.pth")
        torch.save(discriminator.state_dict(), d_model_save_path)
        print(f"Discriminator training finished. Model saved to {d_model_save_path}")
        # Could return a dict of paths, but prompt implies one path. Returning G path.

    writer.close()
    if hasattr(train_vae_model, 'fixed_test_batch'):
        delattr(train_vae_model, 'fixed_test_batch')
    return final_save_path # Returns generator path

# Test for train_vae.py
if __name__ == "__main__":
    print("Testing VAE/VQ-VAE (+GAN) training script...")
    IMAGE_SIZE_FROM_LOADER = 48 # Assuming this is the size from data_loader.py

    # Minimal config for VAE test
    test_config_vae = {
        'lr': 1e-4,
        'batch_size': 4, 
        'epochs': 1,    
        'latent_dim': 16, 
        'log_interval': 1,
        'sample_interval': 1,
        'data_limit': 16, 
        'use_bfloat16': False, # Simpler for quick CPU test
        'use_vq': False,
        'kld_beta': 1.0,
        'use_gan': False, # Test without GAN first
    }
    print("\n--- Training VAE (test) ---")
    vae_model_path = train_vae_model(test_config_vae)
    print(f"VAE test training completed. Model saved to {vae_model_path}")

    # Minimal config for VAE+GAN test
    test_config_vae_gan = {
        **test_config_vae, # Inherit VAE settings
        'use_gan': True,
        'gan_loss_weight': 0.01, # Smaller weight for GAN loss initially
        'lr_d': 1e-4, 
    }
    print("\n--- Training VAE+GAN (test) ---")
    vae_gan_model_path = train_vae_model(test_config_vae_gan)
    print(f"VAE+GAN test training completed. Generator model saved to {vae_gan_model_path}")


    # Minimal config for VQ-VAE test
    test_config_vqvae = {
        'lr': 1e-4,
        'batch_size': 4,
        'epochs': 1,
        'embedding_dim': 8, 
        'num_embeddings': 16,
        'commitment_cost': 0.25,
        'vq_beta': 1.0, 
        'log_interval': 1,
        'sample_interval': 1,
        'data_limit': 16,
        'use_bfloat16': False,
        'use_vq': True,
        'vq_hidden_dims_enc': [32], # 48->24, then conv1x1 to emb_dim.
        'vq_hidden_dims_dec': [],  # emb_dim -> 3_out (direct TConv for 24->48)
        'use_gan': False,
    }
    print("\n--- Training VQ-VAE (test) ---")
    vqvae_model_path = train_vae_model(test_config_vqvae)
    print(f"VQ-VAE test training completed. Model saved to {vqvae_model_path}")

    # Minimal config for VQ-VAE+GAN test
    test_config_vqvae_gan = {
        **test_config_vqvae, # Inherit VQ-VAE settings
        'use_gan': True,
        'gan_loss_weight': 0.01,
        'lr_d': 1e-4,
    }
    print("\n--- Training VQ-VAE+GAN (test) ---")
    vqvae_gan_model_path = train_vae_model(test_config_vqvae_gan)
    print(f"VQ-VAE+GAN test training completed. Generator model saved to {vqvae_gan_model_path}")

    print("\nVAE training script tests finished.")
    print("Check 'runs/vae/' for TensorBoard logs and 'checkpoints/' for models.")
    print("Check 'results/vae_samples/' for image samples.")
