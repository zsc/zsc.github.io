import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import os
import time

from data_loader import get_dataloader, IMAGE_SIZE
from models.vae import VAE, VQVAE # Import both

# Ensure checkpoints and tensorboard run directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("runs", exist_ok=True)
os.makedirs("results/vae_samples", exist_ok=True)


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0] # per image
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] # per image
    return recon_loss + beta * kld_loss, recon_loss, kld_loss

def vqvae_loss_function(recon_x, x, vq_loss, vq_beta=1.0):
    # Ensure recon_x and x have the same shape for MSE loss
    if recon_x.shape != x.shape:
        # This warning was already present, but good to be explicit about the check
        # The core fix is ensuring the model outputs the correct size.
        # However, as a safeguard or for models that might slightly vary, resizing can be an option,
        # but it's better to fix the model architecture.
        # For this specific problem, we expect the model architecture adjustment to fix it.
        # If not, one might add:
        # recon_x = F.interpolate(recon_x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        pass # Assuming the model architecture fix will handle this.

    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    total_loss = recon_loss + vq_beta * vq_loss
    return total_loss, recon_loss


def train_vae_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {config.get('use_vq', False) and 'VQVAE' or 'VAE'} training")

    dataloader = get_dataloader(batch_size=config['batch_size'], data_limit=config.get('data_limit'))

    use_vq = config.get('use_vq', False)
    model_type = "VQVAE" if use_vq else "VAE"

    # Corrected run_name generation
    run_name_parts = [
        model_type,
        f"lr{config['lr']}",
        f"bs{config['batch_size']}",
        f"epochs{config['epochs']}"
    ]
    if use_vq:
        run_name_parts.extend([
            f"ne{config['num_embeddings']}",
            f"ed{config['embedding_dim']}",
            f"cc{config['commitment_cost']}",
            f"encHD{'_'.join(map(str, config.get('vq_hidden_dims_enc', [])))}", # Log hidden dims
            f"decHD{'_'.join(map(str, config.get('vq_hidden_dims_dec', [])))}"  # Log hidden dims
        ])
    else: # For standard VAE
        run_name_parts.append(f"ld{config['latent_dim']}")
    run_name = "_".join(run_name_parts)

    writer = SummaryWriter(log_dir=os.path.join("runs", "vae", run_name))

    if use_vq:
        model = VQVAE(input_channels=3,
                      embedding_dim=config['embedding_dim'],
                      num_embeddings=config['num_embeddings'],
                      commitment_cost=config['commitment_cost'],
                      hidden_dims_enc=config.get('vq_hidden_dims_enc', [128, config['embedding_dim']]),
                      hidden_dims_dec=config.get('vq_hidden_dims_dec', [128])
                     ).to(device)
    else:
        model = VAE(input_channels=3, latent_dim=config['latent_dim']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    use_bfloat16 = config.get('use_bfloat16', False) and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if use_bfloat16:
        print(f"Using bfloat16 for {model_type} training.")

    print(f"Starting {model_type} training...")
    global_step = 0
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0 # For VAE
        epoch_vq_commitment_loss = 0 # For VQVAE

        start_time = time.time()
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True) # More memory efficient

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                if use_vq:
                    recon_batch, vq_loss_val, _, _ = model(images)
                    loss, recon_l = vqvae_loss_function(recon_batch, images, vq_loss_val, vq_beta=config.get('vq_beta', 1))
                    epoch_vq_commitment_loss += vq_loss_val.item() * images.size(0)
                else:
                    recon_batch, mu, logvar, _ = model(images)
                    loss, recon_l, kld_l = vae_loss_function(recon_batch, images, mu, logvar, beta=config.get('kld_beta', 0.3))
                    epoch_kld_loss += kld_l.item() * images.size(0)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_recon_loss += recon_l.item() * images.size(0)
            global_step += 1

            if batch_idx % config.get('log_interval', 100) == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
                writer.add_scalar(f'{model_type}/batch_loss', loss.item(), global_step)
                if use_vq:
                    writer.add_scalar(f'{model_type}/batch_recon_loss', recon_l.item(), global_step)
                    writer.add_scalar(f'{model_type}/batch_vq_loss', vq_loss_val.item(), global_step)
                else:
                    writer.add_scalar(f'{model_type}/batch_recon_loss', recon_l.item(), global_step)
                    writer.add_scalar(f'{model_type}/batch_kld_loss', kld_l.item(), global_step)


        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        avg_epoch_recon_loss = epoch_recon_loss / len(dataloader.dataset)

        writer.add_scalar(f'{model_type}/epoch_loss', avg_epoch_loss, epoch + 1)
        writer.add_scalar(f'{model_type}/epoch_recon_loss', avg_epoch_recon_loss, epoch + 1)
        if use_vq:
            avg_epoch_vq_loss = epoch_vq_commitment_loss / len(dataloader.dataset)
            writer.add_scalar(f'{model_type}/epoch_vq_loss', avg_epoch_vq_loss, epoch + 1)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
                  f"Avg Recon Loss: {avg_epoch_recon_loss:.4f}, Avg VQ Loss: {avg_epoch_vq_loss:.4f}, Time: {epoch_time:.2f}s")
        else:
            avg_epoch_kld_loss = epoch_kld_loss / len(dataloader.dataset)
            writer.add_scalar(f'{model_type}/epoch_kld_loss', avg_epoch_kld_loss, epoch + 1)
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, "
                  f"Avg Recon Loss: {avg_epoch_recon_loss:.4f}, Avg KLD Loss: {avg_epoch_kld_loss:.4f}, Time: {epoch_time:.2f}s")


        if (epoch + 1) % config.get('sample_interval', 1) == 0: # sample_interval default to 1 for quick test
            model.eval()
            with torch.no_grad():
                # Fetch a fixed batch for consistent visualization if possible, or just the next batch
                try:
                    # Attempt to re-use a fixed batch if defined, otherwise grab next
                    fixed_test_batch = getattr(train_vae_model, 'fixed_test_batch', next(iter(dataloader)))
                    train_vae_model.fixed_test_batch = fixed_test_batch # store for next time
                except StopIteration: # Dataloader exhausted
                    dataloader_iter_for_test = iter(get_dataloader(batch_size=8, data_limit=config.get('data_limit')))
                    fixed_test_batch = next(dataloader_iter_for_test)
                    train_vae_model.fixed_test_batch = fixed_test_batch


                test_batch = fixed_test_batch[:8].to(device) # Use first 8 images

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    if use_vq:
                        recon_samples, _, _, _ = model(test_batch)
                    else:
                        recon_samples, _, _, _ = model(test_batch)

                # Ensure recon_samples are on CPU for make_grid and have correct range
                comparison = torch.cat([test_batch.cpu(), recon_samples.cpu()])
                comparison = (comparison.clamp(-1, 1) + 1) / 2 # Unnormalize from [-1,1] to [0,1]
                grid = make_grid(comparison)
                writer.add_image(f'{model_type}/reconstructions', grid, epoch + 1)
                save_image(grid, f"results/vae_samples/{model_type}_reconstruction_epoch_{epoch+1}_{run_name}.png")

                # Log generated samples
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                    if use_vq:
                        # VQVAE.sample might need num_samples and device
                        generated_samples = model.sample(num_samples=8, device=device)
                    else:
                        generated_samples = model.sample(num_samples=8, device=device) # VAE sample method

                generated_samples = (generated_samples.cpu().clamp(-1, 1) + 1) / 2 # Unnormalize
                grid_generated = make_grid(generated_samples)
                writer.add_image(f'{model_type}/generated_samples', grid_generated, epoch + 1)
                save_image(grid_generated, f"results/vae_samples/{model_type}_generated_epoch_{epoch+1}_{run_name}.png")

    # Save model
    model_save_path = os.path.join("checkpoints", f"{run_name}_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"{model_type} training finished. Model saved to {model_save_path}")
    writer.close()
    if hasattr(train_vae_model, 'fixed_test_batch'): # Clean up
        delattr(train_vae_model, 'fixed_test_batch')
    return model_save_path

# Test for train_vae.py
if __name__ == "__main__":
    print("Testing VAE training script...")

    # Minimal config for VAE test
    test_config_vae = {
        'lr': 1e-4,
        'batch_size': 8, # Small for quick test
        'epochs': 2,    # Very few epochs
        'latent_dim': 32, # Small latent dim
        'log_interval': 1,
        'sample_interval': 1,
        'data_limit': 32, # Use very small dataset
        'use_bfloat16': True, # Test bfloat16 if available
        'use_vq': False,
        'kld_beta': 1.0,
    }
    print("\n--- Training VAE (test) ---")
    vae_model_path = train_vae_model(test_config_vae)
    print(f"VAE test training completed. Model saved to {vae_model_path}")

    # Minimal config for VQ-VAE test
    test_config_vqvae = {
        'lr': 1e-4,
        'batch_size': 8,
        'epochs': 2,
        'embedding_dim': 16, # For VQVAE, this is the depth of the quantized latent space.
        'num_embeddings': 64,
        'commitment_cost': 0.25,
        'vq_beta': 1.0, # Weight for vq_loss in total_loss
        'log_interval': 1,
        'sample_interval': 1,
        'data_limit': 32,
        'use_bfloat16': True,
        'use_vq': True,
        'vq_hidden_dims_enc': [32, 16], # 2 downsampling stages: 48->24->12. Last must be embedding_dim.
        # To get 2 upsampling stages (12->24->48) from a model that does len(dims)+1 upsamples:
        # We need len(vq_hidden_dims_dec) = 1.
        'vq_hidden_dims_dec': [32], # 1 configurable upsampling stage (e.g., 16_emb_dim -> 32 channels).
                                    # Plus 1 fixed upsample in model = 2 total upsamples.
    }
    print("\n--- Training VQ-VAE (test) ---")
    vqvae_model_path = train_vae_model(test_config_vqvae)
    print(f"VQ-VAE test training completed. Model saved to {vqvae_model_path}")

    print("\nVAE training script tests finished.")
    print("Check 'runs/vae/' for TensorBoard logs and 'checkpoints/' for models.")
    print("Check 'results/vae_samples/' for image samples.")
