# train.py
import os
import time
import queue
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import gc
from datetime import datetime

# Import models from model.py
from model import VQGAN, NLayerDiscriminator, VQGANTransformer

# --- Data Loading ---

class CelebADataset(Dataset):
    """Custom PyTorch Dataset for the pre-processed CelebA NPZ file."""
    def __init__(self, npz_path, transform=None):
        """
        Args:
            npz_path (string): Path to the npz file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        data = np.load(npz_path)
        self.images = data['images']
        # self.attributes = data['attributes'] # Not used for unconditional generation
        self.transform = transform
        print(f"Loaded {len(self.images)} images from {npz_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def image_transform(image):
    """
    Default transform for images:
    1. Convert numpy array [H, W, C] (uint8) to torch tensor [C, H, W].
    2. Normalize from [0, 255] to [-1, 1] (float32).
    """
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = (image / 255.0) * 2.0 - 1.0
    return image

def get_data_loader(npz_path, batch_size, shuffle=True):
    """Creates and returns a DataLoader for the CelebA dataset."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found at {npz_path}")
    dataset = CelebADataset(npz_path, transform=image_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

# --- Helper Functions ---

def log_message(log_queue, message):
    """Puts a message into the log queue if it exists, otherwise prints."""
    if log_queue:
        log_queue.put(message)
    else:
        print(message)

def get_exp_name(config, stage):
    """Generates a unique experiment name with hyperparameters and timestamp."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if stage == 1:
        name = f"VQGAN_e{config['epochs']}_b{config['batch_size']}_lr{config['lr']}_ne{config['n_embed']}_cc{config['commitment_cost']}_{ts}"
    else: # stage 2
        name = f"Transformer_e{config['epochs']}_b{config['batch_size']}_lr{config['lr']}_nl{config['n_layer']}_nh{config['n_head']}_{ts}"
    return name

def log_images_to_tb(writer, step, real_images, reconstructed_images, tag_prefix):
    """Logs real and reconstructed images to TensorBoard."""
    real_grid = make_grid(real_images.cpu(), nrow=4, normalize=True, value_range=(-1, 1))
    recon_grid = make_grid(reconstructed_images.cpu(), nrow=4, normalize=True, value_range=(-1, 1))
    writer.add_image(f'{tag_prefix}/Real_Images', real_grid, global_step=step)
    writer.add_image(f'{tag_prefix}/Reconstructed_Images', recon_grid, global_step=step)

# --- Main Training Functions ---

def train_stage1(config, training_controls, log_queue, device):
    """
    Main training loop for VQGAN (Stage 1).
    """
    # 1. Setup
    exp_name = get_exp_name(config, 1)
    log_dir = os.path.join('runs', exp_name)
    ckpt_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    log_message(log_queue, f"--- Starting VQGAN Training (Stage 1) ---")
    log_message(log_queue, f"Experiment: {exp_name}")
    log_message(log_queue, f"Config: {config}")

    # 2. Data Loader
    try:
        dataloader = get_data_loader(config['data_path'], config['batch_size'])
    except FileNotFoundError as e:
        log_message(log_queue, f"ERROR: {e}")
        return

    # 3. Models & Optimizers
    vqgan_config = {
        'in_channels': 3, 'out_channels': 3, 'ch': 128, 'ch_mult': [1, 1, 2],
        'num_res_blocks': 2, 'attn_resolutions': [16], 'resolution': 64,
        'embed_dim': 256, 'n_embed': config['n_embed'], 'commitment_cost': config['commitment_cost']
    }
    model = VQGAN(vqgan_config).to(device)
    discriminator = NLayerDiscriminator(input_nc=3, n_layers=3).to(device)

    opt_vq = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.5, 0.9))
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.9))
    
    # Use bfloat16 for training on 4090
    # scaler is a no-op for bfloat16 but good practice for mixed precision code
    scaler_vq = torch.amp.GradScaler()
    scaler_disc = torch.amp.GradScaler()
    autocast_dtype = torch.bfloat16

    # 4. Checkpoint Loading
    start_epoch = 0
    global_step = 0
    if config['checkpoint_path']:
        try:
            log_message(log_queue, f"Loading checkpoint from {config['checkpoint_path']}")
            # NOTE: Using weights_only=False is correct here because you load the config dict.
            # The warning from PyTorch is informational.
            checkpoint = torch.load(config['checkpoint_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            opt_vq.load_state_dict(checkpoint['opt_vq_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            log_message(log_queue, f"Resuming from epoch {start_epoch}, step {global_step}")
        except Exception as e:
            log_message(log_queue, f"ERROR loading checkpoint: {e}. Starting from scratch.")
    
    # 5. Training Loop
    try:
        for epoch in range(start_epoch, config['epochs']):
            log_message(log_queue, f"Epoch {epoch}/{config['epochs']-1}")
            for i, batch in enumerate(dataloader):
                if training_controls['stop']:
                    log_message(log_queue, "Training stopped by user.")
                    break
                
                real_images = batch.to(device)
                
                # ----- Discriminator Update -----
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    x_rec, _, _ = model(real_images)
                    logits_real = discriminator(real_images.contiguous())
                    logits_fake = discriminator(x_rec.detach())
                    d_loss_real = torch.mean(F.relu(1. - logits_real))
                    d_loss_fake = torch.mean(F.relu(1. + logits_fake))
                    d_loss = 0.5 * (d_loss_real + d_loss_fake)
                
                opt_disc.zero_grad(set_to_none=True)
                scaler_disc.scale(d_loss).backward()
                scaler_disc.step(opt_disc)
                scaler_disc.update()

                # ----- Generator (VQGAN) Update -----
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    x_rec, codebook_loss, commit_loss = model(real_images)
                    
                    # Reconstruction Loss
                    rec_loss = F.l1_loss(x_rec, real_images)
                    
                    # Adversarial Loss
                    logits_fake_gen = discriminator(x_rec)
                    gan_loss = -torch.mean(logits_fake_gen)
                    
                    # Total Loss
                    g_loss = rec_loss + config['gan_weight'] * gan_loss + codebook_loss + config['commitment_cost'] * commit_loss

                opt_vq.zero_grad(set_to_none=True)
                scaler_vq.scale(g_loss).backward()
                scaler_vq.step(opt_vq)
                scaler_vq.update()
                
                # Logging
                if global_step % 100 == 0:
                    log_msg = (f"Step {global_step} | G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f} | "
                               f"Rec: {rec_loss.item():.4f} | Codebook: {codebook_loss.item():.4f} | Commit: {commit_loss.item():.4f}")
                    log_message(log_queue, log_msg)
                    writer.add_scalar('Loss/Total_Generator', g_loss.item(), global_step)
                    writer.add_scalar('Loss/Reconstruction', rec_loss.item(), global_step)
                    writer.add_scalar('Loss/Adversarial', gan_loss.item(), global_step)
                    writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                    writer.add_scalar('Loss_Debug/Codebook', codebook_loss.item(), global_step)
                    writer.add_scalar('Loss_Debug/Commitment', commit_loss.item(), global_step)
                
                # Image logging
                if global_step % 1000 == 0:
                    log_images_to_tb(writer, global_step, real_images[:4], x_rec[:4], 'Reconstruction')

                global_step += 1
            
            # Check for stop signal after each epoch
            if training_controls['stop']:
                break
            
            # Checkpointing
            if (epoch + 1) % config['save_epoch_freq'] == 0:
                ckpt_path = os.path.join(ckpt_dir, f'vqgan_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'opt_vq_state_dict': opt_vq.state_dict(),
                    'opt_disc_state_dict': opt_disc.state_dict(),
                    'config': vqgan_config
                }, ckpt_path)
                log_message(log_queue, f"Saved checkpoint to {ckpt_path}")

    except Exception as e:
        import traceback
        log_message(log_queue, f"An error occurred during training: {e}")
        log_message(log_queue, traceback.format_exc())
    finally:
        log_message(log_queue, "--- VQGAN Training Finished ---")
        writer.close()
        gc.collect()
        torch.cuda.empty_cache()


def train_stage2(config, training_controls, log_queue, device):
    """
    Main training loop for VQGAN Transformer (Stage 2).
    """
    # 1. Setup
    exp_name = get_exp_name(config, 2)
    log_dir = os.path.join('runs', exp_name)
    ckpt_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    log_message(log_queue, f"--- Starting Transformer Training (Stage 2) ---")
    log_message(log_queue, f"Experiment: {exp_name}")
    log_message(log_queue, f"Config: {config}")

    # 2. Data Loader
    try:
        dataloader = get_data_loader(config['data_path'], config['batch_size'])
    except FileNotFoundError as e:
        log_message(log_queue, f"ERROR: {e}")
        return

    # 3. Load pre-trained VQGAN
    try:
        log_message(log_queue, f"Loading pre-trained VQGAN from {config['vqgan_checkpoint_path']}")
        # NOTE: Using weights_only=False is correct here because you load the config dict.
        vqgan_ckpt = torch.load(config['vqgan_checkpoint_path'], map_location='cpu')
        vqgan_model = VQGAN(vqgan_ckpt['config']).to(device)
        vqgan_model.load_state_dict(vqgan_ckpt['model_state_dict'])
        vqgan_model.eval()
        for p in vqgan_model.parameters(): p.requires_grad = False
        log_message(log_queue, "VQGAN loaded successfully.")
    except Exception as e:
        log_message(log_queue, f"ERROR loading VQGAN model: {e}. Cannot proceed with Stage 2.")
        return

    # 4. Transformer Model & Optimizer
    f = 2**(len(vqgan_ckpt['config']['ch_mult'])-1)
    latent_h = vqgan_ckpt['config']['resolution'] // f
    latent_w = latent_h
    transformer_config = {
        'vocab_size': vqgan_ckpt['config']['n_embed'],
        'block_size': latent_h * latent_w,
        'n_layer': config['n_layer'], 'n_head': config['n_head'], 'n_embd': config['n_embd']
    }
    model = VQGANTransformer(transformer_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.95))
    
    scaler = torch.amp.GradScaler()
    autocast_dtype = torch.bfloat16

    # 5. Checkpoint Loading
    start_epoch = 0
    global_step = 0
    if config['checkpoint_path']:
        try:
            log_message(log_queue, f"Loading transformer checkpoint from {config['checkpoint_path']}")
            checkpoint = torch.load(config['checkpoint_path'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            log_message(log_queue, f"Resuming from epoch {start_epoch}, step {global_step}")
        except Exception as e:
            log_message(log_queue, f"ERROR loading checkpoint: {e}. Starting from scratch.")
    
    # Add graph to tensorboard
    if global_step == 0:
        # FIX 2: Set model to eval() for tracing to avoid dropout issues, then set back to train()
        model.eval()
        dummy_indices = torch.randint(0, transformer_config['vocab_size'], (1, transformer_config['block_size']), device=device)
        model.train()

    # 6. Training Loop
    try:
        for epoch in range(start_epoch, config['epochs']):
            log_message(log_queue, f"Epoch {epoch}/{config['epochs']-1}")
            for i, batch in enumerate(dataloader):
                if training_controls['stop']:
                    log_message(log_queue, "Training stopped by user.")
                    break
                
                real_images = batch.to(device)
                
                with torch.no_grad():
                    _, _, _, indices = vqgan_model.encode(real_images)
                    indices = indices.view(indices.size(0), -1)

                # Prepare input and target sequences
                input_seq = indices[:, :-1]
                target_seq = indices[:, 1:]

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    logits = model(input_seq)
                    # Reshape for cross-entropy loss
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                # Logging
                if global_step % 100 == 0:
                    log_msg = f"Step {global_step} | Loss: {loss.item():.4f}"
                    log_message(log_queue, log_msg)
                    writer.add_scalar('Loss/Train', loss.item(), global_step)

                # Image generation & logging
                if global_step > 0 and global_step % 1000 == 0:
                    model.eval()
                    with torch.no_grad():
                        # The start token for the transformer is vocab_size (an out-of-band token)
                        # We need to make sure this is handled in the model's sample method.
                        # Assuming model.py handles this correctly.
                        start_token = transformer_config['vocab_size']
                        sampled_indices = model.sample(n_samples=4, seq_len=transformer_config['block_size'],
                                                       start_token=start_token, device=device)
                        
                        sampled_indices = sampled_indices.view(-1, latent_h, latent_w)
                        generated_images = vqgan_model.decode_from_indices(sampled_indices)
                        
                        gen_grid = make_grid(generated_images.cpu(), nrow=4, normalize=True, value_range=(-1, 1))
                        # FIX 3: Use 'global_step' instead of undefined 'step'
                        writer.add_image('Generated/Samples', gen_grid, global_step=global_step)
                    model.train()

                global_step += 1

            if training_controls['stop']:
                break
            
            # Checkpointing
            if (epoch + 1) % config['save_epoch_freq'] == 0:
                ckpt_path = os.path.join(ckpt_dir, f'transformer_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'config': transformer_config
                }, ckpt_path)
                log_message(log_queue, f"Saved checkpoint to {ckpt_path}")

    except Exception as e:
        import traceback
        log_message(log_queue, f"An error occurred during training: {e}")
        log_message(log_queue, traceback.format_exc())
    finally:
        log_message(log_queue, "--- Transformer Training Finished ---")
        writer.close()
        gc.collect()
        torch.cuda.empty_cache()


# --- Unit Test ---
if __name__ == '__main__':
    def create_dummy_npz(path, num_samples=32, img_size=64):
        """Creates a dummy npz file for testing."""
        if os.path.exists(path):
            return
        print(f"Creating dummy dataset at {path}...")
        images = np.random.randint(0, 256, size=(num_samples, img_size, img_size, 3), dtype=np.uint8)
        attributes = np.random.randint(0, 2, size=(num_samples, 40)).astype(np.float32)
        np.savez(path, images=images, attributes=attributes)
        print("Dummy dataset created.")

    def test_training_runs():
        """Runs a few steps of training for both stages to check for errors."""
        dummy_data_path = 'dummy_celeba_cache_64x64.npz'
        create_dummy_npz(dummy_data_path)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Running Unit Tests on device: {device} ---")

        # Mock queue and controls
        log_q = queue.Queue()
        training_ctrl = {'stop': False}

        # --- Test Stage 1 ---
        print("\n--- Testing Stage 1 ---")
        stage1_config = {
            'data_path': dummy_data_path,
            'epochs': 1,
            'batch_size': 4,
            'lr': 1e-4,
            'n_embed': 512,
            'commitment_cost': 0.25,
            'gan_weight': 0.8,
            'save_epoch_freq': 1,
            'checkpoint_path': None
        }
        train_stage1(stage1_config, training_ctrl, log_q, device)
        
        # Check that a checkpoint was created
        exp_dirs = [d for d in os.listdir('checkpoints') if d.startswith('VQGAN')]
        assert len(exp_dirs) > 0, "Stage 1 did not create an experiment checkpoint directory."
        latest_exp_dir = sorted(exp_dirs, key=lambda d: os.path.getmtime(os.path.join('checkpoints', d)))[-1]
        ckpt_files = os.listdir(os.path.join('checkpoints', latest_exp_dir))
        assert 'vqgan_epoch_0.pt' in ckpt_files, "Stage 1 checkpoint file not found."
        print("Stage 1 ran and created checkpoint successfully.")
        vqgan_ckpt_path = os.path.join('checkpoints', latest_exp_dir, 'vqgan_epoch_0.pt')

        # --- Test Stage 2 ---
        print("\n--- Testing Stage 2 ---")
        stage2_config = {
            'data_path': dummy_data_path,
            'vqgan_checkpoint_path': vqgan_ckpt_path,
            'epochs': 1,
            'batch_size': 4,
            'lr': 1e-4,
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 256,
            'save_epoch_freq': 1,
            'checkpoint_path': None
        }
        train_stage2(stage2_config, training_ctrl, log_q, device)
        
        # Check that a checkpoint was created
        exp_dirs_t = [d for d in os.listdir('checkpoints') if d.startswith('Transformer')]
        assert len(exp_dirs_t) > 0, "Stage 2 did not create an experiment checkpoint directory."
        latest_exp_dir_t = sorted(exp_dirs_t, key=lambda d: os.path.getmtime(os.path.join('checkpoints', d)))[-1]
        ckpt_files_t = os.listdir(os.path.join('checkpoints', latest_exp_dir_t))
        assert 'transformer_epoch_0.pt' in ckpt_files_t, "Stage 2 checkpoint file not found."
        print("Stage 2 ran and created checkpoint successfully.")
        
        print("\n--- Unit Tests Passed ---")

        # Clean up
        import shutil
        if os.path.exists(dummy_data_path):
            os.remove(dummy_data_path)
        if os.path.exists('runs'):
            shutil.rmtree('runs')
        if os.path.exists('checkpoints'):
            shutil.rmtree('checkpoints')
        print("Cleaned up dummy files and directories.")

    test_training_runs()
