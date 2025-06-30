# app.py
from flask import Flask, render_template, request, jsonify, Response
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from PIL import Image
import numpy as np
import io
import base64
import threading
import time
import os
from collections import deque
from pathlib import Path
from copy import deepcopy

# Import model and functions
from model_unet import UNet, Discriminator, linear_beta_schedule, get_ddpm_params, q_sample, p_sample_loop
# <<< NEW: Import VGG for Perception Loss >>>
from torchvision.models import vgg16, VGG16_Weights

# --- App Config ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'

# --- Global state ---
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)
MODEL_INSTANCE = {'unet': None, 'discriminator': None}
TRAINED_MODEL_CONFIG = {}
TRAINING_LOGS = deque(maxlen=500)
TRAINING_THREAD = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_PARAMS = None

# <<< NEW: For live generation during training >>>
LIVE_MODEL_SNAPSHOT = {
    'state_dict': None,
    'config': None,
    'ddpm_params': None,
    'epoch': -1,
    'lock': threading.Lock()
}


# --- EMA Helper Class (Unchanged) ---
class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]
        self.original = {}


# <<< NEW: VGG Perceptual Loss Class >>>
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Use new recommended weights parameter
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # Define feature layers to use
        self.slice1 = torch.nn.Sequential(*[vgg[i] for i in range(4)]).eval()  # relu1_2
        self.slice2 = torch.nn.Sequential(*[vgg[i] for i in range(4, 9)]).eval()  # relu2_2
        self.slice3 = torch.nn.Sequential(*[vgg[i] for i in range(9, 16)]).eval() # relu3_3
        
        for p in self.parameters():
            p.requires_grad = False
            
        self.transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = resize

    def forward(self, input_img, target_img):
        # Input images are expected to be in range [-1, 1]
        # Rescale to [0, 1] for VGG
        input_img = (input_img + 1) / 2
        target_img = (target_img + 1) / 2
        
        # Apply VGG's required normalization
        input_img = self.transform(input_img)
        target_img = self.transform(target_img)
        
        if self.resize:
            input_img = F.interpolate(input_img, size=(224, 224), mode='bilinear', align_corners=False)
            target_img = F.interpolate(target_img, size=(224, 224), mode='bilinear', align_corners=False)

        input_feat1 = self.slice1(input_img)
        target_feat1 = self.slice1(target_img).detach()
        input_feat2 = self.slice2(input_feat1)
        target_feat2 = self.slice2(target_feat1).detach()
        input_feat3 = self.slice3(input_feat2)
        target_feat3 = self.slice3(target_feat2).detach()

        loss = F.l1_loss(input_feat1, target_feat1)
        loss += F.l1_loss(input_feat2, target_feat2)
        loss += F.l1_loss(input_feat3, target_feat3)
        return loss

# --- Helper: Logging (Unchanged) ---
def log_message(message):
    print(message)
    TRAINING_LOGS.append(f"{time.strftime('%H:%M:%S')} - {message}")

# --- Data Loading (Unchanged) ---
class CelebADataset(Dataset):
    def __init__(self, npz_path, transform=None):
        self.transform = transform
        self.images = []
        self.attributes = []
        try:
            log_message(f"Loading dataset from {npz_path}...")
            with np.load(npz_path) as data:
                if 'images' not in data or 'attributes' not in data:
                    raise KeyError("'.npz' file must contain 'images' and 'attributes' arrays.")
                self.images = data['images']
                self.attributes = (data['attributes'] + 1) / 2
            log_message(f"Dataset loaded with {len(self.images)} images and attributes.")
        except FileNotFoundError: log_message(f"ERROR: Dataset file not found at {npz_path}"); raise
        except Exception as e: log_message(f"ERROR: Failed to load or parse .npz file: {e}"); raise
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        image_np = self.images[idx]; attrs = torch.tensor(self.attributes[idx], dtype=torch.float32)
        if self.transform: image = self.transform(image_np)
        return image, attrs
def get_dataloader(batch_size, img_size, dataset_path, limit_dataset_size=None):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    try: dataset = CelebADataset(npz_path=dataset_path, transform=transform)
    except (FileNotFoundError, Exception): return None
    if limit_dataset_size and limit_dataset_size < len(dataset):
        log_message(f"Limiting dataset to {limit_dataset_size} images.")
        dataset = Subset(dataset, range(limit_dataset_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


# --- Checkpoint Saving Function (Unchanged) ---
def save_checkpoint(epoch, unet, g_optimizer, config, experiment_name, ema=None, discriminator=None, d_optimizer=None):
    current_config = deepcopy(config)
    checkpoint = {
        'epoch': epoch, 'unet_state_dict': unet.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(), 'config': current_config,
    }
    if ema:
        ema.apply_shadow(unet)
        checkpoint['ema_unet_state_dict'] = unet.state_dict()
        ema.restore(unet)
    if discriminator and d_optimizer:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
        checkpoint['d_optimizer_state_dict'] = d_optimizer.state_dict()
    checkpoint_filename = f"ckpt_{experiment_name}_epoch_{epoch}.pt"
    checkpoint_path = MODELS_DIR / checkpoint_filename
    log_message(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)


# --- Training Loop (HEAVILY MODIFIED for Universal Advanced Losses & Live Snapshot) ---
def train_model_thread(hyperparams):
    global MODEL_INSTANCE, DDPM_PARAMS, DEVICE, TRAINED_MODEL_CONFIG, LIVE_MODEL_SNAPSHOT
    
    # --- Base Hyperparameters ---
    lr = hyperparams.get('learning_rate', 1e-4)
    epochs = hyperparams.get('epochs', 20)
    batch_size = hyperparams.get('batch_size', 32)
    timesteps = hyperparams.get('timesteps', 1000)
    unet_n_channels = hyperparams.get('unet_n_channels', 64)
    unet_ch_mults_str = hyperparams.get('unet_ch_mults_str', '1,2,4')
    unet_ch_mults = tuple(map(int, unet_ch_mults_str.split(',')))
    limit_dataset_size = hyperparams.get('limit_dataset_size', 10000)
    dataset_path = hyperparams.get('dataset_path')
    img_size = hyperparams.get('img_size')
    
    # --- Advanced Training Params ---
    training_mode = hyperparams.get('training_mode', 'unconditional')
    cond_drop_prob = hyperparams.get('cond_drop_prob', 0.1)
    ema_decay = hyperparams.get('ema_decay', 0.999)
    save_every_epochs = hyperparams.get('save_every_epochs', 5)
    
    # <<< MODIFIED: Advanced loss params are now universal >>>
    start_gan_epoch = hyperparams.get('start_gan_epoch', 0)
    gan_loss_weight = hyperparams.get('gan_loss_weight', 0.0)
    start_perception_epoch = hyperparams.get('start_perception_epoch', 0)
    perception_loss_weight = hyperparams.get('perception_loss_weight', 0.0)
    lr_scheduler_type = hyperparams.get('lr_scheduler_type', 'none')

    n_classes = hyperparams.get('n_classes', 40)
    class_emb_dim = hyperparams.get('class_emb_dim', 128)

    log_message("Training started with params: " + str(hyperparams))
    log_message(f"Using device: {DEVICE}")

    # --- Setup ---
    run_ts = time.strftime('%Y%m%d-%H%M%S')
    ch_mults_name = ''.join(map(str, unet_ch_mults))
    experiment_name = (
        f"{run_ts}_{training_mode}_lr{lr:.0e}_bs{batch_size}_ds{limit_dataset_size}"
        f"_ch{unet_n_channels}_mults{ch_mults_name}_lrsched_{lr_scheduler_type}"
    )
    # <<< MODIFIED: Experiment name reflects advanced losses for ANY mode >>>
    if gan_loss_weight > 0 or perception_loss_weight > 0:
        experiment_name += f"_gan{gan_loss_weight}e{start_gan_epoch}_percept{perception_loss_weight}e{start_perception_epoch}"

    writer = SummaryWriter(log_dir=f'runs/{experiment_name}')
    use_amp = DEVICE.type == 'cuda'
    g_scaler = GradScaler(enabled=use_amp)
    d_scaler = GradScaler(enabled=use_amp)
    if use_amp: log_message("Using AMP for training.")
    
    betas = linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02).to(DEVICE)
    DDPM_PARAMS = get_ddpm_params(betas)

    # --- Models & Optimizers ---
    unet = UNet(
        image_channels=3, n_channels=unet_n_channels, ch_mults=unet_ch_mults, time_emb_dim=unet_n_channels*4,
        n_classes=n_classes if training_mode == 'conditional' else None,
        class_emb_dim=class_emb_dim if training_mode == 'conditional' else None,
        cond_drop_prob=cond_drop_prob if training_mode == 'conditional' else 0
    ).to(DEVICE)
    g_optimizer = optim.AdamW(unet.parameters(), lr=lr)
    ema = EMA(unet, decay=ema_decay)

    # <<< MODIFIED: Advanced Loss Components are now universal >>>
    discriminator, d_optimizer, vgg_loss = None, None, None
    bce_loss = torch.nn.BCEWithLogitsLoss()

    if gan_loss_weight > 0:
        log_message("GAN training enabled.")
        discriminator = Discriminator(
            n_classes=n_classes if training_mode == 'conditional' else None,
            class_emb_dim=class_emb_dim
        ).to(DEVICE)
        d_optimizer = optim.AdamW(discriminator.parameters(), lr=lr * 0.8)

    if perception_loss_weight > 0:
        log_message("Perception Loss enabled.")
        vgg_loss = VGGPerceptualLoss().to(DEVICE)

    dataloader = get_dataloader(batch_size, img_size, dataset_path, limit_dataset_size)
    if dataloader is None:
        log_message("Failed to load data. Training aborted."); writer.close(); return

    # <<< MODIFIED: LR Schedulers setup is now cleaner >>>
    total_steps = epochs * len(dataloader)
    g_scheduler, d_scheduler = None, None
    if lr_scheduler_type == 'cosine':
        log_message("Using Cosine Annealing LR scheduler.")
        g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=total_steps, eta_min=1e-6)
        if d_optimizer: d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=total_steps, eta_min=1e-6)
    elif lr_scheduler_type == 'linear':
        log_message("Using Linear LR scheduler.")
        g_scheduler = optim.lr_scheduler.LinearLR(g_optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps)
        if d_optimizer: d_scheduler = optim.lr_scheduler.LinearLR(d_optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_steps)

    config_to_save = hyperparams.copy()
    config_to_save['unet_ch_mults'] = unet_ch_mults

    # --- Training Loop ---
    global_step = 0
    try:
        log_message(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            unet.train()
            if discriminator: discriminator.train()
            
            epoch_g_loss, epoch_d_loss = 0, 0
            
            for step, (x_start, attributes) in enumerate(dataloader):
                x_start = x_start.to(DEVICE)
                attributes = attributes.to(DEVICE)
                batch_size_current = x_start.shape[0]
                t = torch.randint(0, timesteps, (batch_size_current,), device=DEVICE).long()
                
                # Use class labels only for conditional training
                class_labels = attributes if training_mode == 'conditional' else None
                
                # --- Generator Training Step ---
                g_optimizer.zero_grad()
                with autocast(enabled=use_amp):
                    noise = torch.randn_like(x_start)
                    x_t = q_sample(x_start, t, DDPM_PARAMS, noise)
                    predicted_noise = unet(x_t, t.float(), class_labels)
                    mse_loss = F.mse_loss(noise, predicted_noise)
                    g_loss = mse_loss
                    
                    gan_loss, perception_loss, d_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

                    # --- Advanced Loss Calculation (GAN / Perceptual) ---
                    if (discriminator and epoch >= start_gan_epoch) or (vgg_loss and epoch >= start_perception_epoch):
                        sqrt_alpha_prod_t = DDPM_PARAMS["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_prod_t = DDPM_PARAMS["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
                        predicted_x0 = (x_t - sqrt_one_minus_alpha_prod_t * predicted_noise) / sqrt_alpha_prod_t
                        
                        if discriminator and epoch >= start_gan_epoch:
                            d_fake_pred = discriminator(predicted_x0, t.float(), class_labels)
                            gan_loss = bce_loss(d_fake_pred, torch.ones_like(d_fake_pred))
                            g_loss += gan_loss_weight * gan_loss

                        if vgg_loss and epoch >= start_perception_epoch:
                            perception_loss = vgg_loss(predicted_x0, x_start)
                            g_loss += perception_loss_weight * perception_loss

                g_scaler.scale(g_loss).backward()
                g_scaler.step(g_optimizer)
                g_scaler.update()
                if g_scheduler: g_scheduler.step()
                
                ema.update(unet)
                epoch_g_loss += g_loss.item()
                
                # --- Discriminator Training Step (if enabled) ---
                if discriminator and epoch >= start_gan_epoch:
                    d_optimizer.zero_grad()
                    with autocast(enabled=use_amp):
                        d_real_pred = discriminator(x_start.detach(), t.float(), class_labels)
                        d_loss_real = bce_loss(d_real_pred, torch.ones_like(d_real_pred))
                        d_fake_pred = discriminator(predicted_x0.detach(), t.float(), class_labels)
                        d_loss_fake = bce_loss(d_fake_pred, torch.zeros_like(d_fake_pred))
                        d_loss = (d_loss_real + d_loss_fake) / 2
                    
                    d_scaler.scale(d_loss).backward()
                    d_scaler.step(d_optimizer)
                    d_scaler.update()
                    if d_scheduler: d_scheduler.step()
                    epoch_d_loss += d_loss.item()

                if (global_step + 1) % 50 == 0:
                    current_lr = g_optimizer.param_groups[0]['lr']
                    log_msg = f"E {epoch+1}, S {step+1}/{len(dataloader)}, G_Loss: {g_loss.item():.4f}, LR: {current_lr:.2e}"
                    if discriminator and epoch >= start_gan_epoch:
                        log_msg += f", D_Loss: {d_loss.item():.4f}"
                    log_message(log_msg)
                
                # --- Logging to TensorBoard ---
                writer.add_scalar('Loss/G_Loss_Total', g_loss.item(), global_step)
                writer.add_scalar('Loss/G_MSE_Loss', mse_loss.item(), global_step)
                writer.add_scalar('Learning_Rate/Generator', g_optimizer.param_groups[0]['lr'], global_step)
                if discriminator and epoch >= start_gan_epoch:
                    writer.add_scalar('Loss/D_Loss', d_loss.item(), global_step)
                    writer.add_scalar('Loss/G_GAN_Loss', gan_loss.item(), global_step)
                if vgg_loss and epoch >= start_perception_epoch:
                    writer.add_scalar('Loss/G_Perception_Loss', perception_loss.item(), global_step)
                global_step += 1
            
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader) if discriminator and epoch >= start_gan_epoch else 0
            log_message(f"Epoch {epoch+1} finished. Avg G_Loss: {avg_g_loss:.4f}, Avg D_Loss: {avg_d_loss:.4f}")
            
            # --- Evaluation & Checkpointing Logic ---
            ema.apply_shadow(unet)
            unet.eval()
            with torch.no_grad():
                shape = (8, 3, img_size, img_size)
                generated_imgs_tensor, _ = p_sample_loop(unet, shape, timesteps, DDPM_PARAMS, DEVICE)
                grid = torchvision.utils.make_grid((generated_imgs_tensor + 1) / 2.0)
                writer.add_image('Generated Samples (EMA)', grid, epoch)
            
            # <<< NEW: Update live snapshot for generation during training >>>
            with LIVE_MODEL_SNAPSHOT['lock']:
                log_message(f"Updating live model snapshot at epoch {epoch+1}")
                state_dict_cpu = {k: v.cpu() for k, v in unet.state_dict().items()}
                LIVE_MODEL_SNAPSHOT['state_dict'] = state_dict_cpu
                LIVE_MODEL_SNAPSHOT['config'] = deepcopy(config_to_save)
                LIVE_MODEL_SNAPSHOT['ddpm_params'] = deepcopy(DDPM_PARAMS)
                LIVE_MODEL_SNAPSHOT['epoch'] = epoch + 1
            
            unet.train()
            ema.restore(unet)

            if (epoch + 1) % save_every_epochs == 0 and (epoch + 1) < epochs:
                save_checkpoint(epoch + 1, unet, g_optimizer, config_to_save, experiment_name,
                                ema=ema, discriminator=discriminator, d_optimizer=d_optimizer)

    except Exception as e:
        import traceback
        log_message(f"An error occurred during training: {e}")
        traceback.print_exc()
    finally:
        log_message("Training finished.")
        writer.close()
        
        ema.apply_shadow(unet)
        MODEL_INSTANCE['unet'] = unet
        MODEL_INSTANCE['discriminator'] = discriminator
        TRAINED_MODEL_CONFIG = config_to_save
        
        save_checkpoint(epochs, unet, g_optimizer, config_to_save, experiment_name,
                        ema=ema, discriminator=discriminator, d_optimizer=d_optimizer)

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global TRAINING_THREAD
    if TRAINING_THREAD and TRAINING_THREAD.is_alive():
        return jsonify({"status": "error", "message": "Training is already in progress."}), 400
    try:
        hyperparams = request.json
        TRAINING_LOGS.clear()
        # <<< NEW: Clear previous live snapshot >>>
        with LIVE_MODEL_SNAPSHOT['lock']:
            LIVE_MODEL_SNAPSHOT['state_dict'] = None
            LIVE_MODEL_SNAPSHOT['config'] = None
            LIVE_MODEL_SNAPSHOT['ddpm_params'] = None
            LIVE_MODEL_SNAPSHOT['epoch'] = -1
        
        TRAINING_THREAD = threading.Thread(target=train_model_thread, args=(hyperparams,))
        TRAINING_THREAD.start()
        return jsonify({"status": "success", "message": "Training started."})
    except Exception as e:
        log_message(f"Error starting training: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@app.route('/get_logs')
def get_logs():
    return jsonify({"logs": list(TRAINING_LOGS)})

@app.route('/get_checkpoints')
def get_checkpoints():
    try:
        checkpoints = sorted([f.name for f in MODELS_DIR.glob('*.pt')], reverse=True)
        return jsonify({"status": "success", "checkpoints": checkpoints})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    global MODEL_INSTANCE, DDPM_PARAMS, TRAINED_MODEL_CONFIG
    try:
        data = request.json
        filename = data.get('filename')
        if not filename:
            return jsonify({"status": "error", "message": "Filename not provided"}), 400

        filepath = MODELS_DIR / filename
        if not filepath.exists():
            return jsonify({"status": "error", "message": "Checkpoint not found"}), 404
        
        log_message(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filepath, map_location=DEVICE)
        config = checkpoint['config']
        
        unet = UNet(
            image_channels=3,
            n_channels=config.get('unet_n_channels', 64),
            ch_mults=tuple(config.get('unet_ch_mults', [1, 2, 4])),
            time_emb_dim=config.get('unet_n_channels', 64) * 4,
            n_classes=config.get('n_classes') if config.get('training_mode') == 'conditional' else None,
            class_emb_dim=config.get('class_emb_dim') if config.get('training_mode') == 'conditional' else None
        ).to(DEVICE)
        
        if 'ema_unet_state_dict' in checkpoint:
            log_message("Loading EMA weights for generation.")
            unet.load_state_dict(checkpoint['ema_unet_state_dict'])
        else:
            log_message("Warning: EMA weights not found in checkpoint. Loading standard training weights.")
            unet.load_state_dict(checkpoint['unet_state_dict'])

        unet.eval()
        MODEL_INSTANCE['unet'] = unet
        
        if 'discriminator_state_dict' in checkpoint:
            log_message("Discriminator weights found in checkpoint (not loaded for generation).")

        TRAINED_MODEL_CONFIG = config
        betas = linear_beta_schedule(config['timesteps']).to(DEVICE)
        DDPM_PARAMS = get_ddpm_params(betas)

        log_message(f"Checkpoint loaded successfully. Model is ready for generation.")
        return jsonify({"status": "success", "message": "Checkpoint loaded.", "config": config})

    except Exception as e:
        log_message(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate_sample', methods=['POST'])
def generate_sample():
    # <<< MODIFIED: Handle live model snapshot and loaded model >>>
    global LIVE_MODEL_SNAPSHOT, MODEL_INSTANCE, DDPM_PARAMS, TRAINED_MODEL_CONFIG, TRAINING_THREAD
    
    unet_instance = None
    config = None
    ddpm_params = None
    
    # Prioritize live model if training is active
    is_training = TRAINING_THREAD and TRAINING_THREAD.is_alive()
    if is_training:
        with LIVE_MODEL_SNAPSHOT['lock']:
            if LIVE_MODEL_SNAPSHOT['state_dict']:
                log_message(f"Generating from live training snapshot (Epoch {LIVE_MODEL_SNAPSHOT['epoch']}).")
                config = LIVE_MODEL_SNAPSHOT['config']
                ddpm_params = LIVE_MODEL_SNAPSHOT['ddpm_params']
                
                # Create a new model instance on-the-fly to load the snapshot state
                unet_instance = UNet(
                    image_channels=3,
                    n_channels=config.get('unet_n_channels', 64),
                    ch_mults=tuple(config.get('unet_ch_mults', [1, 2, 4])),
                    time_emb_dim=config.get('unet_n_channels', 64) * 4,
                    n_classes=config.get('n_classes') if config.get('training_mode') == 'conditional' else None,
                    class_emb_dim=config.get('class_emb_dim') if config.get('training_mode') == 'conditional' else None
                ).to(DEVICE)
                
                # Load state dict from the snapshot (it's already on CPU)
                unet_instance.load_state_dict(LIVE_MODEL_SNAPSHOT['state_dict'])
                unet_instance.eval()

    # Fallback to the conventionally loaded model if no live one is available
    if unet_instance is None:
        log_message("Generating from pre-loaded checkpoint.")
        if not MODEL_INSTANCE.get('unet') or not TRAINED_MODEL_CONFIG:
            msg = "No model available. Train a model, or load a checkpoint."
            if is_training:
                msg += " (Live model snapshot not ready yet, please wait for the first epoch to complete)."
            return jsonify({"status": "error", "message": msg}), 400
        
        unet_instance = MODEL_INSTANCE['unet']
        config = TRAINED_MODEL_CONFIG
        ddpm_params = DDPM_PARAMS

    try:
        data = request.json
        attributes = data.get('attributes')
        guidance_scale = float(data.get('guidance_scale', 7.5))

        num_samples_to_generate = 4
        log_message(f"Generating {num_samples_to_generate} samples with guidance {guidance_scale}...")
        
        # Use the config from the active model (live or loaded)
        img_size = config['img_size']
        timesteps = config['timesteps']
        training_mode = config.get('training_mode', 'unconditional')
        
        class_labels = None
        if training_mode == 'conditional':
            if attributes and len(attributes) == config.get('n_classes'):
                class_labels = torch.tensor(attributes, dtype=torch.float32).unsqueeze(0).repeat(num_samples_to_generate, 1).to(DEVICE)
                log_message(f"Using attributes: {attributes}")
            else:
                log_message("Warning: Conditional model but no attributes provided. Generating unconditionally.")
        
        shape = (num_samples_to_generate, 3, img_size, img_size)
        
        # Use the determined unet_instance, ddpm_params for generation
        generated_imgs_tensor, _ = p_sample_loop(
            unet_instance, shape, timesteps, ddpm_params, DEVICE,
            class_labels=class_labels, guidance_scale=guidance_scale
        )
        
        image_data_list = []
        for img_tensor in generated_imgs_tensor:
            img_tensor = (img_tensor.cpu() + 1) / 2.0 
            img_tensor = img_tensor.clamp(0, 1) 
            pil_img = T.ToPILImage()(img_tensor)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data_list.append(img_str)
        
        log_message(f"{num_samples_to_generate} sample images generated.")
        return jsonify({"status": "success", "image_data_list": image_data_list})

    except Exception as e:
        log_message(f"Error generating sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error generating sample: {str(e)}"}), 500


if __name__ == '__main__':
    log_message(f"Flask app starting on http://127.0.0.1:5001 with device: {DEVICE}")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
