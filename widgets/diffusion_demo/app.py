from flask import Flask, render_template, request, jsonify, Response
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import requests
import io
import base64
import threading
import time
import os
from collections import deque # For storing logs

# --- MODIFIED IMPORT ---
# Import model and functions from model_dit.py
from model_dit import UNet, linear_beta_schedule, get_ddpm_params, q_sample, p_sample_loop

# --- App Config ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key'

# --- Global state ---
MODEL_INSTANCE = None
TRAINING_LOGS = deque(maxlen=200)
TRAINING_THREAD = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_PARAMS = None
MNIST_IMAGE_URL = "https://zsc.github.io/widgets/celeba/48x48.png"
MNIST_DATA_CACHE = None
IMAGE_SIZE = 48 # This will be our input_size for DiT. Must be divisible by patch_size.

# --- Helper: Logging ---
def log_message(message):
    print(message)
    TRAINING_LOGS.append(f"{time.strftime('%H:%M:%S')} - {message}")

# --- Data Loading ---
class StitchedMNISTDataset(Dataset):
    def __init__(self, image_url, img_size=28, transform=None, limit_images=None): # img_size default updated
        global MNIST_DATA_CACHE
        self.img_size = img_size
        self.transform = transform
        self.images = []

        if MNIST_DATA_CACHE is not None:
            log_message("Using cached MNIST data.")
            self.images = MNIST_DATA_CACHE
            if limit_images and len(self.images) > limit_images:
                 self.images = self.images[:limit_images]
            return

        log_message(f"Downloading stitched MNIST from {image_url}...")
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img_bytes = io.BytesIO(response.content)
            stitched_img = Image.open(img_bytes)#.convert('L')
            stitched_arr = np.array(stitched_img)
            log_message(f"Stitched image loaded: {stitched_arr.shape}")

            total_rows_of_images = stitched_arr.shape[0] // img_size
            total_cols_of_images = stitched_arr.shape[1] // img_size
            
            num_extracted = 0
            for r_idx in range(total_rows_of_images):
                for c_idx in range(total_cols_of_images):
                    if limit_images and num_extracted >= limit_images:
                        break
                    start_y = r_idx * img_size
                    end_y = start_y + img_size
                    start_x = c_idx * img_size
                    end_x = start_x + img_size
                    single_img_arr = stitched_arr[start_y:end_y, start_x:end_x]
                    self.images.append(Image.fromarray(single_img_arr))
                    num_extracted += 1
                if limit_images and num_extracted >= limit_images:
                    break
            
            MNIST_DATA_CACHE = self.images
            log_message(f"Extracted {len(self.images)} individual MNIST images.")

        except requests.exceptions.RequestException as e:
            log_message(f"Error downloading/processing MNIST: {e}")
            raise
        except Exception as e:
            log_message(f"Error processing image: {e}")
            raise

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloader(batch_size, img_size=28, limit_dataset_size=None): # img_size default updated
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    if limit_dataset_size is None:
        limit_dataset_size = 10000 
    
    dataset = StitchedMNISTDataset(MNIST_IMAGE_URL, img_size=img_size, limit_images=limit_dataset_size, transform=transform)
    if not dataset.images: 
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


# --- Training Loop ---
def train_model_thread(hyperparams):
    global MODEL_INSTANCE, DDPM_PARAMS, DEVICE, IMAGE_SIZE
    log_message("DiT Training started with params: " + str(hyperparams))
    
    lr = hyperparams.get('learning_rate', 1e-4)
    epochs = hyperparams.get('epochs', 20) # Increased default
    batch_size = hyperparams.get('batch_size', 64)
    timesteps = hyperparams.get('timesteps', 200) 
    limit_dataset_size = hyperparams.get('limit_dataset_size', 10000) # Increased default

    # --- DiT Specific Hyperparameters ---
    patch_size = hyperparams.get('patch_size', 4) # e.g., 28x28 image, 4x4 patches -> 7x7 grid of patches
    hidden_size = hyperparams.get('hidden_size', 256) # Transformer embedding dimension
    depth = hyperparams.get('depth', 6) # Number of DiT blocks
    num_heads = hyperparams.get('num_heads', 4) # Attention heads
    # time_emb_dim often related to hidden_size, e.g., hidden_size or hidden_size // 4, etc.
    # The DiT model's TimeEmbedding uses time_emb_dim // 4 internally for its first linear layer.
    # Let's keep time_emb_dim flexible or tie it to hidden_size.
    # For now, let's set it to hidden_size, as DiT's adaLN_modulation takes time_emb_dim.
    time_emb_dim = hyperparams.get('time_emb_dim', hidden_size)

    if IMAGE_SIZE % patch_size != 0:
        log_message(f"Error: IMAGE_SIZE ({IMAGE_SIZE}) must be divisible by patch_size ({patch_size}). Training aborted.")
        return

    log_message(f"Using device: {DEVICE}")

    betas = linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02).to(DEVICE)
    DDPM_PARAMS = get_ddpm_params(betas)
    # Ensure all DDPM_PARAMS tensors are on the correct device
    for key in DDPM_PARAMS: 
        if isinstance(DDPM_PARAMS[key], torch.Tensor):
            DDPM_PARAMS[key] = DDPM_PARAMS[key].to(DEVICE)

    # --- Instantiate DiT model ---
    model = UNet( # Class name is UNet, but it's the DiT implementation
        image_channels=3, 
        input_size=IMAGE_SIZE,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        time_emb_dim=time_emb_dim # Passed to TimeEmbedding and DiTBlocks
    ).to(DEVICE)
    MODEL_INSTANCE = model 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Dataloader uses IMAGE_SIZE
    dataloader = get_dataloader(batch_size, img_size=IMAGE_SIZE, limit_dataset_size=limit_dataset_size)
    if dataloader is None:
        log_message("Failed to load data. Training aborted.")
        return

    log_message(f"Starting DiT training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            x_start = batch.to(DEVICE) 
            batch_size_current = x_start.shape[0]

            t = torch.randint(0, timesteps, (batch_size_current,), device=DEVICE).long()
            
            noise = torch.randn_like(x_start)
            # q_sample needs DDPM_PARAMS to be on the same device as t.
            # DDPM_PARAMS are already moved to DEVICE. q_sample internally handles device matching if needed.
            x_t = q_sample(x_start, t, DDPM_PARAMS, noise) 
            
            predicted_noise = model(x_t, t.float()) # DiT's forward method
            
            loss = F.mse_loss(noise, predicted_noise)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                log_message(f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        log_message(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
    
    log_message("DiT Training finished.")
    MODEL_INSTANCE = model

# --- Flask Routes ---
@app.route('/')
def index():
    TRAINING_LOGS.clear() 
    log_message("DiT UI loaded. Ready for training.")
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global TRAINING_THREAD
    if TRAINING_THREAD and TRAINING_THREAD.is_alive():
        return jsonify({"status": "error", "message": "Training is already in progress."}), 400

    try:
        hyperparams = request.json
        TRAINING_LOGS.clear() 
        
        TRAINING_THREAD = threading.Thread(target=train_model_thread, args=(hyperparams,))
        TRAINING_THREAD.start()
        
        return jsonify({"status": "success", "message": "DiT Training started."})
    except Exception as e:
        log_message(f"Error starting DiT training: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@app.route('/get_logs')
def get_logs():
    return jsonify({"logs": list(TRAINING_LOGS)})

@app.route('/generate_sample')
def generate_sample():
    global MODEL_INSTANCE, DDPM_PARAMS, DEVICE, IMAGE_SIZE
    if MODEL_INSTANCE is None or DDPM_PARAMS is None:
        return jsonify({"status": "error", "message": "Model not trained or DDPM params not set."}), 400

    try:
        num_samples_to_generate = 5
        log_message(f"Generating {num_samples_to_generate} sample images with DiT...")
        MODEL_INSTANCE.eval() 
        
        timesteps = DDPM_PARAMS['betas'].shape[0] if DDPM_PARAMS else 200 
        shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE) # Generate 1 image at a time for the loop
        
        # Ensure DDPM_PARAMS tensors are on the correct device for p_sample_loop
        for key in DDPM_PARAMS:
            if isinstance(DDPM_PARAMS[key], torch.Tensor):
                 DDPM_PARAMS[key] = DDPM_PARAMS[key].to(DEVICE)

        image_data_list = []
        for i in range(num_samples_to_generate):
            log_message(f"Generating sample {i+1}/{num_samples_to_generate}...")
            # p_sample_loop expects DDPM_PARAMS that p_sample can use (device handling inside p_sample)
            generated_img_tensor, _ = p_sample_loop(MODEL_INSTANCE, shape, timesteps, DDPM_PARAMS, DEVICE)
            
            generated_img_tensor = (generated_img_tensor + 1) / 2.0 
            generated_img_tensor = generated_img_tensor.clamp(0, 1) 
            
            pil_img = T.ToPILImage()(generated_img_tensor.squeeze(0).cpu()) # Ensure CPU tensor for ToPILImage

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
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("<h1>Hello from Flask! DiT UI will be here.</h1>") # Placeholder
    
    log_message(f"Flask app (DiT version) starting. IMAGE_SIZE={IMAGE_SIZE}. Open http://127.0.0.1:5001")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False) # use_reloader=False to avoid issues with global state and threads in debug mode
