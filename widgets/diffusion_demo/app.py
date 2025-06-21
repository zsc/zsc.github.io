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

# Import model and functions
from model_unet import UNet, linear_beta_schedule, get_ddpm_params, q_sample, p_sample_loop

# --- App Config ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key' # For potential session use later

# --- Global state (for simplicity in a demo) ---
MODEL_INSTANCE = None
TRAINING_LOGS = deque(maxlen=200) # Store last N log messages
TRAINING_THREAD = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDPM_PARAMS = None # To store precomputed DDPM schedule parameters
MNIST_IMAGE_URL = "https://zsc.github.io/widgets/mnist/mnist_train_stitched.png"
MNIST_DATA_CACHE = None # Cache for loaded MNIST images
IMAGE_SIZE = 28
# CHANNELS_PER_IMAGE_ROW = 245 # <<< FIXED: Removed unused global variable

# --- Helper: Logging ---
def log_message(message):
    print(message) # For server console
    TRAINING_LOGS.append(f"{time.strftime('%H:%M:%S')} - {message}")

# --- Data Loading ---
class StitchedMNISTDataset(Dataset):
    # <<< FIXED: Removed unused 'images_per_row' parameter
    def __init__(self, image_url, img_size=14, transform=None, limit_images=None):
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
            stitched_img = Image.open(img_bytes).convert('L') # Grayscale
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
            
            MNIST_DATA_CACHE = self.images # Cache it
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

def get_dataloader(batch_size, img_size=14, limit_dataset_size=None):
    transform = T.Compose([
        T.ToTensor(),                # Converts to [0, 1] tensor
        T.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for tanh activation often used in generators
    ])
    if limit_dataset_size is None:
        limit_dataset_size = 10000 
    
    # Note: 'images_per_row' is no longer passed as it was removed from StitchedMNISTDataset constructor
    dataset = StitchedMNISTDataset(MNIST_IMAGE_URL, img_size=img_size, limit_images=limit_dataset_size, transform=transform)
    if not dataset.images: 
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


# --- Training Loop ---
def train_model_thread(hyperparams):
    global MODEL_INSTANCE, DDPM_PARAMS, DEVICE
    log_message("Training started with params: " + str(hyperparams))
    
    lr = hyperparams.get('learning_rate', 1e-4)
    epochs = hyperparams.get('epochs', 5)
    batch_size = hyperparams.get('batch_size', 64)
    timesteps = hyperparams.get('timesteps', 200) 
    unet_n_channels = hyperparams.get('unet_n_channels', 32)
    unet_ch_mults = tuple(hyperparams.get('unet_ch_mults', [1, 2])) 
    limit_dataset_size = hyperparams.get('limit_dataset_size', 5000)

    log_message(f"Using device: {DEVICE}")

    betas = linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02).to(DEVICE)
    DDPM_PARAMS = get_ddpm_params(betas)
    for key in DDPM_PARAMS: 
        DDPM_PARAMS[key] = DDPM_PARAMS[key].to(DEVICE)

    model = UNet(
        image_channels=1, 
        n_channels=unet_n_channels, 
        ch_mults=unet_ch_mults, 
        time_emb_dim=unet_n_channels * 4
    ).to(DEVICE)
    MODEL_INSTANCE = model 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    dataloader = get_dataloader(batch_size, img_size=IMAGE_SIZE, limit_dataset_size=limit_dataset_size)
    if dataloader is None:
        log_message("Failed to load data. Training aborted.")
        return

    log_message(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            x_start = batch.to(DEVICE) 
            batch_size_current = x_start.shape[0]

            t = torch.randint(0, timesteps, (batch_size_current,), device=DEVICE).long()
            
            noise = torch.randn_like(x_start)
            x_t = q_sample(x_start, t, DDPM_PARAMS, noise)
            
            predicted_noise = model(x_t, t.float()) 
            
            loss = F.mse_loss(noise, predicted_noise)
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                log_message(f"Epoch {epoch+1}/{epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        log_message(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")
    
    log_message("Training finished.")
    MODEL_INSTANCE = model 

# --- Flask Routes ---
@app.route('/')
def index():
    TRAINING_LOGS.clear() 
    log_message("UI loaded. Ready for training.")
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
        
        return jsonify({"status": "success", "message": "Training started."})
    except Exception as e:
        log_message(f"Error starting training: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@app.route('/get_logs')
def get_logs():
    return jsonify({"logs": list(TRAINING_LOGS)})

# <<< FIXED: Modified to generate multiple samples and return as a list
@app.route('/generate_sample')
def generate_sample():
    global MODEL_INSTANCE, DDPM_PARAMS, DEVICE
    if MODEL_INSTANCE is None or DDPM_PARAMS is None:
        return jsonify({"status": "error", "message": "Model not trained or DDPM params not set."}), 400

    try:
        num_samples_to_generate = 5 # Match HTML button/expectation
        log_message(f"Generating {num_samples_to_generate} sample images...")
        MODEL_INSTANCE.eval() 
        
        timesteps = DDPM_PARAMS['betas'].shape[0] if DDPM_PARAMS else 200 
        shape = (1, 1, IMAGE_SIZE, IMAGE_SIZE) 
        
        for key in DDPM_PARAMS:
            if isinstance(DDPM_PARAMS[key], torch.Tensor):
                 DDPM_PARAMS[key] = DDPM_PARAMS[key].to(DEVICE)

        image_data_list = []
        for i in range(num_samples_to_generate):
            log_message(f"Generating sample {i+1}/{num_samples_to_generate}...")
            # p_sample_loop returns image on CPU
            generated_img_tensor, _ = p_sample_loop(MODEL_INSTANCE, shape, timesteps, DDPM_PARAMS, DEVICE)
            
            generated_img_tensor = (generated_img_tensor + 1) / 2.0 
            generated_img_tensor = generated_img_tensor.clamp(0, 1) 
            
            # ToPILImage expects a CPU tensor
            pil_img = T.ToPILImage()(generated_img_tensor.squeeze(0)) 

            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data_list.append(img_str)
        
        log_message(f"{num_samples_to_generate} sample images generated.")
        # Return list under "image_data_list" key
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
            f.write("<h1>Hello from Flask! Put your UI here.</h1>")
    
    log_message(f"Flask app starting. Open http://127.0.0.1:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
