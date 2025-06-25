from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import threading
import traceback
import torch
from PIL import Image
import io
import numpy as np
from torchvision import transforms

from backend.bpe_trainer import train_bpe_tokenizer, load_tokenizer as load_bpe_tokenizer, TOKENIZER_FILE as BPE_TOKENIZER_FILE, BPE_MODEL_DIR, DEFAULT_VOCAB_SIZE
from backend.data_utils import get_dataloaders, ensure_data_exists, SPRITE_PATH, CAPTIONS_PATH, parse_sprite_sheet, load_captions, IMG_WIDTH, IMG_HEIGHT
from backend.trainer import Trainer
from backend.model_clstm import ConvLSTMModel
from backend.model_vit import ViTImageCaptionModel
from backend.model_mllm import MLLM

# Ensure paths are relative to this app.py file if needed, or use absolute paths.
# Flask serves static files from a 'static' folder by default.
# Templates are served from a 'templates' folder.
# We need to adjust for `../templates` relative to `backend/app.py`
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

# --- Configuration ---
# These would ideally come from a config file or environment variables
# For demo, keep them here.
MODEL_CONFIGS = {
    "clstm": {
        "embed_size": 256,
        "hidden_size": 512, # Should match encoder_feature_size
        "encoder_feature_size": 512,
        "num_layers": 1, # For LSTMDecoder
        "dropout": 0.1,
    },
    "vit": {
        "embed_size": 512, # ViT output projected to this, and decoder internal dim
        "num_heads": 8,
        "num_decoder_layers": 4, # Fewer layers for faster demo
        "decoder_ff_dim": 1024, # Smaller FF dim
        "dropout": 0.1,
        "vit_model_name": 'vit_small_patch16_224', # Smaller ViT
    },
    "mllm": {
        "llm_embed_dim": 768, # For BPE tokens, must match llm_hidden_size
        "llm_hidden_size": 768, # GPT-2 base
        "gpt2_model_name": "gpt2",
        "vision_encoder_name": 'vit_small_patch16_224', # Smaller ViT
        # "num_vision_tokens": 16, # Determined by ViT patches
    }
}
DEFAULT_TRAINING_CONFIG = {
    "lr": 1e-4,
    "epochs": 5, # Small for demo
    "batch_size": 32, # Adjust based on GPU memory
    "bfloat16": True, # User configurable
    "max_caption_len": 200, # Used in dataset and generation
    "test_split_ratio": 0.1,
    "num_workers": 2,
    "bpe_vocab_size": DEFAULT_VOCAB_SIZE # From bpe_trainer
}
CHECKPOINT_DIR = "./backend/checkpoints" # Checkpoints saved here
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BPE_MODEL_DIR, exist_ok=True)


# --- Global state (use with caution in web apps, fine for demo) ---
training_thread = None
training_status = {"running": False, "log": "Not started."}
current_model_loaded_for_inference = None
current_tokenizer_loaded_for_inference = None
current_model_type_for_inference = None

# --- Helper Functions ---
def get_model_instance(model_type, vocab_size, tokenizer_instance, for_training=True):
    """Instantiates a model based on type."""
    cfg = MODEL_CONFIGS.get(model_type)
    if not cfg:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_type == "clstm":
        model = ConvLSTMModel(
            embed_size=cfg["embed_size"],
            hidden_size=cfg["hidden_size"],
            vocab_size=vocab_size,
            encoder_feature_size=cfg["encoder_feature_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            tokenizer=tokenizer_instance # For generation
        )
    elif model_type == "vit":
        model = ViTImageCaptionModel(
            vocab_size=vocab_size,
            embed_size=cfg["embed_size"],
            num_heads=cfg["num_heads"],
            num_decoder_layers=cfg["num_decoder_layers"],
            decoder_ff_dim=cfg["decoder_ff_dim"],
            dropout=cfg["dropout"],
            vit_model_name=cfg["vit_model_name"],
            max_seq_len=DEFAULT_TRAINING_CONFIG["max_caption_len"], # Needs this from general config
            tokenizer=tokenizer_instance
        )
        if for_training: # During training, unfreeze decoder. Encoder is frozen by default.
            print("Unfreezing ViT decoder and input_proj for training...")
            for param in model.decoder.parameters():
                param.requires_grad = True
            if hasattr(model, 'input_proj') and isinstance(model.input_proj, torch.nn.Linear):
                 for param in model.input_proj.parameters():
                    param.requires_grad = True

    elif model_type == "mllm":
        model = MLLM(
            bpe_vocab_size=vocab_size,
            llm_embed_dim=cfg["llm_embed_dim"],
            llm_hidden_size=cfg["llm_hidden_size"],
            gpt2_model_name=cfg["gpt2_model_name"],
            vision_encoder_name=cfg["vision_encoder_name"],
            # num_vision_tokens=cfg["num_vision_tokens"], # Determined by ViT
            max_seq_len=DEFAULT_TRAINING_CONFIG["max_caption_len"],
            tokenizer=tokenizer_instance
        )
        if for_training: # During training, unfreeze adapter, BPE text_embedding, and lm_head
            print("Unfreezing MLLM adapter, text_embedding, and lm_head for training...")
            for param in model.adapter.parameters():
                param.requires_grad = True
            for param in model.text_embedding.parameters():
                param.requires_grad = True
            for param in model.lm_head.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Model type {model_type} not implemented for instantiation.")
    return model

def image_preprocess_for_model(image_bytes, target_size=(48,48)): # Default to dataset's native size
    """ Preprocesses uploaded image bytes to a tensor. """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Basic transform, models might do their own specific resizing (e.g. ViT)
    transform = transforms.Compose([
        transforms.Resize(target_size), # Resize to a common size first
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension


# --- Routes ---
@app.route('/')
def index():
    # Pass default configs to the template
    return render_template('index.html', default_training_config=DEFAULT_TRAINING_CONFIG, model_configs=MODEL_CONFIGS)

@app.route('/train_bpe', methods=['POST'])
def handle_train_bpe():
    global training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "Another training process is already running."}), 400
    
    try:
        data = request.json
        vocab_size = int(data.get('bpe_vocab_size', DEFAULT_TRAINING_CONFIG['bpe_vocab_size']))
        training_status = {"running": True, "log": f"Starting BPE tokenizer training with vocab size {vocab_size}..."}
        
        ensure_data_exists() # Make sure face_descriptions.txt is there
        if not os.path.exists(CAPTIONS_PATH) or os.path.getsize(CAPTIONS_PATH) == 0:
            training_status = {"running": False, "log": "Error: Captions file is missing or empty."}
            return jsonify({"status": "error", "message": training_status["log"]}), 400

        train_bpe_tokenizer(CAPTIONS_PATH, vocab_size, BPE_MODEL_DIR, os.path.basename(BPE_TOKENIZER_FILE))
        msg = f"BPE tokenizer training completed. Saved to {BPE_TOKENIZER_FILE}"
        training_status = {"running": False, "log": msg}
        return jsonify({"status": "success", "message": msg})
    except Exception as e:
        tb_str = traceback.format_exc()
        training_status = {"running": False, "log": f"BPE Training Error: {str(e)}\n{tb_str}"}
        return jsonify({"status": "error", "message": f"Error: {str(e)}", "traceback": tb_str}), 500

def model_training_task(config):
    global training_status
    try:
        training_status["log"] = "Model training started. Preparing data and model..."
        
        # 1. Ensure BPE tokenizer exists
        if not os.path.exists(BPE_TOKENIZER_FILE):
            training_status = {"running": False, "log": "Error: BPE tokenizer not found. Please train BPE first."}
            return
        tokenizer = load_bpe_tokenizer(BPE_TOKENIZER_FILE)
        vocab_size = tokenizer.get_vocab_size()

        # 2. Prepare DataLoaders
        ensure_data_exists()
        train_loader, test_loader = get_dataloaders(
            tokenizer,
            batch_size=config['batch_size'],
            test_split_ratio=config['test_split_ratio'],
            max_caption_len=config['max_caption_len'],
            num_workers=config['num_workers']
        )
        training_status["log"] = "Data loaded. Initializing model..."

        # 3. Initialize Model
        model = get_model_instance(config['model_type'], vocab_size, tokenizer, for_training=True)
        
        # 4. Trainer Config
        trainer_config_dict = {
            'lr': config['lr'],
            'epochs': config['epochs'],
            'bfloat16': config['bfloat16'] and torch.cuda.is_available() and hasattr(torch, 'bfloat16'),
            'model_name': f"{config['model_type']}_custom", # Unique name for TensorBoard/checkpoints
            'checkpoint_dir': CHECKPOINT_DIR,
            'max_caption_len': config['max_caption_len'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        training_status["log"] = f"Model {config['model_type']} initialized. Starting training on {trainer_config_dict['device']}..."
        
        # 5. Run Trainer
        trainer_instance = Trainer(model, train_loader, test_loader, tokenizer, trainer_config_dict)
        trainer_instance.train()
        
        training_status = {"running": False, "log": "Model training completed successfully."}
    except Exception as e:
        tb_str = traceback.format_exc()
        training_status = {"running": False, "log": f"Model Training Error: {str(e)}\n{tb_str}"}
        print(training_status["log"]) # Also print to console

@app.route('/train_model', methods=['POST'])
def handle_train_model():
    global training_thread, training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "A training process is already running."}), 400

    try:
        config_from_form = request.json
        
        # Combine with defaults for missing specialized model params
        final_config = DEFAULT_TRAINING_CONFIG.copy()
        final_config.update(MODEL_CONFIGS.get(config_from_form['model_type'], {})) # Model specific defaults
        final_config.update(config_from_form) # User overrides

        # Validate essential params
        if 'model_type' not in final_config or not final_config['model_type']:
            return jsonify({"status": "error", "message": "Model type must be specified."}), 400
        
        training_status = {"running": True, "log": f"Received model training request for {final_config['model_type']}."}
        
        # Run training in a separate thread to not block the web server
        training_thread = threading.Thread(target=model_training_task, args=(final_config,))
        training_thread.start()
        
        return jsonify({"status": "success", "message": "Model training started in background. Check status endpoint or TensorBoard."})
    except Exception as e:
        tb_str = traceback.format_exc()
        training_status = {"running": False, "log": f"Model Training Setup Error: {str(e)}"}
        return jsonify({"status": "error", "message": f"Error: {str(e)}", "traceback": tb_str}), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    global training_status
    # If thread finished, update status if it was running
    if training_thread is not None and not training_thread.is_alive() and training_status["running"]:
        # Thread finished but status wasn't updated by the thread itself (e.g. abrupt end)
        # The thread should ideally update status on completion/error.
        # This is a fallback.
        training_status["running"] = False
        if "Error" not in training_status["log"] and "completed" not in training_status["log"]:
             training_status["log"] += " (Thread finished)"
    return jsonify(training_status)


@app.route('/load_inference_model', methods=['POST'])
def handle_load_inference_model():
    global current_model_loaded_for_inference, current_tokenizer_loaded_for_inference, current_model_type_for_inference
    data = request.json
    model_type = data.get('model_type')
    if not model_type:
        return jsonify({"status": "error", "message": "Model type not provided."}), 400

    try:
        # 1. Load Tokenizer
        if not os.path.exists(BPE_TOKENIZER_FILE):
            return jsonify({"status": "error", "message": f"BPE Tokenizer ({BPE_TOKENIZER_FILE}) not found. Train BPE first."}), 400
        tokenizer = load_bpe_tokenizer(BPE_TOKENIZER_FILE)
        vocab_size = tokenizer.get_vocab_size()

        # 2. Load Model Checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{model_type}_custom.pth") # Matches trainer naming
        if not os.path.exists(checkpoint_path):
            return jsonify({"status": "error", "message": f"Checkpoint for {model_type} ({checkpoint_path}) not found. Train model first."}), 400

        model_instance = get_model_instance(model_type, vocab_size, tokenizer, for_training=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle potential DataParallel wrapping if model was saved like that
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model_instance.load_state_dict(state_dict)
        model_instance.to(device)
        model_instance.eval()

        current_model_loaded_for_inference = model_instance
        current_tokenizer_loaded_for_inference = tokenizer
        current_model_type_for_inference = model_type
        
        return jsonify({"status": "success", "message": f"Model {model_type} loaded for inference on {device}."})
    except Exception as e:
        tb_str = traceback.format_exc()
        return jsonify({"status": "error", "message": f"Error loading model: {str(e)}", "traceback": tb_str}), 500


@app.route('/predict', methods=['POST'])
def predict():
    global current_model_loaded_for_inference, current_tokenizer_loaded_for_inference, current_model_type_for_inference

    if not current_model_loaded_for_inference:
        return jsonify({"status": "error", "caption": "No model loaded for inference. Please load a model first."}), 400

    device = next(current_model_loaded_for_inference.parameters()).device
    max_len = DEFAULT_TRAINING_CONFIG.get('max_caption_len', 50) # Use a consistent max_len

    try:
        image_tensor = None
        original_image_data_url = None # For displaying the input image

        if 'image_file' in request.files:
            file = request.files['image_file']
            if file.filename == '':
                return jsonify({"status": "error", "caption": "No image file selected."}), 400
            
            img_bytes = file.read()
            image_tensor = image_preprocess_for_model(img_bytes) # Uses default 48x48 unless model specifies
            
            # For displaying the uploaded image
            import base64
            original_image_data_url = f"data:{file.mimetype};base64,{base64.b64encode(img_bytes).decode()}"

        elif 'test_image_index' in request.form:
            try:
                idx = int(request.form['test_image_index'])
                ensure_data_exists()
                images_np = parse_sprite_sheet(SPRITE_PATH) # (N, H, W, C) uint8
                if idx < 0 or idx >= len(images_np):
                    return jsonify({"status": "error", "caption": f"Test image index {idx} out of bounds."}), 400
                
                img_np = images_np[idx] # (H,W,C)
                # Convert (H,W,C) numpy to PIL then to tensor
                pil_img = Image.fromarray(img_np)
                
                # Create data URL for test image
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                import base64
                img_str_b64 = base64.b64encode(buffered.getvalue()).decode()
                original_image_data_url = f"data:image/png;base64,{img_str_b64}"

                # Preprocess for model
                # This transform matches dataset's typical one. Model might resize further (e.g. ViT).
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(pil_img).unsqueeze(0)

            except ValueError:
                return jsonify({"status": "error", "caption": "Invalid test image index."}), 400
            except FileNotFoundError:
                 return jsonify({"status": "error", "caption": "Sprite sheet not found for test image."}), 400
        else:
            return jsonify({"status": "error", "caption": "No image provided (either upload or test_image_index)."}), 400

        # Generate caption
        caption = current_model_loaded_for_inference.generate_caption(
            image_tensor.to(device),
            max_len=max_len,
            device=device
        )
        return jsonify({"status": "success", "caption": caption, "image_data_url": original_image_data_url})

    except Exception as e:
        tb_str = traceback.format_exc()
        return jsonify({"status": "error", "caption": f"Prediction error: {str(e)}", "traceback": tb_str}), 500


# Serve TensorBoard static files (conceptual, usually TensorBoard runs as a separate process)
# For a real deployment, you'd run `tensorboard --logdir ../runs` separately.
# This is a hacky way if you absolutely must serve it via Flask, not recommended for production.
@app.route('/runs/<path:path>')
def send_tensorboard_files(path):
    # This won't make TensorBoard UI work, it's more complex.
    # Just shows how to serve files from that directory if needed.
    # For TensorBoard, point users to `localhost:6006` (default port) after they run it.
    runs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs'))
    return send_from_directory(runs_dir, path)


# Self-test for app.py (not standard for Flask apps, but for completion)
if __name__ == '__main__':
    print("Testing app.py related components...")
    # This is not a typical self-test, as Flask apps are run, not imported and tested for functions.
    # We can test helper functions or configurations if desired.
    
    # Example: Test get_model_instance (requires a mock tokenizer)
    class MockTokenizerForAppTest:
        def token_to_id(self, token): return 0
        def get_vocab_size(self): return 100
    
    mock_tokenizer = MockTokenizerForAppTest()
    print("Testing model instantiation:")
    for model_t in MODEL_CONFIGS.keys():
        try:
            print(f"  Instantiating {model_t}...")
            model = get_model_instance(model_t, vocab_size=100, tokenizer_instance=mock_tokenizer, for_training=False)
            assert model is not None
            print(f"  {model_t} OK.")
        except Exception as e:
            print(f"  Error instantiating {model_t}: {e}")
            traceback.print_exc()

    print("To run the Flask app: `flask --app backend.app run --debug` (from project root)")
    print("Or if this file is run directly: `python backend/app.py` (needs app.run() below)")
    # For development server:
    # app.run(debug=True, host='0.0.0.0', port=5000) # This line makes it runnable with `python backend/app.py`
    # However, standard practice is `flask run`. For this structure, ensure backend is in PYTHONPATH or run from root.
    # To run:
    # 1. cd to the root directory (where `backend/`, `data/` etc. are)
    # 2. `export FLASK_APP=backend.app`
    # 3. `export FLASK_ENV=development` (optional, for debug mode)
    # 4. `flask run`
    # Or, if you uncomment app.run() above: `python backend/app.py` from root dir, or `python app.py` from backend dir.
    # For simplicity of this script, let's allow direct run if __name__ == '__main__'
    print("\nStarting Flask development server...")
    print("Access at http://127.0.0.1:5000/")
    print("Ensure data (celeba_48x48.png, face_descriptions.txt) is in data/ folder.")
    print("Run `tensorboard --logdir runs/` in a separate terminal from project root to view logs.")
    app.run(debug=True, host='0.0.0.0', port=5000)
