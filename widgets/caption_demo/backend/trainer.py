import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import random
from tqdm import tqdm
from backend.data_utils import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# Ensure this path is correct relative to where app.py will run it.
# If app.py is in backend/, then ../runs/ is fine.
# If running trainer.py directly from backend/, then ../runs/ is also fine.
TENSORBOARD_LOG_DIR = "/root/caption_demo/runs" 

class Trainer:
    def __init__(self, model, train_loader, test_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.config = config # lr, epochs, device, model_name, etc.

        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        self.lr = config.get('lr', 1e-4)
        self.epochs = config.get('epochs', 10)
        self.bfloat16 = config.get('bfloat16', False) and self.device.type == 'cuda'
        self.model_name = config.get('model_name', 'default_model')
        self.checkpoint_dir = config.get('checkpoint_dir', './backend/checkpoints')
        self.max_caption_len = config.get('max_caption_len', 50)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
        
        # Optimizer: Only optimize parameters that require gradients
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        
        # PAD token id for loss calculation
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        if self.pad_token_id is None:
            raise ValueError(f"PAD_TOKEN '{PAD_TOKEN}' not found in tokenizer.")
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        
        self.writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, self.model_name))
        
        self.scaler = None
        if self.bfloat16:
            if not hasattr(torch.cuda.amp, 'GradScaler'): # Older PyTorch
                 print("Warning: torch.cuda.amp.GradScaler not found. bfloat16 may not work optimally or requires manual scaling.")
            else:
                 self.scaler = torch.cuda.amp.GradScaler()
            print(f"Training with bfloat16 precision: {self.bfloat16}")
        
        self.sos_token_id = self.tokenizer.token_to_id(SOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)


    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Training]")

        for batch_idx, (images, captions_input, captions_target) in enumerate(progress_bar):
            images = images.to(self.device)
            captions_input = captions_input.to(self.device)   # For teacher forcing: (SOS, t1, t2, ..., tn, PAD, ...)
            captions_target = captions_target.to(self.device) # For loss: (t1, t2, ..., tn, EOS, PAD, ...)

            self.optimizer.zero_grad()

            if self.bfloat16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    # captions_input is used as decoder input (teacher forcing)
                    outputs = self.model(images, captions_input) # (batch, seq_len, vocab_size)
                    # For MLLM, outputs are already sliced for text part.
                    # captions_target is (batch, seq_len)
                    loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), captions_target.reshape(-1))
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else: # Manual scaling or no scaler if not available for bfloat16 (older torch)
                    loss.backward()
                    self.optimizer.step()
            else: # FP32 or CPU
                outputs = self.model(images, captions_input)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), captions_target.reshape(-1))
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if batch_idx % 100 == 0: # Log batch loss periodically
                 self.writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.test_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Validation]")

        for images, captions_input, captions_target in progress_bar:
            images = images.to(self.device)
            captions_input = captions_input.to(self.device)
            captions_target = captions_target.to(self.device)

            if self.bfloat16 and self.device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.model(images, captions_input)
                    loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), captions_target.reshape(-1))
            else:
                outputs = self.model(images, captions_input)
                loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), captions_target.reshape(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / len(self.test_loader)
        self.writer.add_scalar('Loss/validation_epoch', avg_loss, epoch)
        print(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}")
        
        # Log some random test results (image + generated caption + ground truth)
        self.log_test_samples(epoch)
        return avg_loss

    @torch.no_grad()
    def log_test_samples(self, epoch, num_samples=4):
        self.model.eval()
        # Fetch some random samples from the test set
        # The test_loader shuffles if shuffle=True, otherwise deterministic.
        # For consistent random samples, could re-initialize a subset of test_dataset or pick random indices.
        
        # Get a batch from test_loader
        try:
            images, captions_input, captions_target = next(iter(self.test_loader))
        except StopIteration:
            print("Test loader is empty, cannot log samples.")
            return

        images = images.to(self.device)[:num_samples]
        captions_target_ids = captions_target[:num_samples] # For ground truth text

        # Denormalize images for visualization if necessary
        # Assuming standard ImageNet normalization:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        img_grid = make_grid(images_denorm.cpu(), nrow=num_samples)
        self.writer.add_image(f'TestSamples/Epoch_{epoch}/Images', img_grid, epoch)

        for i in range(images.size(0)):
            img_tensor_single = images[i].unsqueeze(0) # (1, C, H, W)
            
            # Use model's generate_caption method
            # Ensure model has tokenizer set and generate_caption method implemented
            if hasattr(self.model, 'generate_caption') and callable(getattr(self.model, 'generate_caption')):
                # Some models might need their own tokenizer passed if not stored internally
                generated_caption_text = self.model.generate_caption(
                    img_tensor_single, 
                    max_len=self.max_caption_len, 
                    device=self.device
                )
            else:
                generated_caption_text = "Model does not have generate_caption method."

            # Decode ground truth caption
            gt_ids = captions_target_ids[i].cpu().tolist()
            # Remove padding, SOS, EOS for display
            try:
                gt_ids_cleaned = [tid for tid in gt_ids if tid != self.pad_token_id]
                if gt_ids_cleaned and gt_ids_cleaned[0] == self.sos_token_id: # Should not be there in target typically
                    gt_ids_cleaned = gt_ids_cleaned[1:]
                if gt_ids_cleaned and gt_ids_cleaned[-1] == self.eos_token_id:
                     gt_ids_cleaned = gt_ids_cleaned[:-1]
            except AttributeError: # If sos/eos_token_id is None (should not happen with proper setup)
                gt_ids_cleaned = gt_ids_cleaned

            ground_truth_text = self.tokenizer.decode(gt_ids_cleaned, skip_special_tokens=True)
            
            text_log = f"Sample {i}:\n  GT: {ground_truth_text}\n  Pred: {generated_caption_text}\n"
            self.writer.add_text(f'TestSamples/Epoch_{epoch}/Captions_Sample_{i}', text_log, epoch)
            if i < 2: print(text_log) # Print first few to console

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            val_loss = self.evaluate_epoch(epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.checkpoint_dir, f"best_model_{self.model_name}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config # Save config for reloading
                }, save_path)
                print(f"Epoch {epoch+1}: Validation loss improved to {val_loss:.4f}. Model saved to {save_path}")
        
        self.writer.close()
        print("Training complete.")


# Self-test
if __name__ == "__main__":
    print("Testing trainer.py...")
    # This requires data_utils, tokenizer, and a model. It's more of an integration test.
    # We'll mock these components heavily.

    from backend.data_utils import ImageCaptionDataset # To use its structure
    from backend.bpe_trainer import load_tokenizer as load_bpe_tokenizer, TOKENIZER_FILE as BPE_TOKENIZER_FILE, train_bpe_tokenizer, BPE_MODEL_DIR, DEFAULT_VOCAB_SIZE
    from backend.data_utils import ensure_data_exists, CAPTIONS_PATH, SPRITE_PATH, parse_sprite_sheet, load_captions
    from backend.model_clstm import ConvLSTMModel # Using ConvLSTM for test
    from torch.utils.data import DataLoader, Subset

    ensure_data_exists() # Ensure dummy data/captions exist

    # 1. Prepare a BPE tokenizer (or load existing)
    test_bpe_dir = os.path.join(BPE_MODEL_DIR, "trainer_test_bpe")
    test_tokenizer_filename = "trainer_test_tokenizer.json"
    test_tokenizer_path = os.path.join(test_bpe_dir, test_tokenizer_filename)

    if not os.path.exists(test_tokenizer_path):
        print("Trainer Test: Training a small BPE tokenizer...")
        try:
            train_bpe_tokenizer(CAPTIONS_PATH, vocab_size=500, output_dir=test_bpe_dir, tokenizer_filename=test_tokenizer_filename)
        except Exception as e:
            print(f"Could not train BPE for trainer test: {e}. Skipping trainer test.")
            exit()
    
    try:
        tokenizer = load_bpe_tokenizer(test_tokenizer_path)
    except FileNotFoundError:
        print(f"Test tokenizer not found at {test_tokenizer_path}. Skipping trainer test.")
        exit()

    vocab_size = tokenizer.get_vocab_size()

    # 2. Prepare DataLoaders with a small subset of data
    try:
        all_images = parse_sprite_sheet(SPRITE_PATH)
        all_captions = load_captions(CAPTIONS_PATH)
        
        # Use a very small subset for quick testing
        subset_indices = list(range(20)) # First 20 samples
        images_subset = all_images[subset_indices]
        captions_subset = [all_captions[i] for i in subset_indices]

        # Create a full dataset instance then subset it
        full_dataset_for_test = ImageCaptionDataset(images_subset, captions_subset, tokenizer, max_len=20)
        
        # Split this tiny dataset
        train_size = int(0.8 * len(full_dataset_for_test))
        test_size = len(full_dataset_for_test) - train_size
        if train_size == 0 or test_size == 0: # Ensure not empty
            print("Dataset too small for train/test split in trainer test. Using all for train, duplicating for test.")
            train_subset = full_dataset_for_test
            test_subset = full_dataset_for_test
        else:
            train_subset, test_subset = torch.utils.data.random_split(full_dataset_for_test, [train_size, test_size])

        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        collate_with_pad = lambda batch: torch.utils.data.dataloader.default_collate(batch) if pad_id is None \
            else lambda b: collate_fn(b, pad_id) # Need the custom collate for padding
        
        # Re-define collate_fn locally for test if not easily importable or to avoid circular deps
        def local_collate_fn(batch, pad_token_id_val):
            images, input_captions, target_captions = zip(*batch)
            images = torch.stack(images, 0)
            input_captions = torch.stack(input_captions, 0) # Assumes already padded by dataset
            target_captions = torch.stack(target_captions, 0) # Assumes already padded by dataset
            return images, input_captions, target_captions

        train_loader_test = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=lambda b: local_collate_fn(b, pad_id))
        test_loader_test = DataLoader(test_subset, batch_size=4, shuffle=False, collate_fn=lambda b: local_collate_fn(b, pad_id))

    except Exception as e:
        print(f"Could not prepare data for trainer test: {e}. Skipping trainer test.")
        import traceback
        traceback.print_exc()
        exit()

    # 3. Initialize a Model (ConvLSTM)
    model_config = {
        'embed_size': 64, 
        'hidden_size': 128, 
        'vocab_size': vocab_size,
        'encoder_feature_size': 128, # Match hidden_size for ConvLSTM
        'tokenizer': tokenizer # Pass tokenizer to model for generation
    }
    test_model = ConvLSTMModel(**model_config)

    # 4. Trainer Config
    trainer_config = {
        'lr': 1e-3,
        'epochs': 1, # Just one epoch for test
        'bfloat16': torch.cuda.is_available() and hasattr(torch, 'bfloat16'), # Test if supported
        'model_name': 'clstm_test_trainer',
        'checkpoint_dir': './backend/checkpoints_test_trainer',
        'max_caption_len': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Trainer Test: Using device: {trainer_config['device']}, bfloat16: {trainer_config['bfloat16']}")

    # 5. Run Trainer
    try:
        trainer_instance = Trainer(test_model, train_loader_test, test_loader_test, tokenizer, trainer_config)
        trainer_instance.train()
        print("trainer.py test (1 epoch training) completed.")

        # Check if checkpoint and tensorboard logs were created
        assert os.path.exists(os.path.join(trainer_config['checkpoint_dir'], f"best_model_{trainer_config['model_name']}.pth"))
        # Tensorboard dir is ../runs relative to this file, then model_name subdir
        tb_log_path = os.path.join(TENSORBOARD_LOG_DIR, trainer_config['model_name'])
        assert os.path.exists(tb_log_path) and len(os.listdir(tb_log_path)) > 0
        print(f"Checkpoint and Tensorboard logs created at: {trainer_config['checkpoint_dir']} and {tb_log_path}")

    except Exception as e:
        print(f"Error during trainer.py self-test execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test BPE and checkpoint files/dirs
        if os.path.exists(test_tokenizer_path):
            os.remove(test_tokenizer_path)
        if os.path.exists(test_bpe_dir) and not os.listdir(test_bpe_dir):
            os.rmdir(test_bpe_dir)
        
        # Clean up test checkpoints
        # import shutil
        # if os.path.exists(trainer_config['checkpoint_dir']):
        #     shutil.rmtree(trainer_config['checkpoint_dir'])
        # if os.path.exists(tb_log_path): # Careful with this if other tests use same dir
        #     shutil.rmtree(tb_log_path)
        print("Consider manually cleaning up test artifacts in backend/checkpoints_test_trainer/ and ../runs/clstm_test_trainer/ if needed.")


    print("trainer.py tests logic finished.")
