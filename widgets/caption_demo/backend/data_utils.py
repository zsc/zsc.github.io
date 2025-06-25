import os
import requests
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tokenizers import Tokenizer
from typing import List, Tuple, Callable

# Constants
SPRITE_URL = "http://zsc.github.io/widgets/celeba/48x48.png"
CAPTIONS_URL = "YOUR_URL_FOR_FACE_DESCRIPTIONS_TXT" # Replace with actual URL or local path logic
DATA_DIR = "../data"
SPRITE_PATH = os.path.join(DATA_DIR, "celeba_48x48.png")
CAPTIONS_PATH = "data/face_descriptions.txt"

IMG_WIDTH, IMG_HEIGHT = 48, 48
SPRITE_COLS = 200
SPRITE_ROWS = 150 # 30000 images / 200 per row

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"

def download_file(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            if os.path.exists(local_path): # Clean up partial download
                os.remove(local_path)
            raise
    else:
        print(f"{local_path} already exists.")

def ensure_data_exists():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(SPRITE_URL, SPRITE_PATH)
    # For captions, assuming it's manually placed or you provide a URL
    if not os.path.exists(CAPTIONS_PATH):
        print(f"Error: {CAPTIONS_PATH} not found. Please place it in the data directory or provide a download URL.")
        # If CAPTIONS_URL is set, uncomment:
        # download_file(CAPTIONS_URL, CAPTIONS_PATH)
        # For now, let's create a dummy if it's missing for testing structure
        if not os.path.exists(CAPTIONS_PATH) and CAPTIONS_URL == "YOUR_URL_FOR_FACE_DESCRIPTIONS_TXT":
             print(f"Creating dummy {CAPTIONS_PATH} for structure testing.")
             with open(CAPTIONS_PATH, 'w') as f:
                 for i in range(SPRITE_COLS * SPRITE_ROWS):
                     f.write(f"This is a dummy caption for image {i}.\n")


def parse_sprite_sheet(sprite_path: str) -> np.ndarray:
    """Parses the sprite sheet into individual images."""
    if not os.path.exists(sprite_path):
        raise FileNotFoundError(f"Sprite sheet not found at {sprite_path}")
    
    sprite_img = Image.open(sprite_path).convert("RGB")
    sprite_arr = np.array(sprite_img) # (7200, 9600, 3)

    num_images = SPRITE_COLS * SPRITE_ROWS
    images = np.zeros((num_images, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    
    idx = 0
    for r in range(SPRITE_ROWS):
        for c in range(SPRITE_COLS):
            y_start, y_end = r * IMG_HEIGHT, (r + 1) * IMG_HEIGHT
            x_start, x_end = c * IMG_WIDTH, (c + 1) * IMG_WIDTH
            images[idx] = sprite_arr[y_start:y_end, x_start:x_end, :]
            idx += 1
    # Reshape to (num_images, H, W, C)
    return images # Shape: (30000, 48, 48, 3)

def load_captions(captions_path: str) -> List[str]:
    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions file not found at {captions_path}")
    with open(captions_path, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f if line.strip()]
    return captions

class ImageCaptionDataset(Dataset):
    def __init__(self, images: np.ndarray, captions: List[str], tokenizer: Tokenizer, transform: Callable = None, max_len: int = 50):
        self.images = images
        self.captions = captions
        self.tokenizer = tokenizer
        self.transform = transform if transform else self.default_transform()
        self.max_len = max_len
        
        self.sos_token_id = tokenizer.token_to_id(SOS_TOKEN)
        self.eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
        self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)

    def default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(), # If images are numpy HWC
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # Ensure size if needed, though sprite parsing should handle it
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet mean/std
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_np = self.images[idx] # HWC numpy array
        caption_text = self.captions[idx]

        image_tensor = self.transform(image_np)

        tokenized = self.tokenizer.encode(caption_text)
        caption_ids = tokenized.ids
        
        # For decoder input (teacher forcing)
        input_ids = [self.sos_token_id] + caption_ids
        # For decoder target
        target_ids = caption_ids + [self.eos_token_id]

        # Pad
        input_ids_padded = input_ids[:self.max_len] + [self.pad_token_id] * (self.max_len - len(input_ids))
        target_ids_padded = target_ids[:self.max_len] + [self.pad_token_id] * (self.max_len - len(target_ids))
        
        return image_tensor, torch.LongTensor(input_ids_padded), torch.LongTensor(target_ids_padded)

def collate_fn(batch, pad_token_id):
    images, input_captions, target_captions = zip(*batch)
    images = torch.stack(images, 0)
    
    # input_captions and target_captions are already padded in __getitem__
    input_captions = torch.stack(input_captions, 0)
    target_captions = torch.stack(target_captions, 0)
    
    return images, input_captions, target_captions

def get_dataloaders(tokenizer: Tokenizer, batch_size: int, test_split_ratio: float = 0.1, max_caption_len: int = 50, num_workers: int = 2):
    ensure_data_exists()
    images = parse_sprite_sheet(SPRITE_PATH)
    captions = load_captions(CAPTIONS_PATH)

    if len(images) != len(captions):
        print(f"Warning! Number of images ({len(images)}) and captions ({len(captions)}) do not match. Truncate images to match.")
        images = images[:len(captions)]

    dataset = ImageCaptionDataset(images, captions, tokenizer, max_len=max_caption_len)
    
    total_len = len(dataset)
    test_len = int(total_len * test_split_ratio)
    train_len = total_len - test_len

    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    collate_with_pad = lambda batch: collate_fn(batch, pad_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_with_pad, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_with_pad, pin_memory=True)
    
    return train_loader, test_loader

# Self-test
if __name__ == "__main__":
    print("Testing data_utils.py...")
    ensure_data_exists() # This will create dummy captions if not present
    
    # Test sprite parsing
    images_data = parse_sprite_sheet(SPRITE_PATH)
    print(f"Parsed images shape: {images_data.shape}") # Should be (30000, 48, 48, 3)
    assert images_data.shape == (SPRITE_COLS * SPRITE_ROWS, IMG_HEIGHT, IMG_WIDTH, 3)
    
    # Test caption loading
    captions_data = load_captions(CAPTIONS_PATH)
    print(f"Loaded {len(captions_data)} captions. First: {captions_data[0]}")
    #assert len(captions_data) == SPRITE_COLS * SPRITE_ROWS

    # Dummy Tokenizer for testing Dataset and DataLoader
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    dummy_tokenizer_path = os.path.join(DATA_DIR, "dummy_tokenizer.json")
    if not os.path.exists(dummy_tokenizer_path):
        print("Creating dummy tokenizer for data_utils test...")
        dummy_tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        dummy_tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN], vocab_size=1000) # Small vocab
        # Use a small subset of captions for quick dummy tokenizer training
        sample_captions = captions_data[:100] if len(captions_data) > 100 else captions_data
        if not sample_captions: # If dummy captions were used and empty
            sample_captions = ["this is a test sentence.", "another one here."]
        dummy_tokenizer.train_from_iterator(sample_captions, trainer=trainer)
        dummy_tokenizer.save(dummy_tokenizer_path)
    
    tokenizer_instance = Tokenizer.from_file(dummy_tokenizer_path)

    # Test Dataset
    dataset = ImageCaptionDataset(images_data[:10], captions_data[:10], tokenizer_instance) # Use a small subset
    img, cap_in, cap_out = dataset[0]
    print(f"Sample image tensor shape: {img.shape}")
    print(f"Sample input caption tensor: {cap_in}")
    print(f"Sample target caption tensor: {cap_out}")
    assert img.shape == (3, IMG_HEIGHT, IMG_WIDTH)

    # Test DataLoader
    train_loader, test_loader = get_dataloaders(tokenizer_instance, batch_size=4, test_split_ratio=0.5, max_caption_len=20) # 50% test for small data
    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Test loader size: {len(test_loader.dataset)}")
    
    for batch_idx, (imgs, cap_ins, cap_outs) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {imgs.shape}")      # (batch_size, 3, 48, 48)
        print(f"  Input captions shape: {cap_ins.shape}") # (batch_size, max_len)
        print(f"  Output captions shape: {cap_outs.shape}")# (batch_size, max_len)
        assert imgs.shape[0] <= 4 and imgs.shape[1:] == (3, IMG_HEIGHT, IMG_WIDTH)
        assert cap_ins.shape[0] <=4 and cap_ins.shape[1] == 20
        assert cap_outs.shape[0] <=4 and cap_outs.shape[1] == 20
        break # Only check one batch
    
    print("data_utils.py tests passed!")
