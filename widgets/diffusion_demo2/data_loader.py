import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os

IMAGE_URL = "http://zsc.github.io/widgets/celeba/48x48.png"
SPRITE_SHEET_SHAPE = (150, 200) # 150 rows, 200 images per row
IMAGE_SIZE = 48
NUM_IMAGES = 30000
CACHE_DIR = "cache"
NUMPY_CACHE_FILE = os.path.join(CACHE_DIR, "celeba_48x48_sprites.npy")

class CelebASpriteSheetDataset(Dataset):
    def __init__(self, transform=None, data_limit=None):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.transform = transform
        
        if os.path.exists(NUMPY_CACHE_FILE):
            print(f"Loading images from cache: {NUMPY_CACHE_FILE}")
            self.images_np = np.load(NUMPY_CACHE_FILE)
        else:
            print(f"Downloading and processing image from {IMAGE_URL}")
            response = requests.get(IMAGE_URL)
            response.raise_for_status() # Ensure download was successful
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_np = np.array(img) # Shape: (7200, 9600, 3)
            
            self.images_np = self._extract_sprites(img_np)
            np.save(NUMPY_CACHE_FILE, self.images_np)
            print(f"Saved processed images to cache: {NUMPY_CACHE_FILE}")

        if data_limit is not None and data_limit < len(self.images_np):
            self.images_np = self.images_np[:data_limit]
            print(f"Using a subset of {data_limit} images.")

    def _extract_sprites(self, sprite_sheet_np):
        # sprite_sheet_np shape: (total_height, total_width, 3)
        # total_height = 150 * 48 = 7200
        # total_width = 200 * 48 = 9600
        num_rows, num_cols = SPRITE_SHEET_SHAPE
        images = []
        for r in range(num_rows):
            for c in range(num_cols):
                start_y = r * IMAGE_SIZE
                end_y = start_y + IMAGE_SIZE
                start_x = c * IMAGE_SIZE
                end_x = start_x + IMAGE_SIZE
                sprite = sprite_sheet_np[start_y:end_y, start_x:end_x, :]
                images.append(sprite)
        return np.array(images) # Shape: (30000, 48, 48, 3)

    def __len__(self):
        return len(self.images_np)

    def __getitem__(self, idx):
        # Retrieve the numpy array for the given index
        image_np = self.images_np[idx] # This is (H, W, C), uint8

        if self.transform:
            # If transforms are present, convert to PIL Image first.
            # Many torchvision transforms (e.g., Resize, ColorJitter) expect PIL Images.
            # This maintains the original behavior for the transformed path and ensures
            # compatibility with a wider range of standard torchvision transforms.
            # ToTensor() itself can handle numpy arrays, but it's often not the only transform.
            image_pil = Image.fromarray(image_np)
            return self.transform(image_pil)
        else:
            # If no transform is specified, return the numpy array directly.
            # This addresses the AttributeError in the test (line 83), where a numpy array
            # was expected for 'sample_np' (derived from dataset_no_transform[0]).
            # The expectation was clear from accessing .shape and .dtype attributes,
            # the variable name 'sample_np', and the comment.
            return image_np

def get_dataloader(batch_size=64, shuffle=True, data_limit=None):
    transform_chain = transforms.Compose([ # Renamed variable to avoid confusion
        transforms.ToTensor(),                # Converts to [0, 1] range, (C, H, W)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # To [-1, 1] range
    ])
    dataset = CelebASpriteSheetDataset(transform=transform_chain, data_limit=data_limit)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dataloader

# Test for data_loader.py
if __name__ == "__main__":
    print("Testing DataLoader...")
    
    # Test dataset directly
    print("Testing CelebASpriteSheetDataset...")
    dataset_no_transform = CelebASpriteSheetDataset(data_limit=10)
    print(f"Dataset length (limited): {len(dataset_no_transform)}")
    sample_np = dataset_no_transform[0] # __getitem__ with transform=None now returns ndarray
    print(f"Sample numpy image shape: {sample_np.shape}, dtype: {sample_np.dtype}") # Should be (48, 48, 3) uint8

    # Test with transform
    # Renamed 'transform' variable to 'active_transform' to avoid any potential name collision
    # with the imported 'transforms' module, although it was not an issue in this specific scope.
    active_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_with_transform = CelebASpriteSheetDataset(transform=active_transform, data_limit=10)
    sample_tensor = dataset_with_transform[0]
    print(f"Sample tensor shape: {sample_tensor.shape}, dtype: {sample_tensor.dtype}") # Should be (3, 48, 48) float32
    print(f"Sample tensor min: {sample_tensor.min()}, max: {sample_tensor.max()}") # Should be around -1 to 1

    # Test DataLoader
    print("\nTesting get_dataloader...")
    dataloader = get_dataloader(batch_size=4, data_limit=10)
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} shape: {batch.shape}, dtype: {batch.dtype}")
        if i == 1: # Print first two batches
            break
    print("DataLoader test completed.")

    # You can optionally visualize a sample
    # from torchvision.utils import make_grid
    # import matplotlib.pyplot as plt
    # dataiter = iter(dataloader)
    # images = next(dataiter)
    # img_grid = make_grid(images)
    # img_grid = (img_grid / 2 + 0.5) # Unnormalize
    # plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
    # plt.title("Sample Batch")
    # plt.show()
