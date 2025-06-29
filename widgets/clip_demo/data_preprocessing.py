# data_preprocessing.py
import os
import argparse
import unittest
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def preprocess_celeba(attr_file, img_dir, output_cache_file, img_size, limit=None):
    """
    Reads CelebA images and attributes, processes them, and saves them to a compressed NumPy cache file.
    
    Processing includes:
    1. Cropping images to a standard size (178x178).
    2. Resizing images to the target `img_size`.
    3. Converting attributes from -1/1 to 0/1 format.
    4. Saving images (uint8) and attributes (float32) to an .npz file.
    """
    print("Starting preprocessing...")
    annots = pd.read_csv(attr_file)
    if limit:
        print(f"Limiting preprocessing to the first {limit} images.")
        annots = annots.head(limit)

    images_list = []
    attrs_list = []

    def _crop(image):
        w, h = image.size
        # CelebA standard crop dimensions
        left = (w - 178) // 2
        top = (h - 178) // 2
        right = left + 178
        bottom = top + 178
        return image.crop((left, top, right, bottom))

    for idx in tqdm(range(len(annots)), desc=f"Processing Images for {img_size}x{img_size} cache"):
        row = annots.iloc[idx]
        img_path = os.path.join(img_dir, row['image_id'])
        try:
            image = Image.open(img_path).convert('RGB')
            image = _crop(image)
            image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
            images_list.append(np.array(image))  # (H, W, C) uint8
        except (IOError, FileNotFoundError) as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            continue

        # Attributes: convert -1/1 to 0/1
        attrs = row[1:].values.astype(float)
        attrs = (attrs + 1) / 2
        attrs_list.append(attrs)
    
    if not images_list:
        print("Error: No images were processed. Please check your image directory and attribute file.")
        return

    all_images = np.stack(images_list)
    all_attrs = np.array(attrs_list, dtype='float32')

    cache_dir = os.path.dirname(output_cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Saving {len(all_images)} processed images to cache file: {output_cache_file}")
    np.savez_compressed(output_cache_file, images=all_images, attributes=all_attrs)
    print("Preprocessing complete.")


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = "test_data_preprocess"
        self.img_dir = os.path.join(self.test_dir, "img")
        self.attr_file = os.path.join(self.test_dir, "attr.csv")
        self.output_file = os.path.join(self.test_dir, "test_cache.npz")
        os.makedirs(self.img_dir, exist_ok=True)
        
        # Create a dummy image
        img = Image.new('RGB', (178, 218), color='blue')
        img.save(os.path.join(self.img_dir, "test.jpg"))
        
        # Create a dummy attributes file
        with open(self.attr_file, "w") as f:
            headers = ["image_id"] + [f"attr_{i}" for i in range(40)]
            f.write(",".join(headers) + "\n")
            values = ["test.jpg"] + [str(np.random.choice([-1, 1])) for _ in range(40)]
            f.write(",".join(values) + "\n")

    def tearDown(self):
        # Clean up the temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_preprocess_function(self):
        img_size = 32
        self.assertFalse(os.path.exists(self.output_file))

        # Run the preprocessing function
        preprocess_celeba(self.attr_file, self.img_dir, self.output_file, img_size)

        # Check if cache file was created
        self.assertTrue(os.path.exists(self.output_file))

        # Load and verify cache content
        data = np.load(self.output_file)
        self.assertIn('images', data)
        self.assertIn('attributes', data)
        
        images = data['images']
        attrs = data['attributes']

        self.assertEqual(images.shape, (1, img_size, img_size, 3))
        self.assertEqual(images.dtype, np.uint8)
        
        self.assertEqual(attrs.shape, (1, 40))
        self.assertEqual(attrs.dtype, np.float32)
        
        # Check values (image should be blue)
        self.assertEqual(images[0, 0, 0, 0], 0)   # R
        self.assertEqual(images[0, 0, 0, 1], 0)   # G
        self.assertEqual(images[0, 0, 0, 2], 255) # B

        # Check attributes are 0 or 1
        self.assertTrue(np.all((attrs == 0) | (attrs == 1)))
        print("\nData preprocessing function test successful.")


if __name__ == '__main__':
    # Allows running the script from the command line
    parser = argparse.ArgumentParser(description="Preprocess CelebA dataset into a cache file.")
    parser.add_argument("--attr_file", type=str, required=True, help="Path to the list_attr_celeba.csv file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the img_align_celeba directory.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output .npz cache file.")
    parser.add_argument("--img_size", type=int, default=64, help="The target square image size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit to the first N images.")
    
    # Allows running unit tests like `python data_preprocessing.py test`
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        args = parser.parse_args()
        preprocess_celeba(args.attr_file, args.img_dir, args.output_file, args.img_size, args.limit)
