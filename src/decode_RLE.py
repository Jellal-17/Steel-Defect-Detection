import os
import numpy as np
import pandas as pd
from PIL import Image

# Paths
TRAIN_CSV = "train.csv"
TRAIN_IMAGES = "train_images/"
MASK_OUTPUT = "mask_output/"

# Function to decode RLE
def rle_decode(mask_rle, shape=(256, 1600)):
    """
    Decodes Run-Length Encoding (RLE) into a binary mask.
    Args:
        mask_rle: RLE string (e.g., "1 3 10 2")
        shape: Tuple (height, width) of the mask
    Returns:
        Binary mask (2D array)
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if mask_rle is not np.nan:  # Skip NaN values
        s = list(map(int, mask_rle.split()))
        starts, lengths = s[0::2], s[1::2]
        for start, length in zip(starts, lengths):
            start -= 1  # Convert to zero-based indexing
            mask[start:start + length] = 1
    return mask.reshape(shape).T  # Reshape to original dimensions


if __name__ == "__main__":
    # Load CSV
    train_df = pd.read_csv(TRAIN_CSV)

    # Create Masks
    for idx, row in train_df.iterrows():
        image_id = row["ImageId"]
        class_id = row["ClassId"]
        encoded_pixels = row["EncodedPixels"]

        # Load image dimensions
        image_path = os.path.join(TRAIN_IMAGES, image_id)
        img = Image.open(image_path)
        height, width = img.size

        # Decode RLE to create a mask
        mask = rle_decode(encoded_pixels, shape=(height, width))

        # Save mask
        mask_dir = os.path.join(MASK_OUTPUT, f"{class_id}")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{image_id.split('.')[0]}_class{class_id}.png")
        Image.fromarray(mask * 255).save(mask_path)
        # Convert mask to binary image (0-255)

    print("Masks generated and saved!")
