import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

# Paths
TRAIN_IMAGES = "/media/storage1/Sathvik/Steel-Defect-Detection/notebooks/train_images/"
MASK_OUTPUT = "/media/storage1/Sathvik/Steel-Defect-Detection/notebooks/mask_output/"
PROCESSED_DATA = "/media/storage1/Sathvik/Steel-Defect-Detection/processed_data/"

# Target size for resizing
IMG_SIZE = (512, 512)

# Create processed directories
os.makedirs(f"{PROCESSED_DATA}/images/train", exist_ok=True)
os.makedirs(f"{PROCESSED_DATA}/masks/train", exist_ok=True)
os.makedirs(f"{PROCESSED_DATA}/images/val", exist_ok=True)
os.makedirs(f"{PROCESSED_DATA}/masks/val", exist_ok=True)


# Resize images and masks while preserving aspect ratio
def resize_with_aspect_ratio(image, target_size):
    """
    Resize an image while preserving aspect ratio, with padding to match the target size.
    Args:
        image: The input image (as a PIL Image).
        target_size: Tuple (target_height, target_width).
    Returns:
        Resized and padded image as a PIL Image.
    """
    img = image.copy()
    img.thumbnail((target_size[1], target_size[0]), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality downscaling
    delta_w = target_size[1] - img.size[0]
    delta_h = target_size[0] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded_img = ImageOps.expand(img, padding, fill=0)  # Pad with black pixels
    return padded_img


# Load and preprocess images and masks
image_files = sorted(os.listdir(TRAIN_IMAGES))
mask_files = sorted([os.path.join(root, name)
                     for root, _, files in os.walk(MASK_OUTPUT)
                     for name in files])

images = []
masks = []

for img_path, mask_path in zip(image_files, mask_files):
    # Load and preprocess the image
    img = Image.open(os.path.join(TRAIN_IMAGES, img_path)).convert("RGB")  # Ensure RGB format
    img_resized = resize_with_aspect_ratio(img, IMG_SIZE)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    images.append(img_array)

    # Load and preprocess the mask
    mask = Image.open(mask_path)
    mask_resized = resize_with_aspect_ratio(mask, IMG_SIZE)
    mask_array = np.array(mask_resized) / 255.0  # Scale between [0, 1]
    masks.append(mask_array)

# Convert to numpy arrays
images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Save preprocessed data
for i, (img, mask) in enumerate(zip(X_train, y_train)):
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{PROCESSED_DATA}/images/train/img_{i}.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"{PROCESSED_DATA}/masks/train/mask_{i}.png")

for i, (img, mask) in enumerate(zip(X_val, y_val)):
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{PROCESSED_DATA}/images/val/img_{i}.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(f"{PROCESSED_DATA}/masks/val/mask_{i}.png")

print("Data preprocessing and resizing complete!")
