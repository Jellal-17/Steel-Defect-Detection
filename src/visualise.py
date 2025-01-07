import matplotlib.pyplot as plt
from PIL import Image
import os

# Paths
TRAIN_IMAGES = "train_images/"
MASK_OUTPUT = "mask_output/"

# Choose an example
image_id = "0002cc93b.jpg"  # Replace with an actual ImageId from train.csv
class_id = 1  # Replace with a class (1, 2, 3, or 4)

# Load the original image
image_path = os.path.join(TRAIN_IMAGES, image_id)
original_image = Image.open(image_path)

# Load the corresponding mask
mask_path = os.path.join(MASK_OUTPUT, f"{class_id}", f"{image_id.split('.')[0]}_class{class_id}.png")
mask_image = Image.open(mask_path)

# Visualize the original image and mask
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mask_image, cmap="gray")
ax[1].set_title(f"Mask for Class {class_id}")
ax[1].axis("off")

plt.tight_layout()
plt.show()
