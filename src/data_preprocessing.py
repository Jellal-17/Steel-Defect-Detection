import os
import argparse
import json
from PIL import Image, ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
import albumentations as A


def resize_with_aspect_ratio(image: Image.Image, size: tuple, resample: int) -> Image.Image:
    img = image.copy()
    img.thumbnail((size[1], size[0]), resample)
    delta_w = size[1] - img.size[0]
    delta_h = size[0] - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=0)


def load_and_merge_masks(image_id: str, mask_root: str, image_size: tuple) -> Image.Image:
    base = os.path.splitext(image_id)[0]
    masks = []
    for cls in range(1, 5):
        path = os.path.join(mask_root, str(cls), f"{base}_class{cls}.*")
        # if os.path.exists(path):
        #     masks.append(np.array(Image.open(path).convert("L")))
        for p in glob(path):
            masks.append(np.array(Image.open(p).convert("L")))
            break # Stop after first match
    if masks:
        merged = np.maximum.reduce(masks)
    else:
        merged = np.zeros((image_size[::-1]), dtype=np.uint8)
    return Image.fromarray(merged)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess steel defect dataset")
    parser.add_argument("--image-dir", type=str, default="train_images", help="Directory with raw training images")
    parser.add_argument("--mask-dir", type=str, default="mask_output", help="Directory with class mask subfolders")
    parser.add_argument("--output-dir", type=str, default="processed_data", help="Where to write processed data")
    parser.add_argument("--img-size", type=int, default=512, help="Output square size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--augment", action="store_true", help="Apply Albumentations augmentations")
    return parser.parse_args()


TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(10, 16), hole_width_range=(10, 16), fill=0, p=0.5),
])


def process_dataset(args):
    os.makedirs(os.path.join(args.output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks", "val"), exist_ok=True)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith((".png", ".jpg"))])
    train_ids, val_ids = train_test_split(image_files, test_size=args.val_split, random_state=42)

    sum_channels = np.zeros(3, dtype=np.float64)
    sumsq_channels = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for img_name in tqdm(ids, desc=f"Processing {split}"):
            img_path = os.path.join(args.image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            mask = load_and_merge_masks(img_name, args.mask_dir, img.size)

            img_resized = resize_with_aspect_ratio(img, (args.img_size, args.img_size), Image.Resampling.LANCZOS)
            mask_resized = resize_with_aspect_ratio(mask, (args.img_size, args.img_size), Image.Resampling.NEAREST)

            img_array = np.array(img_resized)
            mask_array = np.array(mask_resized)

            if args.augment and split == "train":
                augmented = TRANSFORMS(image=img_array, mask=mask_array)
                img_array = augmented["image"]
                mask_array = augmented["mask"]

            if split == "train":
                img_float = img_array.astype(np.float32) / 255.0
                sum_channels += img_float.sum(axis=(0, 1))
                sumsq_channels += (img_float ** 2).sum(axis=(0, 1))
                pixel_count += img_float.shape[0] * img_float.shape[1]

            out_img_name = os.path.splitext(img_name)[0] + ".png"
            img_save_path = os.path.join(args.output_dir, "images", split, out_img_name)
            mask_save_path = os.path.join(args.output_dir, "masks", split, f"mask_{out_img_name}")

            Image.fromarray(img_array).save(img_save_path)
            Image.fromarray(mask_array).save(mask_save_path)

    mean = (sum_channels / pixel_count).tolist()
    std = np.sqrt(sumsq_channels / pixel_count - np.square(mean)).tolist()
    with open(os.path.join(args.output_dir, "mean_std.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)

    print("Data preprocessing complete!")


if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)
