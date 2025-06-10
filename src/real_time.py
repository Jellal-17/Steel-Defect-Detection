import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.pool1(enc1))
        enc3 = self.enc_conv3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.dec_conv3(torch.cat([self.up3(bottleneck), enc3], dim=1))
        dec2 = self.dec_conv2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec_conv1(torch.cat([self.up1(dec2), enc1], dim=1))
        return torch.sigmoid(self.output(dec1))


def preprocess_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image_tensor = to_tensor(image).unsqueeze(0)
    return image_tensor


def postprocess_mask(predicted_mask, original_shape):
    predicted_mask = (predicted_mask > 0.5).float().cpu().numpy()[0, 0]
    predicted_mask_resized = cv2.resize(predicted_mask, original_shape[::-1])
    return (predicted_mask_resized * 255).astype(np.uint8)


def real_time_inference(input_directory, output_directory, model_path, multi_gpu=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using device: {device}")

    model = UNet()
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(input_directory):
        if file_name.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(input_directory, file_name)
            original_image = cv2.imread(image_path)
            original_shape = original_image.shape[:2]

            image_tensor = preprocess_image(image_path).to(device)

            with torch.no_grad():
                predicted_mask = model(image_tensor)

            processed_mask = postprocess_mask(predicted_mask, original_shape)
            output_path = os.path.join(output_directory, f"mask_{file_name}")
            cv2.imwrite(output_path, processed_mask)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(processed_mask, cmap="gray")
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    print(f"All masks saved to '{output_directory}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained UNet on images")
    parser.add_argument("--input-dir", type=str, default="real_time_images", help="Directory with input images")
    parser.add_argument("--output-dir", type=str, default="real_time_predictions", help="Where to save predicted masks")
    parser.add_argument("--model-path", type=str, default="unet_model.pth", help="Path to trained model weights")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs with DataParallel",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    real_time_inference(
        args.input_dir, args.output_dir, args.model_path, multi_gpu=args.multi_gpu
    )
