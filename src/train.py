import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
        # return torch.sigmoid(self.output(dec1))
        return self.output(dec1)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg"))]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg"))]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask


def iou_score(logits, targets, eps=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - inter
    return ((inter + eps) / (union + eps)).mean()


def dice_coeff(logits, targets, eps=1e-8):
    """logits: raw (N,1,H,W); targets: 0/1"""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return ((2 * inter + eps) / (union + eps)).mean()


def train(model, train_loader, val_loader, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0], device=device))
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        dice_scores, iou_scores = [], []
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                dice_scores.append(dice_coeff(outputs, masks).item())
                iou_scores.append(iou_score(outputs, masks).item())

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Dice: {np.mean(dice_scores):.4f}, IoU: {np.mean(iou_scores):.4f}"
        )
    return model


def visualize_sample(model, dataset, device, output_path="sample_prediction.png"):
    import matplotlib.pyplot as plt

    model.eval()
    image, mask = dataset[0]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
    # pred = (pred > 0.5).float()[0, 0].cpu().numpy()
    pred = (torch.sigmoid(pred) > 0.5).float()[0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image.permute(1, 2, 0).numpy())
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(mask[0].numpy(), cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for steel defect detection")
    parser.add_argument("--data-dir", type=str, default="processed_data", help="Path to processed_data directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--model-path", type=str, default="unet_model.pth", help="Where to save the trained model")
    parser.add_argument("--visualize", action="store_true", help="Save a sample prediction image after training")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs with DataParallel",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        print(f"Using device: {device}")

    train_imgs = os.path.join(args.data_dir, "images", "train")
    train_masks = os.path.join(args.data_dir, "masks", "train")
    val_imgs = os.path.join(args.data_dir, "images", "val")
    val_masks = os.path.join(args.data_dir, "masks", "val")

    train_dataset = SegmentationDataset(train_imgs, train_masks)
    val_dataset = SegmentationDataset(val_imgs, val_masks)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet()
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    model = train(model, train_loader, val_loader, device, args.epochs)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, args.model_path)
    print(f"Model saved to {args.model_path}")

    if args.visualize:
        visualize_sample(model, val_dataset, device)


if __name__ == "__main__":
    main()
