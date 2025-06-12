#!/usr/bin/env python3
"""sample_prediction.py  —  predict masks **and** compute metrics (Dice, IoU, pixel-Acc, Precision, Recall)

If you pass only ``--images-dir`` the script writes predicted masks.
If you *also* provide ``--masks-dir`` with ground-truth PNGs, it
aggregates TP / FP / FN / TN over the whole set and prints the metrics.

Example : generate masks *and* stats on your validation split
-------------------------------------------------------------
```bash
python sample_prediction.py \
    --images-dir processed_data/images/val \
    --masks-dir  processed_data/masks/val \
    --weights    best_model.pth \
    --arch       resunetpp \
    --classes    1 \
    --out-dir    predictions \
    --tta        flip \
    --postprocess-min-area 50
```
"""

import os, argparse, cv2, numpy as np, torch, albumentations as A, scipy.ndimage as ndi
from glob import glob
from tqdm import tqdm

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# ─────────────────── model factory (same as train_improved) ────────────
class DoubleConv(torch.nn.Sequential):
    def __init__(self, ic, oc, mc=None, norm=torch.nn.BatchNorm2d):
        mc = mc or oc
        super().__init__(
            torch.nn.Conv2d(ic, mc, 3, 1, 1, bias=False),
            norm(mc), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mc, oc, 3, 1, 1, bias=False),
            norm(oc), torch.nn.ReLU(inplace=True),
        )

class UNet(torch.nn.Module):
    def __init__(self, classes=1, base=64):
        super().__init__()
        Norm = torch.nn.BatchNorm2d
        self.inc   = DoubleConv(3, base, norm=Norm)
        self.down1 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base, base*2, norm=Norm))
        self.down2 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base*2, base*4, norm=Norm))
        self.down3 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base*4, base*8, norm=Norm))
        self.down4 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(base*8, base*16, norm=Norm))
        self.up1   = torch.nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.conv1 = DoubleConv(base*16, base*8, norm=Norm)
        self.up2   = torch.nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.conv2 = DoubleConv(base*8, base*4, norm=Norm)
        self.up3   = torch.nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv3 = DoubleConv(base*4, base*2, norm=Norm)
        self.up4   = torch.nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv4 = DoubleConv(base*2, base, norm=Norm)
        self.outc  = torch.nn.Conv2d(base, classes, 1)
    def forward(self,x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2)
        x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5); x=self.conv1(torch.cat([x,x4],1))
        x=self.up2(x);  x=self.conv2(torch.cat([x,x3],1))
        x=self.up3(x);  x=self.conv3(torch.cat([x,x2],1))
        x=self.up4(x);  x=self.conv4(torch.cat([x,x1],1))
        return x

def build_model(arch, classes):
    if arch == "unet":
        return UNet(classes)
    if arch == "resunetpp":
        if smp is None:
            raise RuntimeError("pip install segmentation_models_pytorch")
        return smp.UnetPlusPlus('resnet34', encoder_weights=None, in_channels=3, classes=classes)
    raise ValueError(arch)

# ────────────────── post‑process helper ────────────────────────────────
def post_process(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Morph-close and drop blobs < min_area px (binary uint8 mask)"""
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    for i, s in enumerate(sizes, start=1):
        if s < min_area:
            mask[lab == i] = 0
    return mask

# ────────────────── CLI ────────────────────────────────────────────────

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--arch', choices=['unet','resunetpp'], default='unet')
    p.add_argument('--classes', type=int, default=1)
    p.add_argument('--out-dir', default='predictions')
    p.add_argument('--masks-dir', help='Optional ground-truth masks folder (PNG)')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--tta', choices=['none','flip'], default='none')
    p.add_argument('--postprocess-min-area', type=int, default=0)
    return p.parse_args()

# ────────────────── metric helpers ─────────────────────────────────────

def update_conf(pred: np.ndarray, gt: np.ndarray, conf):
    conf['TP'] += np.logical_and(pred,  gt).sum()
    conf['FP'] += np.logical_and(pred, ~gt).sum()
    conf['FN'] += np.logical_and(~pred, gt).sum()
    conf['TN'] += np.logical_and(~pred, ~gt).sum()

# ────────────────── main ------------------------------------------------

def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(args.arch, args.classes).to(device)

    state = torch.load(args.weights, map_location='cpu')
    if next(iter(state)).startswith('module.'):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    resize = A.Resize(512, 512)
    conf   = {'TP':0,'FP':0,'FN':0,'TN':0} if args.masks_dir else None

    img_paths = sorted(glob(os.path.join(args.images_dir, '*')))
    for img_path in tqdm(img_paths, desc='Predict'):
        img0 = cv2.imread(img_path)[:, :, ::-1]
        h0, w0 = img0.shape[:2]
        img_r = resize(image=img0)['image'].astype(np.float32) / 255.0
        inp = torch.tensor(img_r.transpose(2, 0, 1))[None].to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp))
            if args.tta == 'flip':
                pred += torch.sigmoid(model(torch.flip(inp, [-1]))).flip(-1)
                pred += torch.sigmoid(model(torch.flip(inp, [-2]))).flip(-2)
                pred /= 3

        mask = (pred[0, 0].cpu().numpy() > args.threshold).astype(np.uint8)
        if args.postprocess_min_area > 0:
            mask = post_process(mask, args.postprocess_min_area)

        # save resized prediction PNG
        mask_resized = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST) * 255
        out_name = os.path.join(args.out_dir, os.path.splitext(os.path.basename(img_path))[0] + '_pred.png')
        cv2.imwrite(out_name, mask_resized)

        # update confusion if GT available
        if conf is not None:
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(args.masks_dir, f"mask_{base}.png")
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0
                pred_bin = cv2.resize(mask, gt.shape[::-1],
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
                update_conf(pred_bin, gt, conf)

    # print once, after all images    
    if conf is not None:
        TP, FP, FN, TN = conf['TP'], conf['FP'], conf['FN'], conf['TN']
        eps  = 1e-6
        dice = 2 * TP / (2 * TP + FP + FN + eps)
        iou  =     TP / (    TP + FP + FN + eps)
        acc  = (TP + TN) / (TP + TN + FP + FN + eps)
        prec = TP / (TP + FP + eps)
        rec  = TP / (TP + FN + eps)

        print("\n── Validation metrics ──────────────────────")
        print(f"Dice      : {dice:.4f}")
        print(f"IoU       : {iou:.4f}")
        print(f"Pixel Acc : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}\n")

if __name__ == '__main__':
    main()
