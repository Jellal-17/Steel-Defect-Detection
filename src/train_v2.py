
#!/usr/bin/env python3
"""version 2 - steel defect segmentation trainer"""

import os, argparse
from glob import glob
import cv2, numpy as np
import torch, albumentations as A
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import scipy.ndimage as ndi 

# optional heavy dep
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# ------------------ MODELS -------------------------
class DoubleConv(nn.Sequential):
    def __init__(self, inc, outc, midc=None, norm=nn.BatchNorm2d):
        midc = midc or outc
        super().__init__(
            nn.Conv2d(inc, midc, 3, 1, 1, bias=False),
            norm(midc), nn.ReLU(inplace=True),
            nn.Conv2d(midc, outc, 3, 1, 1, bias=False),
            norm(outc), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, classes=1, base=64, use_gn=False):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(8,c)) if use_gn else nn.BatchNorm2d
        self.inc   = DoubleConv(3, base, norm=Norm)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2, norm=Norm))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4, norm=Norm))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8, norm=Norm))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16, norm=Norm))
        self.up1   = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.conv1 = DoubleConv(base*16, base*8, norm=Norm)
        self.up2   = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.conv2 = DoubleConv(base*8, base*4, norm=Norm)
        self.up3   = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.conv3 = DoubleConv(base*4, base*2, norm=Norm)
        self.up4   = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.conv4 = DoubleConv(base*2, base, norm=Norm)
        self.outc  = nn.Conv2d(base, classes, 1)
    def forward(self,x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2)
        x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5); x=self.conv1(torch.cat([x,x4],1))
        x=self.up2(x);  x=self.conv2(torch.cat([x,x3],1))
        x=self.up3(x);  x=self.conv3(torch.cat([x,x2],1))
        x=self.up4(x);  x=self.conv4(torch.cat([x,x1],1))
        return self.outc(x)  # raw logits

def build_model(arch, classes, batch, gpus):
    if arch=='unet':
        return UNet(classes, use_gn=(batch//gpus)<8)
    if arch=='resunetpp':
        if smp is None:
            raise RuntimeError('segmentation_models_pytorch required for ResUNet++')
        return smp.UnetPlusPlus('resnet34', encoder_weights='imagenet',
                                in_channels=3, classes=classes, activation=None)
    raise ValueError(arch)

# ---------------- DATASET --------------------------
def aug(size, train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5), A.RandomBrightnessContrast(p=0.2),
            A.CLAHE(p=0.2),
            A.CoarseDropout(num_holes_range=(1,8), hole_height_range=(10,32),
                            hole_width_range=(10,32), p=0.5),
            A.Resize(size,size)
        ])
    return A.Resize(size,size)

class SteelDS(Dataset):
    def __init__(self, img_dir, mask_dir, classes=1, transform=None):
        self.imgs=sorted(glob(os.path.join(img_dir,'*')))
        self.masks=sorted(glob(os.path.join(mask_dir,'*')))
        assert len(self.imgs)==len(self.masks)
        self.C=classes; self.T=transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,i):
        img=cv2.imread(self.imgs[i])[:,:,::-1]
        m=cv2.imread(self.masks[i], cv2.IMREAD_UNCHANGED)
        
        if self.C==1 and m.ndim==3: 
            m=m[:,:,0]
            
        if self.C>1 and m.ndim==2:
            h,w=m.shape; oh=np.zeros((self.C,h,w),np.float32)
            for c in range(self.C):
                oh[c][m > 0] = 1 
            m=oh.transpose(1,2,0)
        if self.T: tmp=self.T(image=img,mask=m); img,m=tmp['image'],tmp['mask']
       
        img=img.astype(np.float32)/255.; img=img.transpose(2,0,1)
        
        m=m.astype(np.float32)/255.
        
        if m.ndim==2: 
            m=np.expand_dims(m,0)
        else: 
            m=m.transpose(2,0,1)
        return torch.tensor(img),torch.tensor(m)

def dice_loss(l,t,eps=1e-6):
    p=torch.sigmoid(l)
    num=2*(p*t).sum((2,3))+eps
    den=p.sum((2,3))+t.sum((2,3))+eps
    return 1-(num/den).mean()

def post_process(bin_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Morphologically close the mask and strip blobs smaller than min_area pixels.
    Expects a uint8 0/1 array; returns the same shape array.
    """
    mask = ndi.binary_closing(bin_mask, structure=np.ones((3, 3))).astype(np.uint8)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    for i, s in enumerate(sizes, start=1):
        if s < min_area:
            mask[lab == i] = 0
    return mask

class FocalBCE(nn.Module):
    def __init__(self,gamma=2,pos_weight=None):
        super().__init__(); self.g=gamma
        self.bce=nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none')
    def forward(self,l,t):
        b=self.bce(l,t); pt=torch.exp(-b)
        return ((1-pt)**self.g*b).mean()

def metrics(l,t,thr=0.5,eps=1e-6):
    p=(torch.sigmoid(l)>thr).float()
    inter=(p*t).sum((2,3))
    union=p.sum((2,3))+t.sum((2,3))-inter
    dice=(2*inter+eps)/(p.sum((2,3))+t.sum((2,3))+eps)
    iou=(inter+eps)/(union+eps)
    return dice.mean().item(), iou.mean().item()

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    train: bool = True,
    amp: bool = True,
    min_area: int = 0,            # ← new (0 = no post-proc)
):
    """Train or evaluate one epoch and return (loss, dice, iou)."""
    mode = "train" if train else "val"
    model.train() if train else model.eval()

    total_loss, dice_sum, iou_sum = 0.0, 0.0, 0.0
    torch.set_grad_enabled(train)

    for imgs, masks in tqdm(loader, desc=f"{mode}-loop", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        # ----- forward & loss (mixed precision if enabled) -------------
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(imgs)
            loss = criterion(logits, masks) + dice_loss(logits, masks)

        # ----- backward / optimiser update -----------------------------
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        # ----- prediction → binary → optional post-processing ----------
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()

            if min_area > 0:
                # loop on CPU for scipy.ndimage
                cleaned = []
                for sample in preds.cpu().numpy():
                    chans = []
                    for ch in sample:
                        chans.append(post_process(ch.astype(np.uint8), min_area))
                    cleaned.append(np.stack(chans))
                preds = torch.tensor(np.array(cleaned), device=device)

            d, i = metrics(preds, masks)
            dice_sum += d
            iou_sum += i

    n_batches = len(loader)
    return total_loss / n_batches, dice_sum / n_batches, iou_sum / n_batches


# ---------------- MAIN ----------------------------
def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='processed_data')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--classes',type=int,default=1,choices=[1,4])
    p.add_argument('--arch',choices=['unet','resunetpp'],default='unet')
    p.add_argument('--pos-weight',type=float,default=30)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--save',default='best_model.pth')
    p.add_argument('--scheduler',choices=['cosine','none'],default='cosine')
    p.add_argument("--postprocess-min-area",type=int, default=50, help=">0: apply morphological closing and remove blobs below this area (px)")
    return p.parse_args()

def main() -> None:

    args   = parse()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu  = torch.cuda.device_count() or 1
    print(f"☑ Using {n_gpu} GPU(s) on {device}")

    model = build_model(args.arch, args.classes, args.batch_size, n_gpu).to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    pos_weight = (
        torch.tensor([args.pos_weight] * args.classes, device=device)
        if args.classes == 1              # binary case
        else None                         # multi-class → leave at None
    )

    criterion = FocalBCE(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=args.amp)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=args.epochs, T_mult=1, eta_min=args.lr * 1e-2
        )
        if args.scheduler == "cosine"
        else None
    )

    train_set = SteelDS(f"{args.data_dir}/images/train",f"{args.data_dir}/masks/train",classes=args.classes,transform=aug(768, train=True),)
    val_set = SteelDS(f"{args.data_dir}/images/val",f"{args.data_dir}/masks/val",classes=args.classes,transform=aug(768, train=False),)

    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True,)
    val_loader = DataLoader(val_set,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model,train_loader,criterion,optimiser,scaler,device,train=True,amp=args.amp,min_area=args.postprocess_min_area)

        val_loss, dice, iou = run_epoch(model,val_loader,criterion,optimiser,scaler,device,train=False,amp=args.amp,min_area=args.postprocess_min_area)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:2d}/{args.epochs} │ "
            f"train {train_loss:.4f} │ val {val_loss:.4f} │ "
            f"dice {dice:.4f} │ IoU {iou:.4f}")

        # Save best model (highest Dice)
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), args.save)
            print(f"  ✓ Saved new best (Dice {best_dice:.4f}) to {args.save}")

if __name__=='__main__':
    main()
