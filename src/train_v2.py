
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
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img=cv2.imread(self.imgs[i])[:,:,::-1]
        m=cv2.imread(self.masks[i], cv2.IMREAD_UNCHANGED)
        if self.C==1 and m.ndim==3: m=m[:,:,0]
        if self.C>1 and m.ndim==2:
            h,w=m.shape; oh=np.zeros((self.C,h,w),np.float32)
            for c in range(self.C): oh[c][m==c+1]=1
            m=oh.transpose(1,2,0)
        if self.T: tmp=self.T(image=img,mask=m); img,m=tmp['image'],tmp['mask']
        img=img.astype(np.float32)/255.; img=img.transpose(2,0,1)
        m=m.astype(np.float32)/255.
        if m.ndim==2: m=np.expand_dims(m,0)
        else: m=m.transpose(2,0,1)
        return torch.tensor(img),torch.tensor(m)

def dice_loss(l,t,eps=1e-6):
    p=torch.sigmoid(l)
    num=2*(p*t).sum((2,3))+eps
    den=p.sum((2,3))+t.sum((2,3))+eps
    return 1-(num/den).mean()

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

def run_epoch(model, loader, crit, opt, scaler, dev, train=True, amp=True):
    model.train() if train else model.eval()
    tot=d_sum=i_sum=0.
    torch.set_grad_enabled(train)
    for x,y in tqdm(loader, disable=len(loader)<10):
        x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=amp):
            out=model(x); loss=crit(out,y)+dice_loss(out,y)
        if train:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot+=loss.item(); d,i=metrics(out,y); d_sum+=d; i_sum+=i
    n=len(loader); return tot/n, d_sum/n, i_sum/n

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
    return p.parse_args()

def main():
    a=parse(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    
    g=torch.cuda.device_count() or 1
    
    model=build_model(a.arch,a.classes,a.batch_size,g).to(dev)
    
    if g>1: model=nn.DataParallel(model)
    
    pos=torch.tensor([a.pos_weight]*a.classes,device=dev) if a.classes==1 else None
    crit=FocalBCE(pos_weight=pos)
    opt=torch.optim.AdamW(model.parameters(), lr=a.lr)
    scaler=torch.amp.GradScaler(enabled=a.amp)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,a.lr*1e-2) if a.scheduler=='cosine' else None
    ds_tr=SteelDS(f'{a.data_dir}/images/train', f'{a.data_dir}/masks/train',
                  classes=a.classes, transform=aug(512,True))
    ds_va=SteelDS(f'{a.data_dir}/images/val', f'{a.data_dir}/masks/val',
                  classes=a.classes, transform=aug(512,False))
    dl_tr=DataLoader(ds_tr,batch_size=a.batch_size,shuffle=True,
                     num_workers=a.workers,pin_memory=True,drop_last=True)
    dl_va=DataLoader(ds_va,batch_size=a.batch_size,shuffle=False,
                     num_workers=a.workers,pin_memory=True)
    best=0.
    for e in range(1,a.epochs+1):
        tr,_,_=run_epoch(model,dl_tr,crit,opt,scaler,dev,True,a.amp)
        va, d, i = run_epoch(model,dl_va,crit,opt,scaler,dev,False,a.amp)
        if sched: sched.step()
        print(f'E{e}/{a.epochs}  train {tr:.4f} | val {va:.4f}  dice {d:.4f} iou {i:.4f}')
        if d>best:
            best=d; torch.save(model.state_dict(), a.save)
            print('  ✓ saved best')
if __name__=='__main__':
    main()
