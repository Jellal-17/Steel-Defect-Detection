# sample_prediction.py – quick inference helper for train_improved models
# ------------------------------------------------------------
# Usage example:
#   python sample_prediction.py \
#       --images-dir demo_images \
#       --weights best_model.pth \
#       --arch resunetpp --classes 1 \
#       --out-dir preds --threshold 0.5 --tta flip
# ------------------------------------------------------------

import os, argparse, cv2, numpy as np, torch
from glob import glob
from tqdm import tqdm
import albumentations as A

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

# ---------- minimal model factory (same as train_improved) -------------
class DoubleConv(torch.nn.Sequential):
    def __init__(self, ic, oc, mc=None, norm=torch.nn.BatchNorm2d):
        mc = mc or oc
        super().__init__(
            torch.nn.Conv2d(ic, mc, 3, 1, 1, bias=False),
            norm(mc), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mc, oc, 3, 1, 1, bias=False),
            norm(oc), torch.nn.ReLU(inplace=True))

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
    if arch=='unet':
        return UNet(classes)
    if arch=='resunetpp':
        if smp is None:
            raise RuntimeError("Install segmentation_models_pytorch for ResUNet++")
        return smp.UnetPlusPlus('resnet34', encoder_weights=None, in_channels=3, classes=classes)
    raise ValueError(arch)

# ------------------------- post‑process helpers -------------------------
import scipy.ndimage as ndi

def post_process(mask, min_area=50):
    # Remove tiny blobs and close small holes
    mask = ndi.binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n+1))
    for i, s in enumerate(sizes, start=1):
        if s < min_area:
            mask[lab == i] = 0
    return mask

# ------------------------- main ----------------------------------------

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--images-dir', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--arch', choices=['unet','resunetpp'], default='unet')
    p.add_argument('--classes', type=int, default=1)
    p.add_argument('--out-dir', default='predictions')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--tta', choices=['none','flip'], default='none')
    p.add_argument('--postprocess', action='store_true')
    return p.parse_args()


def inference():
    a=parse(); os.makedirs(a.out_dir, exist_ok=True)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(a.arch,a.classes).to(dev)
    state=torch.load(a.weights, map_location='cpu'); model.load_state_dict(state)
    model.eval()
    tf=A.Resize(512,512)

    img_paths = sorted(glob(os.path.join(a.images_dir,'*')))
    for p in tqdm(img_paths):
        orig=cv2.imread(p)[:,:,::-1]
        h0,w0,_=orig.shape
        img=tf(image=orig)['image'].astype(np.float32)/255.
        inp=torch.tensor(img.transpose(2,0,1))[None].to(dev)
        with torch.no_grad():
            pred=torch.sigmoid(model(inp))
            if a.tta=='flip':
                inp_h=torch.flip(inp,[-1])
                pred+=torch.sigmoid(model(inp_h)).flip(-1)
                inp_v=torch.flip(inp,[-2])
                pred+=torch.sigmoid(model(inp_v)).flip(-2)
                pred/=3
        mask= (pred[0,0].cpu().numpy()>a.threshold).astype(np.uint8)
        if a.postprocess:
            mask=post_process(mask)
        mask=cv2.resize(mask,(w0,h0), interpolation=cv2.INTER_NEAREST)*255
        out_name=os.path.join(a.out_dir, os.path.basename(p).split('.')[0]+'_pred.png')
        cv2.imwrite(out_name, mask)

if __name__=='__main__':
    inference()
