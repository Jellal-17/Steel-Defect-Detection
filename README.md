# Steel-Defect-Detection

This project implements a steel defect detection system using a UNet-based semantic segmentation model. The model identifies defects in steel surfaces from images, making it suitable for industrial automation and quality control.

## Features
- **Configurable architectures**  `--arch unet | resunetpp`
- **Binary *or* 4-class masks**  `--classes 1 | 4`
- **Mixed-precision & multi-GPU auto-detect**
- **Cosine warm-restart scheduler**
- **Optional post-processing** (morph close + tiny-blob removal)
- **Sample-level TTA** in the provided `sample_prediction.py`

<!---- Project is now feature-complete – edit below as you update checkpoints. -->
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/steel-defect-detection.git
   cd steel-defect-detection
    ```
2. Install dependencies:
```bash
    pip install -r requirements.txt
    pip install segmentation_models_pytorch scipy
```

## Usage
### Training
#### 1  Decode the Kaggle RLE masks
```bash
python src/decode_RLE.py --train_csv train.csv --mask_output mask_output
```
#### 2 Offline pre-processing (resize ✚ split)
```bash
python src/data_preprocessing.py \
    --image-dir train_images \
    --mask-dir  mask_output \
    --output-dir processed_data \
    --img-size   768 \
    --val-split  0.2 \
    --augment
```
#### 3 Train (binary U-Net example)
```bash
python src/train_improved.py \
    --data-dir processed_data \
    --arch unet          \
    --classes 1          \
    --epochs 40 \
    --batch-size 32 \
    --workers 8 \
    --amp
```
GPU usage is automatic – the script wraps in nn.DataParallel if >1 GPU is visible.

#### Multi-class + ResUNet++
```bash
python src/train_improved.py \
    --data-dir processed_data \
    --arch resunetpp \
    --classes 4 \
    --epochs 50 \
    --batch-size 24 \
    --scheduler cosine \
    --postprocess-min-area 50
```
### Inference / sample prediction
```bash
python src/sample_prediction.py \
    --images-dir demo_images \
    --weights     best_model.pth \
    --arch        resunetpp \
    --classes     1 \
    --tta         flip \
    --postprocess
```
Predicted masks will be written to predictions/ with blob-filtering and TTA applied.

### Post-processing
The helper erodes small false-positive speckles by
1) morphological closing and  
2) dropping blobs smaller than `--postprocess-min-area` pixels (default = 50).

### Visualising
To visulaise the model performance, run:

```bash
python src/visualise.py
```

## Results
Binary ResUNet++ @768² ⇒ Dice 0.57  /  IoU 0.50 (val split 20 %)

## License
This project is licensed under the MIT License.
