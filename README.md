# Steel-Defect-Detection

This project implements a steel defect detection system using a UNet-based semantic segmentation model. The model identifies defects in steel surfaces from images, making it suitable for industrial automation and quality control.

## Features
- **UNet Model**: Semantic segmentation architecture for pixel-level defect detection.
- **Real-Time Inference**: Process images in real-time and save predicted masks.
- **Metrics**: Evaluate model performance using Dice coefficient and IoU.

--Note: It is still under work. I'm making this a side project but as I have limited time, I do not work on this often.
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/steel-defect-detection.git
   cd steel-defect-detection
    ```
2. Install dependencies:
```bash
    pip install -r requirements.txt
```

## Usage
### Training
decode RLE, run:

```bash
python src/decode_RLE.py
```
Preprocess the Data:

```bash
python src/data_preprocessing.py
```
Train the UNet model:

```bash
python src/train.py --data-dir processed_data --epochs 25 --model-path unet_model.pth
```
To leverage multiple GPUs, add `--multi-gpu`:

```bash
python src/train.py --data-dir processed_data --epochs 25 --model-path unet_model.pth --multi-gpu
```
## Real-Time Inference
To perform real-time inference, run:

```bash
python src/real_time.py
```
Use `--multi-gpu` to run inference on all available GPUs:

```bash
python src/real_time.py --multi-gpu
```

### Visualising
To visulaise the model performance, run:

```bash
python src/visualise.py
```

## Results
- Yet to publish

## License
This project is licensed under the MIT License.
