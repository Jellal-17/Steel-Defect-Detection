# Steel-Defect-Detection

This project implements a steel defect detection system using a UNet-based semantic segmentation model. The model identifies defects in steel surfaces from images, making it suitable for industrial automation and quality control.

## Features
- **UNet Model**: Semantic segmentation architecture for pixel-level defect detection.
- **Real-Time Inference**: Process images in real-time and save predicted masks.
- **Metrics**: Evaluate model performance using Dice coefficient and IoU.


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
To train the model, run:

```bash
python src/train.py
```
### Evaluation
To evaluate the model, run:

```bash
python src/evaluate.py
```

## Real-Time Inference
To perform real-time inference, run:

```bash
python src/real_time_inference.py
```

## Results
- Validation Dice Coefficient: XX.XX
- Validation IoU: XX.XX

## License
This project is licensed under the MIT License.