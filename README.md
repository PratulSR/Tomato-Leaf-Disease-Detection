# Tomato Leaf Disease Detection

## Overview

This project focuses on the automatic detection and localisation of tomato leaf diseases using computer vision techniques. It implements a custom, lightweight object detector built from scratch using the YOLOv8 architecture. The goal is to accurately identify diseased regions on tomato leaves to assist in agricultural monitoring and disease management.

## Objectives

The primary objective is to build a robust object detector that achieves the following performance targets on the test set:
- **Box Precision:** ≥ 0.75 (IoU ≥ 0.50)
- **mAP@50:** ≥ 0.75
- **F1-Score@0.50:** optimized as an additional metric.

Additionally, the project incorporates an enhancement phase using **Fourier-based regularisation loss** to improve the detection of high-frequency texture patterns, such as small lesion edges.

## Dataset

The dataset consists of RGB images of tomato leaves with corresponding bounding box annotations in YOLO format.

### Classes
The model detects 7 categories:
0. **Bacterial Spot**
1. **Early Blight**
2. **Healthy**
3. **Late Blight**
4. **Leaf Mold**
5. **Target Spot**
6. **Black Spot**

### Structure
The data is split into:
- **Training Set:** ~600 images with annotations.
- **Test Set:** Unseen images for final evaluation.

*Note: Grayscale images are converted to RGB, and empty/invalid annotations are filtered out during preprocessing.*

## Methodology

### Model Architecture
The project utilises the **YOLOv8** (You Only Look Once) architecture, known for its speed and accuracy in real-time object detection.

### Data Augmentation
To combat the limited dataset size and improve generalisation, an extensive data augmentation pipeline is implemented using the `albumentations` library. Augmentations include:
- **Geometric:** Shift, Scale, Rotate.
- **Color/Lighting:** Random Brightness & Contrast, HSV Jitter.
- **Noise:** Gaussian Noise.
- **Resizing:** Random Scaling and Letterboxing (padding to 640x640).

### Regularisation
A Fourier-based loss term is added to the training objective to encourage the network to learn fine-grained textural features crucial for distinguishing between similar disease spots.

## Setup and Installation

The project is designed to run in a Python 3.11 environment. Key dependencies include:

- `ultralytics` (YOLOv8)
- `torch`, `torchvision`
- `opencv-python`
- `albumentations`
- `scikit-learn`
- `matplotlib`, `numpy`, `pandas`, `tqdm`

### Install Dependencies
```bash
pip install ultralytics opencv-python tqdm albumentations
```

## Usage

The entire training, evaluation, and inference pipeline is contained within the Jupyter Notebook `Code.ipynb`.

1. **Open the Notebook:**
   ```bash
   jupyter notebook Code.ipynb
   ```
2. **Run Cells:** Execute the cells sequentially to:
   - Install dependencies.
   - Set up the dataset directory structure.
   - Preprocess data (clean, resize, augment).
   - Train the YOLOv8 model.
   - Evaluate performance on the test set.
