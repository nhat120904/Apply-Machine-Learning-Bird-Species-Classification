# Bird Species Classification using CUB-200 Dataset

## Overview

This project focuses on the classification of 200 bird species using the Caltech-UCSD Birds 200 (CUB-200) dataset. Various machine learning models, including traditional algorithms and deep learning architectures, are applied to achieve this multi-class classification. Techniques to mitigate issues such as class imbalance and background clutter are also explored, particularly through preprocessing with the YOLOv8 object detection model.

## Dataset

- **Dataset**: [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Training Images**: 4,829
- **Test Images**: 1,204

## Methodology

1. **Data Preprocessing**:
   - Images are resized to 256x256 pixels.
   - YOLOv8 is used for bird detection and background cropping.

2. **Models Used**:
   - **Traditional ML Models**: Random Forest, Decision Tree, Support Vector Machine (SVM).
   - **Deep Learning Models**: ResNet50, MobileNetV2, EfficientNet (B0, B3, B5).

3. **Training Process**:
   - Cross-entropy loss function.
   - AdamW optimizer with learning rate adjustments using ReduceLROnPlateau.
   - Trained for 25 epochs with batch size of 16.

## Results

- **Traditional Models**:
  - SVM achieved the highest accuracy at 67.57% on the original dataset.
- **Deep Learning Models**:
  - EfficientNet B5 achieved the highest Top-1 accuracy (81.79%) on the preprocessed dataset.

## Dependencies

- Python 3.x
- PyTorch / TensorFlow
- YOLOv8
- TensorBoard (for logging)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/nhat120904/Apply-Machine-Learning-Bird-Species-Classification.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the training script:
   ```bash
   python train.py
