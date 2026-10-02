# CIFAR-10 Classification and OOD Detection

A PyTorch-based computer vision project that trains a **Vision Transformer (ViT)** and a **CIFAR-adapted ResNet-50** from scratch on CIFAR-10, followed by **Out-of-Distribution (OOD) detection** using MSP and Energy-based methods.

## Overview

This project focuses on two tasks:

1. **Closed-set classification** on CIFAR-10 using ViT and ResNet-50.
2. **OOD detection** to distinguish CIFAR-10 in-distribution samples from CIFAR-100 out-of-distribution samples.

Both models are trained directly on **32×32 CIFAR-10 images without ImageNet pretraining**.

## Models

### Vision Transformer

The ViT is implemented from scratch and adapted for CIFAR-10's small image resolution.

Key components:

* 32×32 input resolution
* Patch-based tokenization
* 12 Transformer blocks
* 8 attention heads
* Pre-LayerNorm architecture
* LayerScale
* Stochastic Depth
* Approximately **9.5M parameters**

### CIFAR-Adapted ResNet-50

The standard ResNet-50 architecture is modified for CIFAR-10:

* 3×3 convolutional stem
* No initial max-pooling layer
* 32×32 input resolution
* Approximately **23.5M parameters**

## Training

Both architectures use a matched training recipe to provide a fair comparison.

### Data Augmentation

* RandAugment
* Random Erasing
* MixUp
* CutMix
* Label smoothing

### Optimization

* AdamW
* Linear warmup
* Cosine learning-rate decay
* Gradient clipping
* Automatic Mixed Precision (AMP)

## Results

| Model                   | Parameters | Test Accuracy |
| ----------------------- | ---------: | ------------: |
| Vision Transformer      |      ~9.5M |    **85.46%** |
| CIFAR-Adapted ResNet-50 |     ~23.5M |    **93.50%** |

The results demonstrate the difference in performance between a Transformer-based architecture and a convolutional architecture when both are trained from scratch on the small CIFAR-10 image resolution.

## OOD Detection

CIFAR-100 is used as a surrogate **Out-of-Distribution (OOD)** dataset, while CIFAR-10 acts as the **In-Distribution (ID)** dataset.

Two OOD scoring methods are implemented:

### 1. Maximum Softmax Probability (MSP)

MSP uses the maximum predicted class probability as the confidence score.

Low confidence indicates that the sample may be OOD.

### 2. Energy Score

The Energy-based method uses the model's logits to produce an energy score for identifying samples that differ from the training distribution.

## OOD Evaluation

The OOD detectors are evaluated using:

* **AUROC**
* **FPR@95TPR**
* **Open-set accuracy**

Thresholds are calibrated using held-out in-distribution validation data before evaluating OOD samples.

## Project Structure

```text
cifar10_ood_project/
│
├── models/
│   ├── vit.py
│   └── resnet.py
│
├── training/
│   ├── train_vit.py
│   └── train_resnet.py
│
├── ood/
│   ├── msp.py
│   └── energy.py
│
├── evaluation/
│   └── evaluate.py
│
├── checkpoints/
│
├── local_evaluate.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses:

* **CIFAR-10** for in-distribution classification and evaluation.
* **CIFAR-100** as the surrogate OOD dataset.

PyTorch/Torchvision can download the datasets automatically.

## Training

Train the Vision Transformer:

```bash
python train_vit.py
```

Train the CIFAR-adapted ResNet-50:

```bash
python train_resnet.py
```

## OOD Evaluation

Evaluate an OOD detection method using:

```bash
python local_evaluate.py --method msp
```

Available methods include:

```text
msp
energy
```

## Key Features

* Vision Transformer implemented from scratch
* CIFAR-adapted ResNet-50
* No ImageNet pretraining
* Training directly on 32×32 images
* Modern data augmentation and regularization
* MSP-based OOD detection
* Energy-based OOD detection
* CIFAR-100 surrogate OOD evaluation
* AUROC and FPR@95TPR evaluation
* PyTorch implementation

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* scikit-learn
* Matplotlib

## Results Summary

The project compares two fundamentally different architectures under the same CIFAR-10 training setting:

**Vision Transformer**

> 85.46% test accuracy

**CIFAR-adapted ResNet-50**

> 93.50% test accuracy

The OOD experiments further evaluate whether the trained classifiers can distinguish CIFAR-10 samples from samples belonging to the CIFAR-100 distribution.

## Future Improvements

Potential extensions include:

* Testing additional OOD scoring methods
* Evaluating on more challenging OOD datasets
* Calibration of confidence scores
* Comparing additional Transformer and CNN architectures
* Evaluating performance under common image corruptions

## Author

**Anshul Thakur**

If you found this project useful, consider giving the repository a ⭐.
