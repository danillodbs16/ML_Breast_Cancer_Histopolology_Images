# Deep Learning Models for Breast Cancer Histopathological Tumor Classification

## Project Overview

This project investigates the application of deep learning techniques for the classification of breast cancer histopathological images. The primary objective was to identify optimized convolutional neural network architectures capable of providing accurate binary tumor classification (benign versus malignant) from microscopy tissue images.

Different combinations of image representations and feature extraction strategies were evaluated, including grayscale imaging, RGB color channels, contour-based representations, and handcrafted statistical descriptors. Multiple progressively refined deep learning architectures based on transfer learning were developed and compared.

The study focused on achieving high classification performance while maintaining model simplicity and reducing overfitting, particularly considering the relatively limited dataset size.

---

# Dataset

## BreakHis — Breast Cancer Histopathological Dataset

The study used the **BreakHis (Breast Cancer Histopathological Database)** dataset, a publicly available benchmark dataset widely used for breast cancer image classification research.

### Dataset Description

The dataset contains high-resolution microscopic images of breast tissue samples acquired using different magnification factors. Images are labeled for both binary and multi-class tumor classification tasks.

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset Name | BreakHis |
| Image Type | Histopathological breast tissue microscopy |
| Total Images | 1,995 |
| Image Resolution | 700 × 460 pixels |
| Target Task | Binary classification |
| Magnification Used | 40× |
| Class 0 | Benign (1,370 images) |
| Class 1 | Malignant (625 images) |

---

# Tumor Classes

## Binary Classification

| Class | Description |
|---|---|
| 0 | Benign |
| 1 | Malignant |

## Multi-Class Categories Available in BreakHis

| Tumor Type |
|---|
| Adenosis |
| Ductal Carcinoma |
| Fibroadenoma |
| Lobular Carcinoma |
| Mucinous Carcinoma |
| Papillary Carcinoma |
| Phyllodes Tumor |
| Tubular Adenoma |

---

# Data Preprocessing and Feature Engineering

## Image Standardization

To ensure consistent model input dimensions, all images were centrally cropped to:

| Parameter | Value |
|---|---|
| Final Resolution | 224 × 224 pixels |

---

## Image Representations Evaluated

Images were processed using multiple spectral representations:

| Representation | Description |
|---|---|
| RGB | Full color image |
| R Channel | Red channel only |
| G Channel | Green channel only |
| B Channel | Blue channel only |
| Grayscale (L) | Single-channel luminance representation |

---

## Extracted Feature Sets

Several complementary image features were investigated as possible model inputs.

### 1. Statistical Image Features

Global statistical descriptors extracted from both the original image and its Fourier transform representation.

| Feature |
|---|
| Mean intensity |
| Standard deviation |
| Average gradient |
| Entropy |
| L1 norm |

---

### 2. Contour-Based Features

Binary contour maps representing tissue structural boundaries.

| Feature Type | Description |
|---|---|
| Contour Matrix | Binary matrix encoding contour information |

---

### 3. Full Image Representations

| Representation | Description |
|---|---|
| RGB Tensor | Full 3-channel image input |
| Grayscale Tensor | Single-channel image input |

---

# Deep Learning Architecture

## Backbone Network

All models were based on transfer learning using the pretrained DenseNet121 architecture.

| Component | Configuration |
|---|---|
| Backbone | DenseNet121 |
| Pretraining Dataset | ImageNet |
| Include Top Layers | No |
| Input Resolution | 224 × 224 |
| Transfer Learning Strategy | Frozen backbone during initial training |

---

# Data Augmentation

The same augmentation strategy was applied across all experiments.

| Augmentation Technique | Configuration |
|---|---|
| Random Flip | Horizontal and vertical |
| Random Rotation | ±20% |

---

# Model Architectures

Four progressively refined architectures were evaluated.

## Common Architecture Components

All models shared the following structure:

| Layer / Operation | Description |
|---|---|
| Input Layer | 224 × 224 grayscale image |
| Data Augmentation | Random transformations |
| DenseNet Preprocessing | ImageNet normalization |
| DenseNet121 Backbone | Feature extraction |
| GlobalAveragePooling2D | Spatial feature aggregation |
| Batch Normalization | Regularization |
| Dropout | Overfitting reduction |
| Dense Classification Layer | Feature compression |
| Sigmoid Output | Binary classification |

---

# Model Comparison

## Architectural Differences

| Model | Dense Layer Size | Dropout Rate | Objective |
|---|---|---|---|
| Model 0 (Original) | 256 neurons | 0.30 | Baseline architecture |
| Model 1 (1st Refinement) | 128 neurons | 0.15 | Reduced complexity |
| Model 2 (2nd Refinement) | 64 neurons | 0.15 | Further parameter reduction |
| Model 3 (3rd Refinement) | 64 neurons | 0.15 | Final optimized configuration |

---

# Training Configuration

## Optimization Parameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1 × 10⁻³ |
| Loss Function | Binary Crossentropy |
| Epochs | 15 |
| Batch Size | 32 |

---

# Evaluation Metrics

The following metrics were used during training and validation:

| Metric | Purpose |
|---|---|
| Recall | Sensitivity to malignant cases |
| AUC (ROC-AUC) | Classification separability |

---

# Class Imbalance Handling

Because the dataset presented significant class imbalance, additional experiments were conducted using class weighting during training.

## Class Distribution Weighting

$$
w_c = \frac{N}{2n_c}
$$

Where:

| Variable | Description |
|---|---|
| \(N\) | Total number of samples |
| \(n_c\) | Number of samples in class \(c\) |

This weighting strategy increased the contribution of minority malignant samples during optimization.

---

# Experimental Findings

## Observations

The experiments revealed several important findings:

| Observation | Result |
|---|---|
| Grayscale images | Outperformed RGB representations |
| Smaller dense layers | Reduced overfitting |
| Class weighting | Improved minority class sensitivity |
| DenseNet121 transfer learning | Provided robust tissue feature extraction |

---

# Best Performing Configuration

## Final Selected Model

| Component | Final Choice |
|---|---|
| Architecture | Model 3 |
| Input Representation | Grayscale images |
| Backbone | DenseNet121 |
| Class Weighting | Enabled |
| Task | Binary classification |

---

# Conclusion

This study demonstrates that transfer learning approaches based on DenseNet121 can effectively classify breast cancer histopathological tissue images, even with relatively limited datasets.

The results suggest that:

- grayscale representations can preserve sufficient pathological information for accurate classification;
- simpler classification heads may generalize better than larger dense architectures;
- handling class imbalance is essential for improving malignant tumor detection performance.

The final optimized model achieved the best balance between model complexity, stability, and classification capability, making it a promising framework for future computer-aided pathology systems and medical image analysis research.
