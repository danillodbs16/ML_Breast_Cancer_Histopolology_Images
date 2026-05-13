# Deep Learning for Breast Cancer Histopathological Classification

**Author:** Danillo Barros de Souza  
**Email:** danillo.dbs16@gmail.com  
**GitHub:** https://github.com/danillodbs16/Breast_Cancer_Analysis  

---

## Repository Structure

```
├── img/                # Figures and visualizations used in the report
├── src/                # Source code
│   ├── data/           # Project data
│   ├── resources/      # Auxiliary resources
│   ├── notebooks/      # Jupyter notebooks for analysis and exploration
│   ├── utils/          # Helper functions and utilities
│   ├── data/           # Processed datasets
│   ├── html/           # Interactive HTML visualizations
│   └── dashboard/      # Exported dashboard files (e.g., Superset)
├── extra/              # Additional outputs (HTML visualizations and dashboards)
```

## Overview

This project applies deep learning techniques to classify breast cancer histopathological images into benign and malignant categories. It explores different image representations, feature engineering strategies, and transfer learning architectures to optimize classification performance under limited data conditions.

A series of progressively refined models based on DenseNet121 (ImageNet pretrained) were evaluated, focusing on improving generalization while reducing overfitting.

---

## Dataset

The BreakHis (Breast Cancer Histopathological Dataset) was used in this study.

- 1,995 histopathological images  
- 700 × 460 resolution  
- Magnification: 40×  
- Classes:
  - Benign (1,370 images)
  - Malignant (625 images)

[Link Here](https://www.kaggle.com/code/nasrulhakim86/breast-cancer-histopathology-images-classificationhttps://www.kaggle.com/code/nasrulhakim86/breast-cancer-histopathology-images-classification)
---

## Methodology

### Image Representations
- RGB images  
- Individual R, G, B channels  
- Grayscale (luminance-based)

### Feature Engineering
- Statistical descriptors (mean, std, entropy, gradient, L1 norm)
- Fourier-based features
- Contour-based binary maps

### Model Architecture

All experiments used DenseNet121 (transfer learning) with:

- Frozen convolutional backbone
- Global Average Pooling
- Batch Normalization
- Dropout regularization
- Fully connected classification head
- Sigmoid output for binary classification

---

## Training Configuration

- Optimizer: Adam  
- Learning rate: 1e-3  
- Loss function: Binary Crossentropy  
- Epochs: 15  
- Batch size: 32  
- Class imbalance handling: Class weighting  

---

## Key Findings

- Grayscale inputs outperformed RGB representations  
- Smaller dense layers improved generalization  
- Class weighting significantly improved malignant recall  
- DenseNet121 provided strong feature extraction for histopathology images  

---

## Best Model

- Architecture: Model 3  
- Backbone: DenseNet121  
- Input: Grayscale images  
- Class weighting: Enabled  
- Task: Binary classification  

---

## Conclusion

Transfer learning with DenseNet121 is effective for breast cancer histopathology classification. Simpler classification heads and grayscale preprocessing improved generalization, while class imbalance handling was critical for detecting malignant cases.

---

