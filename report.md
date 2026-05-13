Deep Learning models for tumor classification of breast cancer histopatology images.

Here, we investigate histopatological image patterns through a deep learning aiming onoptimal models that are able to provide precise breast tumeor classification. We tested different suggested models (from complex and simple) and different image features (color channel, statists, contour detection).

Dataset: BreakHis

BreakHis - Breast Cancer Histopathological Dataset

Description
The BreakHis - Breast Cancer Histopathological Dataset is a valuable resource for medical image analysis, particularly in the classification of breast cancer. This dataset contains high-resolution histopathological images of breast tissue, divided into both binary and multi-class labels, to support the development and evaluation of machine learning models in cancer classification.

Context and Sources
Source: The dataset was initially developed by the P&D Laboratory - Pathological Anatomy and Cytopathology, in collaboration with the University of Porto, to support research on breast cancer diagnosis through digital pathology. The dataset is freely accessible for non-commercial research and educational purposes.
Dataset Structure: The dataset contains images organized by:
Classification Type: Binary classification (Benign vs. Malignant) and Multi-Class classification (8 different tumor types).
Magnification Levels: Images are available at 40X, 100X, 200X, and 400X magnifications, allowing models to learn across varying levels of tissue detail.
Classes:
Binary: Benign and Malignant categories.
Multi-Class: Adenosis, Ductal Carcinoma, Fibroadenoma, Lobular Carcinoma, Mucinous Carcinoma, Papillary Carcinoma, Phyllodes Tumor, and Tubular Adenoma.
Inspiration
This dataset is ideal for various research and practical applications, including:

Binary Classification: Distinguish between benign and malignant breast tissue, a crucial step in cancer diagnosis.
Multi-Class Classification: Identify specific tumor types, aiding in the development of models that can support pathologists in detailed cancer analysis.

Total images: 1995
Image shapes: 700 x 460
Class 0: Beningnant (1370)
Class 1: Malign (625)

Target images: 40X zoom images

Data treatment:

Due to different image shape issues, we decided to crop the images in a 224x224 pixel resolution in the central image area.
We than use PIL for reading the images as numpy arrays in different spectras ("R","G","B", "RGB", and gray scale- "L")
We further extract image information as possible imputs for the models, namelly:

1) Image global statistical features (average, standard deviation, average gradiant, entropy, L1 norm, for both original image and its Fourrier transform);
2) Contour image information (matrix of 0's and 1's mapping each image contour information);
3) Full RGB color channel image information;
4) Full gray channel image information;


Models:

We used 4 deep learning model that used a pre-trained ImageNet model as a basis:

1) Model 0 (Original):

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
   # layers.RandomZoom(0.10),
   # layers.RandomContrast(0.10),
])

# =========================================================
# PRETRAINED BACKBONE (DenseNet121)
# =========================================================

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze backbone initially
base_model.trainable = False

# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Augmentation
x = data_augmentation(inputs)

# DenseNet preprocessing
x = tf.keras.applications.densenet.preprocess_input(x)

# Backbone
x = base_model(x, training=False)

# Better than Flatten for histopathology
x = layers.GlobalAveragePooling2D()(x)

# Regularization
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Dense layer for high-level tissue patterns
x = layers.Dense(256, activation='relu')(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Binary classification
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

2) Model 1 (1st Refinement model)

 =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
   # layers.RandomZoom(0.10),
   # layers.RandomContrast(0.10),
])

# =========================================================
# PRETRAINED BACKBONE (DenseNet121)
# =========================================================

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze backbone initially
base_model.trainable = False

# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Augmentation
x = data_augmentation(inputs)

# DenseNet preprocessing
x = tf.keras.applications.densenet.preprocess_input(x)

# Backbone
x = base_model(x, training=False)

# Better than Flatten for histopathology
x = layers.GlobalAveragePooling2D()(x)

# Regularization
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Dense layer for high-level tissue patterns
#x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Binary classification
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

3) Model 2 (2nd model refinement)

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
   # layers.RandomZoom(0.10),
   # layers.RandomContrast(0.10),
])

# =========================================================
# PRETRAINED BACKBONE (DenseNet121)
# =========================================================

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze backbone initially
base_model.trainable = False

# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Augmentation
x = data_augmentation(inputs)

# DenseNet preprocessing
x = tf.keras.applications.densenet.preprocess_input(x)

# Backbone
x = base_model(x, training=False)

# Better than Flatten for histopathology
x = layers.GlobalAveragePooling2D()(x)

# Regularization
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Dense layer for high-level tissue patterns
#x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Binary classification
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# ======================================

4) Model 3 (3rd Model refinement)

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
   # layers.RandomZoom(0.10),
   # layers.RandomContrast(0.10),
])

# =========================================================
# PRETRAINED BACKBONE (DenseNet121)
# =========================================================

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze backbone initially
base_model.trainable = False

# =========================================================
# MODEL
# =========================================================

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Augmentation
x = data_augmentation(inputs)

# DenseNet preprocessing
x = tf.keras.applications.densenet.preprocess_input(x)

# Backbone
x = base_model(x, training=False)

# Better than Flatten for histopathology
x = layers.GlobalAveragePooling2D()(x)

# Regularization
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Dense layer for high-level tissue patterns
#x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.15)(x)

# Binary classification
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# =======================================

All  models has used:

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

and 

history = model.fit(
     X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        verbose=False
)

After, we also test the model efficency by informing the class inbalance:

class_weight = {
    0: total / (2 * n0),
    1: total / (2 * n1)
}
history = model.fit(
     X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
      #  class_weight=class_weight,
        verbose=False
)

Best model choice:

Model 3 with class inbalance informed and gray scale images.


