import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from PIL import Image
from scipy.sparse import csr_matrix
import numpy as np
import tensorflow as tf
import pickle as pkl
from skimage import measure
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from PIL import Image
import numpy as np

def load_crop_resize_image(image_path, s, mode="RGB"):
    """
    Load an image with PIL, center-crop it to a square using
    min(width, height), resize to s x s, and select channels.

    Parameters
    ----------
    image_path : str
        Path to the image.
    
    s : int
        Output image size (s x s).
    
    mode : str
        One of:
        - "R"   : red channel
        - "G"   : green channel
        - "B"   : blue channel
        - "RGB" : full RGB image
        - "L"   : grayscale

    Returns
    -------
    np.ndarray
        Processed image as numpy array.
    """

    # Open image
    img = Image.open(image_path).convert("RGB")

    width, height = img.size

    # Square crop size
    crop_size = min(width, height)

    # Center crop coordinates
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Crop
    img = img.crop((left, top, right, bottom))

    # Resize
    img = img.resize((s, s), Image.Resampling.LANCZOS)

    # Channel selection
    mode = mode.upper()

    if mode == "RGB":
        return np.array(img)

    elif mode in ["R", "G", "B"]:
        channel_index = {"R": 0, "G": 1, "B": 2}[mode]
        return np.array(img)[:, :, channel_index]

    elif mode == "L":
        return np.array(img.convert("L"))

    else:
        raise ValueError("mode must be one of: 'R', 'G', 'B', 'RGB', 'L'")

import numpy as np
from skimage import measure


def contour_mask(img_np, threshold=None):
    """
    Returns a binary contour mask.

    Parameters
    ----------
    img_np : np.ndarray
        Can be:
        - RGB image: (H, W, 3)
        - Single channel image: (H, W)

    threshold : float or None

    Returns
    -------
    np.ndarray of bool
    """

    # RGB image
    if img_np.ndim == 3:
        gray = (
            0.299 * img_np[:, :, 0] +
            0.587 * img_np[:, :, 1] +
            0.114 * img_np[:, :, 2]
        )

    # Already grayscale / single-channel
    elif img_np.ndim == 2:
        gray = img_np.astype(np.float32)

    else:
        raise ValueError("Unsupported image shape")

    # Threshold
    if threshold is None:
        threshold = gray.mean()

    binary = gray < threshold

    # Find contours
    contours = measure.find_contours(binary, level=0.5)

    # Output mask
    mask = np.zeros(binary.shape, dtype=bool)

    # Draw contours
    for contour in contours:
        coords = np.round(contour).astype(int)

        for y, x in coords:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask[y, x] = True

    return mask

def read_img(img_path):
    pil = Image.open(img_path).convert("RGB")
    M = np.array(pil)#.astype(np.uint8)
    return M
    
def img2vec(img_path,conversion="L"):
    mapping={"R":0,"G":1,"B":2}
    if conversion=="L":
        gray_pil = Image.open(img_path).convert(conversion)
        gray = np.array(gray_pil)#.astype(np.uint8)
        return gray
    else:
        pil = Image.open(img_path).convert("RGB")
        M = np.array(pil)#.astype(np.uint8)
        return M[:,:,mapping[conversion.upper()]]
        
    

def fft_img2stats(img_path,mag=5,scale=1):
    img = Image.open(img_path).convert("L")
    img=img.resize(((int(scale*img.width)),int(scale*img.height)))
   
    img_array = np.array(img).astype(np.uint8)

    # Fourier Transform
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)

    # Magnitude spectrum
    gray =mag * np.log(np.abs(fshift) + 1).astype(np.uint8)
    m=gray.mean()
    s=gray.std()
    minv=gray.min()
    maxv=gray.max()
    median=np.median(gray)
    
    return np.asarray([m,median,s,minv,maxv])

def fft_img2vec(img_path,mag=5,scale=1,conversion="L"):
    mapping={"R":0,"G":1,"B":2}
    if conversion=="L":
        img = Image.open(img_path).convert(conversion)
        img=img.resize(((int(scale*img.width)),int(scale*img.height)))
    else:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = img[:,:,mapping[conversion.upper()]]
        
   
    img_array = np.array(img).astype(np.uint8)

    # Fourier Transform
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)

    # Magnitude spectrum
    magnitude =mag * np.log(np.abs(fshift) + 1).astype(np.uint8)
    
    return magnitude

import numpy as np


from scipy.stats import skew, kurtosis, entropy
D={0:"mean",1:"std",2:"range_val",3:"sparsity",4:"skew",5:"kurt",6:"ent",
   7:"avg_grad",8:"std_grad",9:"cov_feat",10:"fft_mean",11:"fft_std",12:"fft_range_val",13:"fft_sparsity",
   14:"fft_skew",15:"fft_kurt",16:"fft_ent",17:"fft_avg_grad",18:"fft_std_grad",19:"fft_cov_feat",20:"target"}

D={0:"mean",1:"std",2:"range_val",3:"skew",4:"kurt",5:"ent",
   6:"avg_grad",7:"std_grad",8:"cov_feat",9:"fft_mean",10:"fft_std",11:"fft_range_val",
   12:"fft_skew",13:"fft_kurt",14:"fft_ent",15:"fft_avg_grad",16:"fft_std_grad",17:"fft_cov_feat",18:"target"}
def stats2vec(X):
   # X=img2vec(path)
    X = np.asarray(X)
    flat = X.ravel()

    # --- Basic stats ---
    mean = np.mean(flat)
    std = np.std(flat)

    # --- Range ---
    range_val = np.ptp(flat)

    # --- Sparsity ---
   # sparsity = np.mean(flat == 0)

    # --- Shape stats ---
    skewness = skew(flat)
    kurt = kurtosis(flat)

    # --- Entropy ---
    hist, _ = np.histogram(flat, bins=50, density=True)
    ent = entropy(hist + 1e-12)

    # --- Gradient (for spatial structure) ---
    if X.ndim >= 2:
        gx, gy = np.gradient(X)
        grad_mag = np.sqrt(gx**2 + gy**2)
        avg_gradient = np.mean(grad_mag)
        std_gradient=np.std(grad_mag)
    else:
        avg_gradient = 0.0  # not meaningful for 1D
        std_gradient=0.0

    # --- Covariance (reduced to scalar) ---
    if X.ndim >= 2:
        cov_matrix = np.cov(X, rowvar=False)
        cov_feature = np.mean(np.abs(cov_matrix))  # robust scalar summary
    else:
        cov_feature = np.var(flat)

    # --- Final feature vector ---
    features = np.array([
        mean,
        std,
        range_val,
       # sparsity,
        skewness,
        kurt,
        ent,
        avg_gradient,
        std_gradient,
        cov_feature
    ])

    return features
def full_stats2vec(path,conversion="L"):
   
    X=img2vec(path,conversion=conversion)
   #print(X.shape)
    V1=stats2vec(X)
    X=fft_img2vec(path,conversion=conversion)
    V2=stats2vec(X)
    return np.hstack([V1,V2])

