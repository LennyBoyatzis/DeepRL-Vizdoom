from skimage import transform
import numpy as np

resolution = (30, 45)

def preprocess(img):
    img = transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img
