import numpy as np
from imageio import imread

def load_as_float_img(path):
    img =  imread(path).astype(np.float32)
    if len(img.shape) == 2: # for NIR and thermal images
        img = np.expand_dims(img, axis=2)
    return img

def load_as_float_depth(path):
    if 'png' in path:
        depth =  np.array(imread(path).astype(np.float32))
    elif 'npy' in path:
        depth =  np.load(path).astype(np.float32)
    return depth

# Thermal image transformation
# The parameters (R,B,F,O) vary depending on the thermal camera.
# Refer to your thermal camera's parameter for accurate temperature conversion.
def Celsius2Raw(celcius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celcius_degree + 273.15)) - F) + O;
    return raw_value

def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15;
    return Celsius
