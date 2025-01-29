import torchvision.transforms as T
import torch
import numpy as np
import cv2
from PIL import Image

import matplotlib as mpl
import matplotlib.cm as cm

def visualize_image(image, flag_np=False):
    """
    tensor image: (3, H, W)
    """
    C,H,W = image.shape
    if C == 1:
        image = image.repeat(3,1,1)  # (3, H, W)

    x = (image.cpu() * 0.225 + 0.45)

    if flag_np:
        return np.array(x*255.).astype(np.uint8)
    else:
        return x


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    # mi = np.percentile(x[x!=0], 1)
    # ma = np.percentile(x[x!=0], 98)
    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_

def visualize_disp_as_numpy(disp, cmap='jet'):
    """
    Args:
        data (HxW): disp data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): disp visualization (RGB)
    """

    disp = disp.cpu().numpy()
    disp = np.nan_to_num(disp)  # change nan to 0

    vmin = np.percentile(disp[disp!=0], 0)
    vmax = np.percentile(disp[disp!=0], 95)

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

def visualize_depth_as_numpy(depth, cmap='jet', is_sparse=False):
    """
    Args:
        data (HxW): depth data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0

    inv_depth = 1 / (x + 1e-6)

    if is_sparse:
        vmax = 1/np.percentile(x[x!=0], 5)
    else:
        vmax = np.percentile(inv_depth, 95)

    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    if is_sparse:
        vis_data[inv_depth>vmax] = 0
    return vis_data
