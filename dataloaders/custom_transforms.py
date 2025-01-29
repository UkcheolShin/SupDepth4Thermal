from __future__ import division
import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

class CustomCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, depths=None, intrinsics=None):
        for t in self.transforms:
            images, depths, intrinsics = t(images, depths, intrinsics)
        return images, depths, intrinsics

class do_nothing(object):
    def __call__(self, images, depths, intrinsics):
        return images, depths, intrinsics

class Normalize(object):
    def __init__(self, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, images, depths, intrinsics):
        for tensor in images['img_in']:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        for tensor in images['img_eh']:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, depths, intrinsics

class TensorImgEnhance(object):
    def __init__(self, modality, flags):
        self.transforms_input = list()
        self.transforms_enhance = list()

        if flags.FlagImgWiseClip: 
            self.transforms_input.append(TensorIWMM(False))

        if flags.FlagGroupWiseClip:
            self.transforms_enhance.append(TensorGWMM(False))
        if flags.FlagGroupWiseRearrange:
            self.transforms_enhance.append(TensorGWRedist(bin_num=flags.bin_num, flag_LCE=flags.FlagLCE))

    def __call__(self, images, depths=None, intrinsics=None):
        images_in = images
        images_eh = [img.clone() for img in images]

        for tf in self.transforms_input:
            images_in = tf(images_in)

        for tf in self.transforms_enhance:
            images_eh = tf(images_eh)
        
        output = {}
        output['img_in'] = images_in
        output['img_eh'] = images_eh

        return output, depths, intrinsics

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, Itype='thr'):
        if Itype == 'thr':
            self.max_value = 2**14
        else:
            self.max_value = 255

    def __call__(self, images, depths, intrinsics):
        tensor_imgs = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensor_imgs.append(torch.from_numpy(im).float()/self.max_value)

        if depths is not None:
            tensor_depths = []
            for depth in depths:
                tensor_depths.append(torch.from_numpy(depth).float())
        else:
            tensor_depths = None

        if intrinsics is not None:
            tensor_intrinsics = torch.from_numpy(intrinsics).float()
        else:
            tensor_intrinsics = None
            
        return tensor_imgs, tensor_depths, tensor_intrinsics
       
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, depths, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            if depths is not None:
                output_depths = [np.copy(np.fliplr(dep)) for dep in depths]
            else:
                output_depths = None
            w = output_images[0].shape[1]
            output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
        else:
            output_images = images
            output_depths = depths
            output_intrinsics = intrinsics
        return output_images, output_depths, output_intrinsics

class RandomVerticalFlip(object):
    """Randomly vertically filps"""

    def __call__(self, sample):
        if np.random.random() < 0.5:
            sample['left'] = np.copy(np.flipud(sample['left']))
            sample['right'] = np.copy(np.flipud(sample['right']))
            sample['disp'] = np.copy(np.flipud(sample['disp']))

            if 'pseudo_disp' in sample.keys():
                sample['pseudo_disp'] = np.copy(np.flipud(sample['pseudo_disp']))

        return sample

class RandomScaleCenterCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, depths, intrinsics):

        in_h, in_w, ch = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)

        if intrinsics is not None :
            output_intrinsics = np.copy(intrinsics)

            output_intrinsics[0] *= x_scaling
            output_intrinsics[1] *= y_scaling
            
            output_intrinsics[0, 2] -= offset_x
            output_intrinsics[1, 2] -= offset_y
        else:
            output_intrinsics = intrinsics

        if ch == 1:
            scaled_images = [np.expand_dims(cv2.resize(im, dsize=(
                scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR), axis=2) for im in images]
        else:
            scaled_images = [cv2.resize(im, dsize=(
                scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR) for im in images]

        cropped_images = [im[offset_y:offset_y + in_h,
                             offset_x:offset_x + in_w] for im in scaled_images]

        if depths is not None:
            scaled_depths = [cv2.resize(dep, dsize=(scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST) for dep in depths]
            cropped_depths = [dep[offset_y:offset_y + in_h,
                                 offset_x:offset_x + in_w] for dep in scaled_depths]
        else:
            cropped_depths = None

        return cropped_images, cropped_depths, output_intrinsics

class RescaleTo(object):
    """Rescale images to training or validation """

    def __init__(self, output_size=[256, 832], flag_resize_depth=True):
        self.output_size = output_size
        self.resize_depth = flag_resize_depth

    def __call__(self, images, depths, intrinsics):
        in_h, in_w, ch = images[0].shape
        out_h, out_w = self.output_size[0], self.output_size[1]

        if in_h != out_h or in_w != out_w:
            scaled_images = []
            scaled_depths = []
            for im in images:
                if ch == 1 :  # NIR, THR
                    scaled_images.append(np.expand_dims(cv2.resize(im, dsize=(
                        out_w, out_h), interpolation=cv2.INTER_LINEAR), axis=2))
                else:
                    scaled_images.append(cv2.resize(im, dsize=(
                        out_w, out_h), interpolation=cv2.INTER_LINEAR))

            if depths is not None:
                if self.resize_depth:
                    for depth in depths:
                        scaled_depths.append(cv2.resize(depth, dsize=(
                            out_w, out_h), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST))
                else:
                    scaled_depths = depths
            else: 
                scaled_depths = None
        else:
            scaled_images = images
            scaled_depths = depths

        if intrinsics is not None:
            output_intrinsics = np.copy(intrinsics)
            output_intrinsics[0] *= (out_w * 1.0 / in_w)
            output_intrinsics[1] *= (out_h * 1.0 / in_h)
        else:
            output_intrinsics = None

        return scaled_images, scaled_depths, output_intrinsics

class ColorAugTransform(object):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        Itype,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        assert Itype in ["rgb", "thr", "nir"]
        self.img_format = Itype
        if self.img_format == "thr":
            self.max_value = 2**14
            self.dtype = np.float32
            self.brightness_delta = brightness_delta*10
        else:
            self.max_value = 255
            self.dtype = np.float32
            self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta

    def __call__(self, imgs, depths, intrinsics):
        if self.img_format == "rgb":
            imgs = self.brightness(imgs)
            if random.randrange(2):
                imgs = self.contrast(imgs)
                imgs = self.saturation(imgs)
                imgs = self.hue(imgs)
            else:
                imgs = self.saturation(imgs)
                imgs = self.hue(imgs)
                imgs = self.contrast(imgs)
        elif (self.img_format == "thr") or (self.img_format == "nir"):
            if random.randrange(2):
                imgs = self.brightness(imgs)
                imgs = self.contrast(imgs)
            else:
                imgs = self.contrast(imgs)
                imgs = self.brightness(imgs)
        return imgs, depths, intrinsics

    def convert(self, imgs, alpha=1, beta=0):
        out_im=[]
        for im in imgs : 
            im = im.astype(np.float32) * alpha + beta
            im = np.clip(im, 0, self.max_value)#.astype(self.dtype)
            out_im.append(im)
        return out_im

    def brightness(self, imgs):
        if random.randrange(2):
            return self.convert(
                imgs, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return imgs

    def contrast(self, imgs):
        if random.randrange(2):
            return self.convert(imgs, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return imgs

    def saturation(self, imgs):
        if random.randrange(2):
            out_im = []
            alpha = random.uniform(self.saturation_low, self.saturation_high)
            for img in imgs : 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img[:, :, 1] = img[:, :, 1].astype(np.float32) * alpha
                img[:, :, 1] = np.clip(img[:, :, 1], 0, self.max_value)#.astype(self.dtype)
                out_im.append(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
            return out_im
        return imgs

    def hue(self, imgs):
        if random.randrange(2):
            out_im = []
            beta =  random.randint(-self.hue_delta, self.hue_delta)
            for img in imgs : 
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img[:, :, 0] = (img[:, :, 0].astype(int) + beta) % 180
                out_im.append(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
            return out_im
        return imgs

#Image wise min-max normalize
class TensorIWMM(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, colorize=False):
        self.colorize = colorize
        if self.colorize : 
            import matplotlib.pyplot as plt
            self.cmap = plt.get_cmap('jet')

    def __call__(self, images):
        imgs = []
        for im in images : 
            im_srt = np.sort(im.squeeze().numpy().reshape(-1))
            tmax = im_srt[round(len(im_srt)*0.99)-1]
            tmin = im_srt[round(len(im_srt)*0.01)]

            # handle numpy array
            img = im.clone()
            img[img<tmin] = torch.tensor(tmin)
            img[img>tmax] = torch.tensor(tmax)
            img_out = (img.float() - tmin)/(tmax - tmin)
            if self.colorize:
                img_out = self.cmap(img_out.squeeze())[:,:,0:3]
                img_out = torch.from_numpy(img_out.transpose(2,0,1)).float()
            imgs.append(img_out) #CHW

        return imgs

# Image Group wise min-max normalize
"""
Implementation based on:
    Ukcheol Shin et al, Maximizing Self-Supervision From Thermal Image 
    for Effective Self-Supervised Learning of Depth and Ego-Motion, RA-L 2022
    https://github.com/UkcheolShin/ThermalMonoDepth
"""
class TensorGWMM(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, colorize=False):
        self.colorize = colorize
        if self.colorize : 
            import matplotlib.pyplot as plt
            self.cmap = plt.get_cmap('jet')

    def __call__(self, images):
        imgs = []
        tmin = 0
        tmax = 0
        for im in images : 
            im = im.mean(axis=0).numpy() #HW
            im_srt = np.sort(im.reshape(-1))
            tmax += im_srt[round(len(im_srt)*0.99)-1]
            tmin += im_srt[round(len(im_srt)*0.01)]
        tmax /= len(images)
        tmin /= len(images)

        for im in images: #CHW
            # handle numpy array
            img = im.clone().mean(axis=0, keepdim=True)
            # img = ft.gaussian_blur2d(img.unsqueeze(0), (3, 3), (0.5, 0.5)).squeeze(0)
            img[img<tmin] = tmin
            img[img>tmax] = tmax
            img_out = (img.float() - tmin)/(tmax - tmin)
            if self.colorize:
                img_out = self.cmap(img_out.squeeze())[:,:,0:3]
                img_out = torch.from_numpy(img_out.transpose(2,0,1)).float()
            imgs.append(img_out) #CHW

        return imgs

# Image Group wise histogram rearrangement
"""
Implementation based on:
    Ukcheol Shin et al, Maximizing Self-Supervision From Thermal Image 
    for Effective Self-Supervised Learning of Depth and Ego-Motion, RA-L 2022
    https://github.com/UkcheolShin/ThermalMonoDepth
"""
class TensorGWRedist(object):
    def __init__(self, bin_num = 30, AH_param1 = 3, AH_param2 = 8, flag_LCE=False):
        self.bins = bin_num
        self.CLAHE = cv2.createCLAHE(clipLimit=AH_param1, tileGridSize=(AH_param2,AH_param2))
        self.flag_LCE = flag_LCE

    def __call__(self, images):
        imgs = []
        tmp_img = torch.cat(images, axis=0)
        hist = torch.histc(tmp_img, bins=self.bins)
        imgs_max = tmp_img.max()
        imgs_min = tmp_img.min()
        itv = (imgs_max - imgs_min)/self.bins
        total_num = hist.sum() 

        for im in images : #CHW
            _,H,W = im.shape
            mul_mask_ = torch.zeros((self.bins,H,W))
            sub_mask_ = torch.zeros((self.bins,H,W))
            subhist_new_min = imgs_min.clone()

            for x in range(0,self.bins) : 
                subhist = (im > imgs_min+itv*x) & (im <= imgs_min+itv*(x+1))
                if (subhist.sum() == 0):
                    continue
                subhist_new_itv = hist[x]/total_num        
                mul_mask_[x,...] = subhist * (subhist_new_itv / itv) 
                sub_mask_[x,...] = subhist * (subhist_new_itv / itv * -(imgs_min+itv*x) + subhist_new_min) 
                subhist_new_min += subhist_new_itv

            mul_mask = mul_mask_.sum(axis=0, keepdim=True).detach()
            sub_mask = sub_mask_.sum(axis=0, keepdim=True).detach()
            im_ = mul_mask*im + sub_mask
            if self.flag_LCE : 
                im_ = self.CLAHE.apply((im_.squeeze()*255).numpy().astype(np.uint8)).astype(np.float32)
                im_ = np.expand_dims(im_, axis=2)
                img_out = torch.from_numpy(np.transpose(im_/255., (2, 0, 1)))
                # img_out = ft.gaussian_blur2d(img_out.unsqueeze(0), (3, 3), (0.5, 0.5)).squeeze(0)
            else:
                img_out = im_
            imgs.append(img_out) #CHW

        return imgs
