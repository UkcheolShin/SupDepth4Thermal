# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
import torch.utils.data as data
import numpy as np
import os

import math
import random
from path import Path
from dataloaders.utils import load_as_float_img, load_as_float_depth

class DataLoader_ViViD(data.Dataset):
    """A data loader where the files are arranged in this way:
        * Structure of "KAIST ViViD dataset"
        |-- scene_1
            |-- RGB
            |-- Depth_RGB
            |-- Thermal
            |-- Depth_T
            |-- cam_RGB.txt
            |-- cam_T.txt
            |-- poses_RGB.txt
            |-- poses_T.txt
            |-- Tr_T2RGB.txt
        |-- scene_2
        |-- scene_3
        ...
        |-- train_indoor.txt
        |-- train_outdoor.txt
        |-- val_indoor.txt
        |-- val_outdoor.txt
        |-- test_indoor_dark.txt
        |-- test_indoor_light.txt
        |-- test_outdoor.txt
    """
    def __init__(self, root, seed=None, data_split='train', modality='thr', tf_dict={}, \
                 data_format='MonoMultiDepth', sampling_step=3, set_length=3, set_interval=1, opt=None):
        super(DataLoader_ViViD, self).__init__()

        np.random.seed(seed)
        random.seed(seed)

        # read (train/val/test) data list
        self.name = opt.name
        self.root = Path(root)
        if 'train' in data_split:
            data_list_file = [os.path.join(self.root,'train_indoor.txt'),\
                              os.path.join(self.root,'train_outdoor.txt')]
        elif 'val' in data_split:
            data_list_file = [os.path.join(self.root,'val_indoor.txt'),\
                              os.path.join(self.root,'val_outdoor.txt')]
        elif 'test' in data_split:
            data_list_file = [os.path.join(self.root,'test_indoor.txt'),\
                              os.path.join(self.root,'test_outdoor.txt')]
        else: # if data_split is a specific sequence
            data_list_file = self.root+'/'+data_split

        if isinstance(data_list_file, list):
            if 'indoor' in data_split:
                data_list_file.pop(1)
            elif 'outdoor' in data_split:
                data_list_file.pop(0)

        self.folders = []
        if isinstance(data_list_file, list):
            for file in data_list_file:
                self.folders += [self.root/folder[:-1] for folder in open(file)]
        else:
            self.folders.append(data_list_file)

        # determine which data getter function to use 
        if data_format == 'MonoDepth': # Monocular depth estimation, dict: {'img', 'depth'}
            self.data_getter = self.get_data_MonoDepth
            self.crawl_folders_depth(sampling_step, set_length, set_interval)
        elif data_format == 'MonoMultiViewDepth':
            self.data_getter = self.get_data_MonoMultiDepth
            self.crawl_folders_depth(sampling_step, set_length, set_interval)
        elif data_format == 'MultiPose':
            self.data_getter = self.get_data_MultiPose
            self.crawl_folders_pose(sampling_step, set_length, set_interval)
        else:
            raise NotImplementedError(f"not supported type {data_format} in MS2 dataset.")

        self.modality = modality
        self.set_augmentations(data_split, tf_dict)

    def __getitem__(self, index):
        if isinstance(self.modality, list):
            results = {}
            results['rgb'] = self.data_getter(index, 'rgb')
            results['thr'] = self.data_getter(index, 'thr')
            results['extrinsics'] = self.get_extrinsic(index)
            return results
        else:
            return self.data_getter(index, self.modality)

    def __len__(self):
        return len(self.samples)

    def set_augmentations(self, data_split, tf_dict):
        self.tf = {}
        if data_split == 'train':
            self.tf['rgb'] = tf_dict['rgb']['train']
            self.tf['thr'] = tf_dict['thr']['train']
        else:
            self.tf['rgb'] = tf_dict['rgb']['eval']
            self.tf['thr'] = tf_dict['thr']['eval']

    def crawl_folders_depth(self, sampling_step, set_length, set_interval):
        sequence_set = []
        demi_length = (set_length-1)//2 + set_interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)
        
        # iterate over the different sensor modalities
        for folder in self.folders:
            imgs_thr           = sorted((folder/"Thermal").files('*.png')) # "RGB" "Depth" "Thermal"
            imgs_rgb           = sorted((folder/"RGB").files('*.png')) 

            # Subsampling the list of images according to the sampling step
            imgs_thr   = imgs_thr[0:-1:sampling_step]
            imgs_rgb   = imgs_rgb[0:-1:sampling_step]

            intrinsics_thr     = np.genfromtxt(folder/'cam_T.txt').astype(np.float32).reshape((3, 3))
            intrinsics_rgb     = np.genfromtxt(folder/'cam_RGB.txt').astype(np.float32).reshape((3, 3))

            if "pp" in self.root :     
                extrinsics_rgb2thr = np.genfromtxt(folder/'Tr_RGB2T.txt').astype(np.float32).reshape((4, 4)) 
                extrinsics_thr2rgb = np.linalg.inv(extrinsics_rgb2thr)
            else:
                extrinsics_thr2rgb = np.genfromtxt(folder/'Tr_T2RGB.txt').astype(np.float32).reshape((4, 4)) 
                extrinsics_rgb2thr = np.linalg.inv(extrinsics_thr2rgb)

            if "pp" in self.root :     
                for i in range(demi_length, len(imgs_thr)-demi_length):
                    sample = {'intrinsics_thr': intrinsics_thr, 'tgt_thr': imgs_thr[i], 'ref_thr_imgs': [],
                              'intrinsics_rgb': intrinsics_rgb, 'tgt_rgb': imgs_rgb[i], 'ref_rgb_imgs': [],
                              'extrinsics_thr2rgb' : extrinsics_thr2rgb}

                    for j in shifts:
                        sample['ref_thr_imgs'].append(imgs_thr[i+j])
                        sample['ref_rgb_imgs'].append(imgs_rgb[i+j])
                    sequence_set.append(sample)
            else:
                for i in range(demi_length, len(imgs_thr)-demi_length):
                    depth_thr          = folder/"Depth_T"/(imgs_thr[i].name[:-4] + '.npy')
                    depth_rgb          = folder/"Depth_RGB"/(imgs_rgb[i].name[:-4] + '.npy')
                    sample = {'intrinsics_thr': intrinsics_thr, 'tgt_thr': imgs_thr[i], 'ref_thr_imgs': [],
                              'intrinsics_rgb': intrinsics_rgb, 'tgt_rgb': imgs_rgb[i], 'ref_rgb_imgs': [],
                              'extrinsics_rgb2thr' : extrinsics_rgb2thr, 
                              'extrinsics_thr2rgb' : extrinsics_thr2rgb, 
                              'tgt_depth_thr' : depth_thr, 'tgt_depth_rgb' : depth_rgb,
                              'ref_depth_thrs' : [], 'ref_depth_rgbs' : [] }

                    for j in shifts:
                        depth_thr          = folder/"Depth_T"/(imgs_thr[i+j].name[:-4] + '.npy')
                        depth_rgb          = folder/"Depth_RGB"/(imgs_rgb[i+j].name[:-4] + '.npy')
                        sample['ref_thr_imgs'].append(imgs_thr[i+j])
                        sample['ref_rgb_imgs'].append(imgs_rgb[i+j])
                        sample['ref_depth_thrs'].append(depth_thr)
                        sample['ref_depth_rgbs'].append(depth_rgb)

                    sequence_set.append(sample)

        self.samples = sequence_set

    def crawl_folders_pose(self, sampling_step, set_length, set_interval):
        sequence_set = []
        demi_length = (set_length - 1) // 2 
        shift_range = np.array([set_interval*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
        for folder in self.folders:      
            imgs_thr = sorted((folder/"Thermal").files('*.png')) 
            poses_thr  = np.genfromtxt(folder/'poses_T.txt').astype(np.float64).reshape(-1, 3, 4)

            imgs_rgb = sorted((folder/"RGB").files('*.png')) 
            poses_rgb  = np.genfromtxt(folder/'poses_RGB.txt').astype(np.float64).reshape(-1, 3, 4)

            # construct 5-snippet sequences
            tgt_indices = np.arange(demi_length, len(imgs_rgb) - demi_length).reshape(-1, 1)
            snippet_indices = shift_range + tgt_indices
            for indices in snippet_indices :
                sample = {'imgs_rgb' : [], 'poses_rgb' : [],\
                          'imgs_thr' : [], 'poses_thr' : []}
                for i in indices :
                    sample['imgs_rgb'].append(imgs_rgb[i])
                    sample['poses_rgb'].append(poses_rgb[i])
                    sample['imgs_thr'].append(imgs_thr[i])
                    sample['poses_thr'].append(poses_thr[i])
                sequence_set.append(sample)

        self.samples = sequence_set

    def get_extrinsic(self, index):
        sample = self.samples[index]

        result = {}
        result["RGB2THR"] = torch.as_tensor(sample['extrinsics_rgb2thr'][0:3,:].astype(np.float32))
        result["THR2RGB"] = torch.as_tensor(sample['extrinsics_thr2rgb'][0:3,:].astype(np.float32))
        return result

    # For monocular depth estimation
    def get_data_MonoDepth(self, index, modality):
        sample = self.samples[index]

        tgt_img = load_as_float_img(sample['tgt_img_left'])
        tgt_depth_gt = np.array(load_as_float_depth(sample['tgt_depth_gt']))/256.0

        if modality == 'rgb':
            tgt_img      = load_as_float_img(sample['tgt_rgb'])
            tgt_depth_gt = load_as_float_depth(sample['tgt_depth_rgb'])
            intrinsics   = np.copy(sample['intrinsics_rgb'])
        elif modality == 'thr':
            tgt_img      = load_as_float_img(sample['tgt_thr'])
            tgt_depth_gt = load_as_float_depth(sample['tgt_depth_thr'])
            intrinsics   = np.copy(sample['intrinsics_thr'])

        imgs, depths, intrinsics, _ = self.tf[modality]([tgt_img], [tgt_depth_gt], \
                                                           intrinsics)
        result = {}
        result["tgt_image"]     = imgs['img_in'][0]
        result["tgt_image"]     = imgs['img_in'][0]
        result["tgt_depth_gt"]  = depths[0]
        result["intrinsics"]    = intrinsics.squeeze()
        return result

    # For multi-view depth estimation
    def get_data_MonoMultiDepth(self, index, modality):
        sample = self.samples[index]

        if modality == 'rgb':
            tgt_img  = load_as_float_img(sample['tgt_rgb'])
            ref_imgs = [load_as_float_img(ref_img) for ref_img in sample['ref_rgb_imgs']]
            intrinsics = np.copy(sample['intrinsics_rgb'])

            tgt_depth_gt  = np.array(load_as_float_depth(sample['tgt_depth_rgb']))
            ref_depths_gt = [np.array(load_as_float_depth(ref_depth)) for ref_depth in sample['ref_depth_rgbs']]
        elif modality == 'thr':
            tgt_img    = load_as_float_img(sample['tgt_thr'])
            ref_imgs   = [load_as_float_img(ref_img) for ref_img in sample['ref_thr_imgs']]
            intrinsics = np.copy(sample['intrinsics_thr'])

            tgt_depth_gt  = np.array(load_as_float_depth(sample['tgt_depth_thr']))
            ref_depths_gt = [np.array(load_as_float_depth(ref_depth)) for ref_depth in sample['ref_depth_thrs']]

        imgs, depths, intrinsics = self.tf[modality]([tgt_img] + ref_imgs, \
                                                        [tgt_depth_gt] + ref_depths_gt, \
                                                        intrinsics)
        result = {}
        result["tgt_image"]     = imgs['img_in'][0]
        result["ref_images"]    = imgs['img_in'][1:]
        result["tgt_image_eh"]  = imgs['img_eh'][0]
        result["ref_images_eh"] = imgs['img_eh'][1:]
        result["tgt_depth_gt"]  = depths[0]
        result["ref_depth_gts"] = depths[1:]
        result["intrinsics"]    = intrinsics.squeeze()
        return result

    # For multi-view pose estimation
    def get_data_MultiPose(self, index, modality):
        sample = self.samples[index]

        if modality == 'rgb':
            imgs = [load_as_float_img(img) for img in sample['imgs_rgb']]
            poses = np.stack([pose for pose in sample['poses_rgb']])
        elif modality == 'thr' :
            imgs = [load_as_float_img(img) for img in sample['imgs_thr']]
            poses = np.stack([pose for pose in sample['poses_thr']])

        imgs, _, _ = self.tf[modality](imgs, None, None)

        first_pose = poses[0]
        poses[:,:,-1] -= first_pose[:,-1]
        compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

        result = {}
        result["images"]    = imgs['img_in']
        result["poses"]     = torch.as_tensor(compensated_poses)

        return result
