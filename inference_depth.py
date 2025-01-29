import numpy as np
from tqdm import tqdm
import torch

import os 
import os.path as osp
from argparse import ArgumentParser
from mmcv import Config
from models import MODELS
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.visualization import *
import cv2

from path import Path
from imageio import imread, imwrite
from dataloaders.utils import load_as_float_img
import dataloaders.custom_transforms as cs_tf

def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--input_dir' , type=str, default='./demo_images/MS2_sample/')
    parser.add_argument('--save_dir' , type=str, default=' ')
    parser.add_argument('--modality' , type=str, default='thr')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    return parser.parse_args()

@torch.no_grad()
def main():
    # parse args
    args = parse_args()

    # parse cfg
    cfg = Config.fromfile(osp.join(args.config))

    # show information
    print(f'Now training with {args.config}...')

    # prepare data loader
    dataset_name = cfg.dataset['list'][0]
    cfg.dataset[dataset_name].test.modality = args.modality

    input_dir = Path(args.input_dir)
    image_files = sum([(input_dir+'/left/').files('*_{}.{}'.format(args.modality,ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)
    print('{} samples found for evaluation'.format(len(image_files)))

    # define model
    model = MODELS.build(name=cfg.model.name, option=cfg)

    if args.ckpt_path != None:
        print('load pre-trained model from {}'.format(args.ckpt_path))
        weight = torch.load(args.ckpt_path)
        model.load_state_dict(weight["state_dict"])
        # model = model.load_from_checkpoint(args.ckpt_path, strict=True)
    model.cuda()
    model.eval()

    # normaliazation
    img_size = cfg.dataset[dataset_name][args.modality]['test_size']
    flags = cfg.dataset['Augmentation'][args.modality]

    inference_transform = cs_tf.CustomCompose([
        cs_tf.RescaleTo(img_size),
        cs_tf.ArrayToTensor(Itype=args.modality),
        cs_tf.TensorImgEnhance(args.modality, flags),
        cs_tf.Normalize()]
    )

    if args.save_dir != ' ':
        save_dir_all   = osp.join(args.save_dir, 'all')
        os.makedirs(save_dir_all, exist_ok=True)
        
    # model inference
    for i, img_file in enumerate(tqdm(image_files)):
        filename = os.path.splitext(os.path.basename(img_file))[0]

        tgt_img_left  = load_as_float_img(img_file) # HWC
        tensor_img = inference_transform([tgt_img_left], None, None)

        tgt_left_in = tensor_img[0]['img_in'][0].cuda()
        pred_depth = model.inference_depth(tgt_left_in.unsqueeze(0).cuda())

        # save prediction
        if  args.save_dir != ' ':
            img_vis = visualize_image(tgt_left_in, flag_np=True).transpose(1,2,0)
            pred_depth_vis = visualize_depth_as_numpy(pred_depth.squeeze(), 'jet')

            png_path = osp.join(save_dir_all, "{}.png".format(filename))
            stack = cv2.cvtColor(np.concatenate((img_vis, pred_depth_vis), axis=0), cv2.COLOR_RGB2BGR)
            cv2.imwrite(png_path, stack)


if __name__ == '__main__':
    main()
