import numpy as np
from tqdm import tqdm
import torch

import os 
import os.path as osp
from argparse import ArgumentParser
from mmcv import Config
from models import MODELS
from dataloaders import build_dataset
from torch.utils.data import DataLoader

from models.metrics.eval_metric import compute_depth_errors
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.visualization import *
import cv2

def parse_args():
    parser = ArgumentParser()   

    # configure file
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--test_env' , type=str, default='test_day') # test_night, test_rain
    parser.add_argument('--save_dir' , type=str, default=' ')
    parser.add_argument('--modality' , type=str, required=True)
    parser.add_argument('--seed', type=int, default=1024)
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

    # configure seed
    seed_everything(args.seed)

    # prepare data loader
    dataset_name = cfg.dataset['list'][0]
    cfg.dataset[dataset_name].test_env = args.test_env
    cfg.dataset[dataset_name].test.modality = args.modality
    dataset = build_dataset(cfg.dataset, eval_mode='depth', split='test')

    test_loader     = DataLoader(dataset['test']['depth'], 
                                batch_size=1,
                                shuffle=False, 
                                num_workers=cfg.workers_per_gpu, 
                                drop_last=False)

    print('{} samples found for evaluation'.format(len(test_loader)))

    # define model
    model = MODELS.build(name=cfg.model.name, option=cfg)

    if args.ckpt_path != None:
        print('load pre-trained model from {}'.format(args.ckpt_path))
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'],strict=False)
        # model = model.load_from_checkpoint(args.ckpt_path,strict=False)
    model.cuda()
    model.eval()

    if args.save_dir != ' ':
        save_dir_all   = osp.join(args.save_dir, 'all')
        os.makedirs(save_dir_all, exist_ok=True)

    # model inference
    all_errs = []
    for i, batch in enumerate(tqdm(test_loader)):
        # if 'MSCRF' in args.ckpt_path :
        #     tgt_img     = batch["tgt_left"]
        #     tgt_eh_img  = batch['tgt_left_eh']
        tgt_img     = batch['tgt_image']
        tgt_eh_img  = batch['tgt_image_eh']
        gt_depth    = batch['tgt_depth_gt']

        if 'Midas' in args.ckpt_path or 'DPT' in args.ckpt_path :
            # need to estimate scale and shift factor
            pred_depth = model.inference_depth(tgt_img.cuda(), gt_depth.cuda())
        else:
            pred_depth = model.inference_depth(tgt_img.cuda())
        
        # reshape dimension for evaluation and interpolation, (b,h,w)
        if len(pred_depth.shape) == 4:
            pred_depth = pred_depth.squeeze(1) # BHW
        elif len(pred_depth.shape) == 2:
            pred_depth = pred_depth.unsqueeze(1) # BHW

        # resizing prediction
        batch_size, h, w = gt_depth.size()
        if pred_depth.nelement() != gt_depth.nelement():
            pred_depth = torch.nn.functional.interpolate(pred_depth.unsqueeze(1), [h, w], mode='bilinear').squeeze(1)

        if 'Midas' in args.ckpt_path or 'DPT' in args.ckpt_path :
            errs = compute_depth_errors(gt_depth.cuda(), pred_depth, align=False)
        else:
            errs = compute_depth_errors(gt_depth.cuda(), pred_depth)

        all_errs.append(np.array(errs))
 
        if args.save_dir != ' ':
            if i%10 == 0 :
                c_, h, w = tgt_img[0].size()
                if tgt_img.nelement() != gt_depth.nelement():
                    pred_depth = torch.nn.functional.interpolate(pred_depth.unsqueeze(1), [h, w], mode='bilinear').squeeze(1)
                    gt_depth = torch.nn.functional.interpolate(gt_depth.unsqueeze(1), [h, w], mode='nearest').squeeze(1)

                img_vis = visualize_image(tgt_img[0], flag_np=True).transpose(1,2,0)
                pred_depth_ = visualize_depth_as_numpy(pred_depth.squeeze(), 'jet')
                gt_depth = visualize_depth_as_numpy(gt_depth.squeeze(), 'jet', is_sparse=True)

                png_path = osp.join(save_dir_all, "{:05}.png".format(i))
                stack = cv2.cvtColor(np.concatenate((img_vis, gt_depth, pred_depth_), axis=0), cv2.COLOR_RGB2BGR)
                cv2.imwrite(png_path, stack)
                
    all_errs = np.stack(all_errs)
    mean_errs = np.mean(all_errs, axis=0)
    print('test set: {}, len: {}'.format(args.test_env, len(test_loader)))
    print("\n  " + ("{:>8} | " * 9).format("abs_diff", "abs_rel",
          "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.7f}  " * 9).format(*mean_errs.tolist()) + "\\\\")

if __name__ == '__main__':
    main()
