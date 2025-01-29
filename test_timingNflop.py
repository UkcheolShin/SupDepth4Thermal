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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--test_env' , type=str, default='test_day') # test_night, test_rain
    parser.add_argument('--modality' , type=str, default='thr')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--is_stereo', action='store_true')

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
        model = model.load_from_checkpoint(args.ckpt_path,strict=False)
    model.cuda()
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))

    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    # model inference
    if args.is_stereo:
        for i, batch in enumerate(tqdm(test_loader)):
            tgt_left_in = batch["tgt_left"].cuda()
            tgt_right_in = batch["tgt_right"].cuda()

            # Warm up
            for _ in range(10):
                pred_disp = model.inference_disp(tgt_left_in, tgt_right_in)

            # Measure performance
            with torch.no_grad():
                for rep in range(repetitions):
                    starter.record()
                    _ = model.inference_disp(tgt_left_in, tgt_right_in)
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
                flops = FlopCountAnalysis(model, (tgt_left_in, tgt_right_in))
                print(flop_count_table(flops))
                break
    else:
        for i, batch in enumerate(tqdm(test_loader)):
            tgt_img     = batch['tgt_image'].cuda()

            # Warm up
            for _ in range(10):
                pred_depth = model.inference_depth(tgt_img)

            # Measure performance
            with torch.no_grad():
                for rep in range(repetitions):
                    starter.record()
                    _ = model.inference_depth(tgt_img)
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[rep] = curr_time
                flops = FlopCountAnalysis(model, (tgt_img))
                print(flop_count_table(flops))
                break
    mean_inference_time = np.mean(timings, axis=0)
    print(("mean inference time: {} " ).format(mean_inference_time))

if __name__ == '__main__':
    main()
