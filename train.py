import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from models import MODELS
from dataloaders import build_dataset
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

def parse_args():
    parser = ArgumentParser()

    # configure file
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--out_dir' , type=str, default='checkpoints')
    parser.add_argument('--exp_name', type=str, default='test_', help='experiment name')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    return parser.parse_args()

if __name__ == '__main__':

    # parse args
    args = parse_args()

    # parse cfg
    cfg = Config.fromfile(osp.join(args.config))

    # show information
    print(f'Now training with {args.config}...')

    # configure seed
    seed_everything(args.seed)

    # prepare data loader & ckpt_callback
    dataset = build_dataset(cfg.dataset, cfg.model.eval_mode, split='train_val')

    train_loader = DataLoader(dataset['train'], 
                              batch_size=cfg.imgs_per_gpu, 
                              shuffle=True, 
                              num_workers=cfg.workers_per_gpu, 
                              pin_memory=True,
                              drop_last=True)

    # define ckpt_callback
    val_loaders = []
    checkpoint_callbacks = []
    work_dir = osp.join(args.out_dir, args.exp_name)

    if 'depth' in cfg.model.eval_mode: 
      val_loader_  = DataLoader(dataset['val']['depth'], 
                                batch_size=cfg.imgs_per_gpu,
                                shuffle=False, 
                                num_workers=cfg.workers_per_gpu, 
                                pin_memory=True,
                                drop_last=True)

      callback_   = ModelCheckpoint(dirpath=work_dir,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_top_k=1,
                                    filename='ckpt_{epoch:02d}_{step}')
                                    # every_n_epochs=cfg.checkpoint_epoch_interval)

      val_loaders.append(val_loader_)             
      checkpoint_callbacks.append(callback_)                        

    print('{} samples found for training'.format(len(train_loader)))
    for idx, val_loader in enumerate(val_loaders):
      print('{} samples found for validatioin set {}'.format(len(val_loader), idx))

    # build model
    model = MODELS.build(name=cfg.model.name, option=cfg)

    # load checkpoint if exist
    if args.ckpt_path is not None:
        print('load pre-trained model from {}'.format(args.ckpt_path))
        # model = model.load_from_checkpoint(args.ckpt_path, option=cfg)
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'],strict=False)

    # training
    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=False) if args.num_gpus > 1 else None,
                      accelerator="gpu", 
                      devices=args.num_gpus,
                      default_root_dir=work_dir,
                      num_nodes=1,
                      num_sanity_val_steps=5,
                      max_epochs=cfg.total_epochs,
                      check_val_every_n_epoch=1,
                      limit_train_batches=cfg.batch_lim_per_epoch,                      
                      callbacks=checkpoint_callbacks,
                      benchmark=True,
                      precision=32)
    trainer.fit(model, train_loader, val_dataloaders=val_loaders)
