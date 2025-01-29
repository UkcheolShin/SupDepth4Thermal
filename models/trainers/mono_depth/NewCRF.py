# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/aliyun/NeWCRFs
import torch

from models.network import NewCRFDepth
from models.losses.loss_depth import SilogLoss

from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule

@MODELS.register_module(name='NewCRF')
class NewCRF(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # model
        self.depth_net = NewCRFDepth(version=opt.model.encoder,\
                                     inv_depth=False,\
                                     pre_trained=opt.model.pre_trained,\
                                     ckpt_path=opt.model.ckpt_path,\
                                     frozen_stages=-1,\
                                     min_depth=opt.model.min_depth,\
                                     max_depth=opt.model.max_depth)  

        self.criterion  = SilogLoss()

    def get_optimize_param(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.optim_opt.learning_rate}
        ]
        return optim_params

    def get_losses(self, pred_depth, gt_depth):
        loss = self.criterion(pred_depth, gt_depth)
        return loss
