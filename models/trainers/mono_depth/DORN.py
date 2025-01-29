# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/liviniuk/DORN_depth_estimation_Pytorch
import torch

from models.network import DeepOrdinalRegression
from models.losses.ordinal_regression_loss import OrdinalRegressionLoss

from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule

@MODELS.register_module(name='DORN')
class DORN(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # model
        self.depth_net = DeepOrdinalRegression(ord_num=opt.model.ord_num,\
                                               beta=opt.model.beta,\
                                               num_layers=opt.model.resnet_layers,\
                                               discretization=opt.model.discretization,\
                                               num_channel=3)
        self.criterion = OrdinalRegressionLoss(opt.model.ord_num, opt.model.beta, \
                                               opt.model.discretization)

    def get_optimize_param(self):
        optim_params = [
            {'params': self.depth_net.get_1x_lr_params(), 'lr': self.optim_opt.learning_rate/10.},
            {'params': self.depth_net.get_10x_lr_params(), 'lr': self.optim_opt.learning_rate},
        ]
        return optim_params

    # overide "inference depth" function
    def inference_depth(self, tgt_img, gt_depth=None):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        prediction = self.depth_net(tgt_img)
        return prediction["target"]
        
    def get_losses(self, pred_prob, gt_depth):
        loss = self.criterion(pred_prob, gt_depth)
        return loss
