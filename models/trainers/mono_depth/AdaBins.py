# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/shariqfarooq123/AdaBins
import torch

from models.network import UnetAdaptiveBins
from models.losses.loss_depth import SilogLoss, BinsChamferLoss

from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule

@MODELS.register_module(name='AdaBins')
class AdaBins(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # define model
        self.depth_net = UnetAdaptiveBins.build(n_bins=opt.model.n_bins,\
                                                min_val=opt.model.min_depth,\
                                                max_val=opt.model.max_depth,\
                                                norm=opt.model.norm)

        self.criterion_ueff = SilogLoss()
        self.criterion_bins = BinsChamferLoss()
        self.w_bin = opt.model.bin_weight

    def get_optimize_param(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.optim_opt.learning_rate}
        ]
        return optim_params

    # overide "inference depth" function
    def inference_depth(self, tgt_img, gt_depth=None):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        bin_edges, pred_depth = self.depth_net(tgt_img)
        return pred_depth

    def get_losses(self, predictions, gt_depth):
        bin_edges  = predictions[0]
        pred_depth = predictions[1]

        loss_bin = self.criterion_bins(bin_edges, gt_depth)
        loss_sup = self.criterion_ueff(pred_depth, gt_depth)

        loss = loss_sup + self.w_bin*loss_bin 

        return loss