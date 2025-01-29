# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/isl-org/MiDaS
import torch

from models.network import DPTDepthModel, MidasNet, MidasNet_small
from models.losses.midas_loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift

from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule

@MODELS.register_module(name='Midas')
class Midas(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # define model
        self.model = opt.model.mode

        if opt.model.pre_trained == False:
            ckpt_path = None
        else:
            ckpt_path = opt.model.ckpt_path

        if self.model == 'midas':
            self.depth_net = MidasNet(path=ckpt_path, non_negative=True) 
        elif self.model == 'midas_small':
            self.depth_net = MidasNet_small(path=ckpt_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        elif self.model == 'dpt_hybrid':
            self.depth_net = DPTDepthModel(path=ckpt_path, backbone="vitb_rn50_384", non_negative=True) 
        elif self.model == 'dpt_large':
            self.depth_net = DPTDepthModel(path=ckpt_path, backbone="vitl16_384", non_negative=True) 

        self.criterion    = ScaleAndShiftInvariantLoss()
        self.min_depth   = opt.model.min_depth
        self.max_depth   = opt.model.max_depth

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

        pred_depth = self.depth_net(tgt_img)
        if gt_depth is not None:
            pred_depth = self.fit_scale_shfit_depth(pred_depth, gt_depth)
        return pred_depth

    def fit_scale_shfit_depth(self, tgt_depth, gt_depth):
        mask = (gt_depth > self.min_depth) & (gt_depth < self.max_depth)
        batch_size, h, w = gt_depth.size()
        if tgt_depth.nelement() != gt_depth.nelement():
            tgt_depth = torch.nn.functional.interpolate(tgt_depth.unsqueeze(1), [h, w], mode='bilinear').squeeze(1)

        scale, shift = compute_scale_and_shift(tgt_depth, gt_depth, mask)
        tgt_depth = scale.view(-1, 1, 1) * tgt_depth + shift.view(-1, 1, 1)
        return tgt_depth

    def get_losses(self, pred_depth, gt_depth):
        loss = self.criterion(pred_depth, gt_depth)
        return loss