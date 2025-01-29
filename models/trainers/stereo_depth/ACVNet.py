# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/gangweiX/ACVNet
import torch

from models.network import ACVNetModel
from models.registry import MODELS
from models.trainers.stereo_depth.BaseModule import StereoDepthBaseModule

@MODELS.register_module(name='ACVNet')
class ACVNet(StereoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # Network
        self.disp_net = ACVNetModel(maxdisp=opt.model.max_disp,
                           attn_weights_only=opt.model.attention_weights_only,
                           freeze_attn_weights=opt.model.freeze_attention_weights,
                           pre_trained=opt.model.pre_trained,
                           ckpt_path=opt.model.ckpt_path)
        self.max_disp = opt.model.max_disp
        self.criterion = torch.nn.functional.smooth_l1_loss

        if opt.model.attention_weights_only:
            self.loss_weights = [1.0]
        elif opt.model.freeze_attention_weights:
            self.loss_weights = [0.5, 0.7, 1.0]
        else:
            self.loss_weights = [0.5, 0.5, 0.7, 1.0] 

    def get_optimize_param(self):
        optim_params = [
            {'params': self.disp_net.parameters(), 'lr': self.optim_opt.learning_rate},
        ]
        return optim_params

    def inference_disp(self, left_img, right_img):
        B,C,H,W = left_img.shape
        if C == 1:
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        pred_disp_pyramid = self.disp_net(left_img, right_img)
        return pred_disp_pyramid[-1]

    def get_losses(self, pred_disp_pyramid, gt_disp):
        mask = (gt_disp > 0) & (gt_disp < self.max_disp)

        all_losses = []
        for disp_est, weight in zip(pred_disp_pyramid, self.loss_weights):
            all_losses.append(weight * self.criterion(disp_est[mask], gt_disp[mask], size_average=True))
        loss = sum(all_losses)
        return loss
