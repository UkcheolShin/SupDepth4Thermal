# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/JiaRenChang/PSMNet
import torch

from models.network import PSMNetModel
from models.registry import MODELS
from models.trainers.stereo_depth.BaseModule import StereoDepthBaseModule

@MODELS.register_module(name='PSMNet')
class PSMNet(StereoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # Network
        self.disp_net = PSMNetModel(maxdisp=opt.model.max_disp)
        self.criterion = torch.nn.functional.smooth_l1_loss

        self.max_disp = opt.model.max_disp

    def get_optimize_param(self):
        optim_params = [
            {'params': self.disp_net.parameters(), 'lr': self.optim_opt.learning_rate},
        ]
        return optim_params

    def get_losses(self, predictions, gt_disp):
        disp0, disp1, disp2 = predictions

        mask = (gt_disp > 0) & (gt_disp < self.max_disp)
        loss = 0.5 * self.criterion(disp0.squeeze()[mask], gt_disp[mask], reduction='mean') 
        loss += 0.7 * self.criterion(disp1.squeeze()[mask], gt_disp[mask], reduction='mean') 
        loss += self.criterion(disp2.squeeze()[mask], gt_disp[mask], reduction='mean')
        return loss