# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/cleinc/bts
import torch

from models.network import BtsModel
from models.losses.loss_depth import SilogLoss

from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule

@MODELS.register_module(name='BTS')
class BTS(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # model
        self.depth_net = BtsModel(params=opt.model)
        self.criterion  = SilogLoss()

    def get_optimize_param(self):
        optim_params = [
            {'params': self.depth_net.encoder.parameters(), 'lr': self.optim_opt.learning_rate},
            {'params': self.depth_net.decoder.parameters(), 'weight_decay': 0, 'lr': self.optim_opt.learning_rate},
        ]
        return optim_params

    # overide "inference depth" function
    def inference_depth(self, tgt_img, gt_depth=None):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        lpg8x8, lpg4x4, lpg2x2, reduc1x1, pred_depth = self.depth_net(tgt_img)
        return pred_depth

    def get_losses(self, predictions, gt_depth):
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, pred_depth = predictions
        loss = self.criterion(pred_depth, gt_depth)
        return loss

    def val_additional_vis(self, batch, predictions, batch_idx):
        lpg8x8, lpg4x4, lpg2x2, reduc1x1, pred_depth = predictions

        # plot
        if batch_idx < 2:
            vis_red  = visualize_depth(reduc1x1[0].squeeze())  # (3, H, W)
            vis_lpg1 = visualize_depth(lpg2x2[0].squeeze())  # (3, H, W)
            vis_lpg2 = visualize_depth(lpg4x4[0].squeeze())  # (3, H, W)
            vis_lpg3 = visualize_depth(lpg8x8[0].squeeze())  # (3, H, W)

            stack = torch.cat([vis_red, vis_lpg1, vis_lpg2, vis_lpg3], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)

            self.logger.experiment.add_images(
                'val/Adabin_lpg_{}'.format(batch_idx), stack, self.current_epoch)
        return errs
