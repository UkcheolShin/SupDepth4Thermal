# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
import numpy as np

from pytorch_lightning import LightningModule
from models.metrics.eval_metric import compute_depth_errors
from utils.visualization import *

class MonoDepthBaseModule(LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.optim_opt = opt.optim
        self.dataset_name = opt.dataset.list[0]
        self.automatic_optimization = True

    def get_optimize_param(self):
        pass
    
    def get_losses(self):
        pass
    
    def configure_optimizers(self):
        optim_params = self.get_optimize_param()

        if self.optim_opt.optimizer == 'Adam' :
            optimizer = torch.optim.Adam(optim_params)
        elif self.optim_opt.optimizer == 'AdamW' :
            optimizer = torch.optim.AdamW(optim_params)
        elif self.optim_opt.optimizer == 'SGD' :
            optimizer = torch.optim.SGD(optim_params)

        if self.optim_opt.scheduler == 'CosineAnnealWarm' :
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                            T_0=self.optim_opt.CosineAnnealWarm.T_0, 
                                            T_mult=self.optim_opt.CosineAnnealWarm.T_mult,
                                            eta_min=self.optim_opt.CosineAnnealWarm.eta_min)

        return [optimizer], [scheduler]

    def forward(self, tgt_img):
        # in lightning, forward defines the prediction/inference actions for training
        B,C,H,W = tgt_img.shape
        if C == 1: # for the single-channel input
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        pred_depth = self.depth_net(tgt_img)
        return pred_depth

    def inference_depth(self, tgt_img, gt_depth=None):
        B,C,H,W = tgt_img.shape
        if C == 1:
            tgt_img  = tgt_img.repeat_interleave(3, axis=1)

        pred_depth = self.depth_net(tgt_img)
        return pred_depth

    def training_step(self, batch, batch_idx):
        input_imgs = batch["tgt_image"]
        depth_gts  = batch["tgt_depth_gt"]

        # network forward
        predictions = self.forward(input_imgs)
        total_loss = self.get_losses(predictions, depth_gts)

        # record log
        self.log('train/total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        input_imgs    = batch["tgt_image"]
        input_eh_imgs = batch["tgt_image_eh"]
        depth_gts     = batch["tgt_depth_gt"]

        # inference depth
        pred_depths = self.inference_depth(input_imgs, depth_gts)
        errs = compute_depth_errors(depth_gts, pred_depths, self.dataset_name)

        errs = {'abs_rel': errs[1], 'sq_rel': errs[2], 
                'rmse': errs[4], 'rmse_log': errs[5],
                'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        # plot
        if batch_idx < 2:
            if input_eh_imgs[0].size(-1) != pred_depths[0].size(-1):
                C,H,W = input_eh_imgs[0].shape
                pred_depths = torch.nn.functional.interpolate(pred_depths, [H, W], mode='nearest')

            vis_img = visualize_image(input_eh_imgs[0])  # (3, H, W)
            vis_depth = visualize_depth(pred_depths[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_depth_{}'.format(batch_idx), stack, self.current_epoch)

        return errs

    def validation_epoch_end(self, outputs):
        mean_rel    = np.array([x['abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['sq_rel'] for x in outputs]).mean()
        mean_rmse   = np.array([x['rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['a3'] for x in outputs]).mean()

        self.log('val_loss', mean_rel, prog_bar=True)
        self.log('val/abs_rel', mean_rel)
        self.log('val/sq_rel', mean_sq_rel)
        self.log('val/rmse', mean_rmse)
        self.log('val/rmse_log', mean_rmse_log)
        self.log('val/a1', mean_a1)
        self.log('val/a2', mean_a2)
        self.log('val/a3', mean_a3)
    
    def test_step(self, batch_data, batch_idx, dataloader_idx=0):
        return self.validation_step(batch_data, batch_idx, dataloader_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)
