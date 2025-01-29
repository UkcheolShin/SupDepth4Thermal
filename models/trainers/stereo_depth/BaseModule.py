# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
import numpy as np

from pytorch_lightning import LightningModule
from models.metrics.eval_metric import compute_depth_errors, compute_disp_errors
from utils.visualization import *

class StereoDepthBaseModule(LightningModule):
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

    def forward(self, left_img, right_img):
        # in lightning, forward defines the prediction/inference actions for training
        B,C,H,W = left_img.shape
        if C == 1: # for the single-channel input
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        predictions = self.disp_net(left_img, right_img)
        return predictions

    def inference_disp(self, left_img, right_img):
        B,C,H,W = left_img.shape
        if C == 1:
            left_img  = left_img.repeat_interleave(3, axis=1)
            right_img  = right_img.repeat_interleave(3, axis=1)

        pred_disp = self.disp_net(left_img, right_img)
        return pred_disp

    def training_step(self, batch, batch_idx):
        left_img = batch["tgt_left"]
        right_img = batch["tgt_right"]
        disp_gt = batch["tgt_disp_gt"]

        # network forward
        predictions = self.forward(left_img, right_img)
        total_loss  = self.get_losses(predictions, disp_gt)

        # record log
        self.log('train/total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):

        left_img  = batch["tgt_left"]
        right_img = batch["tgt_right"]
        left_vis  = batch["tgt_left_eh"]
        depth_gt  = batch["tgt_depth_gt"]
        disp_gt   = batch["tgt_disp_gt"]
        focal     = batch["focal"]
        baseline  = batch["baseline"]

        pred_disp  = self.inference_disp(left_img, right_img)
        pred_depth = baseline[0] * focal[0] / (pred_disp +1e-10)

        errs_disp  = compute_disp_errors(disp_gt, pred_disp)
        errs_depth = compute_depth_errors(depth_gt, pred_depth, self.dataset_name)

        errs = {'abs_rel': errs_depth[1], 'sq_rel': errs_depth[2], 
                'rmse': errs_depth[4], 'rmse_log': errs_depth[5],
                'a1': errs_depth[6], 'a2': errs_depth[7], 'a3': errs_depth[8],
                'epe' : errs_disp[0], 'd1' : errs_disp[1], 'thres1' : errs_disp[2],
                'thres2' : errs_disp[3], 'thres3' : errs_disp[4]}

        # plot
        if batch_idx < 2:
            if left_vis[0].size(-1) != pred_depth[0].size(-1):
                C,H,W = left_vis[0].size()
                pred_depth = torch.nn.functional.interpolate(pred_depth, [H, W], mode='nearest')
                pred_disp  = torch.nn.functional.interpolate(pred_disp, [H, W], mode='nearest')

            vis_img = visualize_image(left_vis[0])  # (3, H, W)
            vis_disp = visualize_depth(pred_disp[0].squeeze())  # (3, H, W)
            vis_depth = visualize_depth(pred_depth[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_disp, vis_depth], dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            self.logger.experiment.add_images(
                'val/img_disp_depth_{}'.format(batch_idx), stack, self.current_epoch)
        return errs

    def validation_epoch_end(self, outputs):
        mean_rel    = np.array([x['abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['sq_rel'] for x in outputs]).mean()
        mean_rmse   = np.array([x['rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['a3'] for x in outputs]).mean()

        mean_epe = np.array([x['epe'] for x in outputs]).mean()
        mean_d1 = np.array([x['d1'] for x in outputs]).mean()
        mean_th1 = np.array([x['thres1'] for x in outputs]).mean()
        mean_th2 = np.array([x['thres2'] for x in outputs]).mean()
        mean_th3 = np.array([x['thres3'] for x in outputs]).mean()

        self.log('val_loss', mean_epe, prog_bar=True)
        self.log('val/abs_rel', mean_rel)
        self.log('val/sq_rel', mean_sq_rel)
        self.log('val/rmse', mean_rmse)
        self.log('val/rmse_log', mean_rmse_log)
        self.log('val/a1', mean_a1)
        self.log('val/a2', mean_a2)
        self.log('val/a3', mean_a3)  

        self.log('val/epe', mean_epe)
        self.log('val/d1', mean_d1)
        self.log('val/th1', mean_th1)
        self.log('val/th2', mean_th2)
        self.log('val/th3', mean_th3)

    def test_step(self, batch_data, batch_idx, dataloader_idx=0):
        return self.validation_step(batch_data, batch_idx, dataloader_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)
