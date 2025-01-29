# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: CVPR23 Deep_Depth_Estimation_From_Thermal_Image
import torch

from models.network import MonoStereoCRFDepth

from models.registry import MODELS
from models.trainers.stereo_depth.BaseModule import StereoDepthBaseModule
from models.metrics.eval_metric import compute_depth_errors, compute_disp_errors
from utils.visualization import *

@MODELS.register_module(name='MonoStereoCRF')
class MonoStereoCRF(StereoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # Network
        self.disp_net = MonoStereoCRFDepth(maxdisp=opt.model.max_disp,\
                                            use_concat_volume=opt.model.use_concat_volume,\
                                            use_3d_decoder=opt.model.use_3d_decoder, \
                                            encoder_model=opt.model.encoder,\
                                            pre_trained=opt.model.pre_trained,\
                                            ckpt_path=opt.model.ckpt_path,\
                                            disp_head=opt.model.disp_head,)

        self.w_mono     = opt.model.mono_weight
        self.w_stereo   = opt.model.stereo_weight
        self.max_disp   = opt.model.max_disp
        self.criterion  = torch.nn.functional.smooth_l1_loss
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

        disp_mono, disp_stereo = self.disp_net(left_img, right_img)
        return disp_stereo

    def inference_depth(self, left_img):
        B,C,H,W = left_img.shape
        if C == 1:
            left_img  = left_img.repeat_interleave(3, axis=1)

        prediction = self.disp_net.forward_mono(left_img)
        return prediction

    def multiscale_loss(self, pred_disp_pyramid, gt_disp):
        mask = (gt_disp > 0) & (gt_disp < self.max_disp)
        all_losses = []
        for disp_est, weight in zip(pred_disp_pyramid, self.loss_weights):

            if disp_est.size(-1) != gt_disp.size(-1):
                disp_est = disp_est.unsqueeze(1)  # [B, 1, H, W]
                disp_est = torch.nn.functional.interpolate(disp_est, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / disp_est.size(-1))
                disp_est = disp_est.squeeze(1)  # [B, H, W]

            all_losses.append(weight * self.criterion(disp_est[mask], gt_disp[mask], size_average=True))
        loss = sum(all_losses)
        return loss

    def get_losses(self, predictions, gt_disp):
        disp_pyramid_mono_l, disp_pyramid_stereo = predictions
        loss_mono   = self.multiscale_loss(disp_pyramid_mono_l, gt_disp)
        loss_stereo = self.multiscale_loss(disp_pyramid_stereo, gt_disp)
        return self.w_mono*loss_mono + self.w_stereo*loss_stereo

    def validation_step(self, batch, batch_idx):
        left_img  = batch["tgt_left"]
        right_img = batch["tgt_right"]
        left_vis  = batch["tgt_left_eh"]
        depth_gt  = batch["tgt_depth_gt"]
        disp_gt   = batch["tgt_disp_gt"]
        focal     = batch["focal"]
        baseline  = batch["baseline"]

        pred_disp_mono, pred_disp_stereo = self.forward(left_img, right_img)

        pred_depth = baseline[0] * focal[0] / (pred_disp_mono +1e-10)

        errs_depth_m = compute_depth_errors(depth_gt, pred_depth)
        errs_disp_m  = compute_disp_errors(disp_gt, pred_disp_mono)

        pred_depth = baseline[0] * focal[0] / (pred_disp_stereo +1e-10)

        errs_depth = compute_depth_errors(depth_gt, pred_depth)
        errs_disp = compute_disp_errors(disp_gt, pred_disp_stereo)

        errs = {'abs_rel': errs_depth[1], 'sq_rel': errs_depth[2], 
                'rmse': errs_depth[4], 'rmse_log': errs_depth[5],
                'a1': errs_depth[6], 'a2': errs_depth[7], 'a3': errs_depth[8],
                'epe' : errs_disp[0], 'd1' : errs_disp[1], 'thres1' : errs_disp[2],
                'thres2' : errs_disp[3], 'thres3' : errs_disp[4],
                'mono/abs_rel': errs_depth_m[1], 'mono/sq_rel': errs_depth_m[2], 
                'mono/rmse': errs_depth_m[4], 'mono/rmse_log': errs_depth_m[5],
                'mono/a1': errs_depth_m[6], 'mono/a2': errs_depth_m[7], 'mono/a3': errs_depth_m[8],
                'mono/epe' : errs_disp_m[0], 'mono/d1' : errs_disp_m[1], 'mono/thres1' : errs_disp_m[2],
                'mono/thres2' : errs_disp_m[3], 'mono/thres3' : errs_disp_m[4]}

        # plot
        if batch_idx < 2:
            if left_vis[0].size(-1) != pred_depth[0].size(-1):
                C,H,W = left_vis[0].size()
                pred_depth = torch.nn.functional.interpolate(pred_depth, [H, W], mode='nearest')
                pred_disp_stereo  = torch.nn.functional.interpolate(pred_disp_stereo, [H, W], mode='nearest')
    
            vis_img = visualize_image(left_vis[0])  # (3, H, W)
            vis_disp = visualize_depth(pred_disp_stereo[0].squeeze())  # (3, H, W)
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

        mean_rel    = np.array([x['mono/abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['mono/sq_rel'] for x in outputs]).mean()
        mean_rmse   = np.array([x['mono/rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['mono/rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['mono/a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['mono/a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['mono/a3'] for x in outputs]).mean()

        mean_epe = np.array([x['mono/epe'] for x in outputs]).mean()
        mean_d1 = np.array([x['mono/d1'] for x in outputs]).mean()
        mean_th1 = np.array([x['mono/thres1'] for x in outputs]).mean()
        mean_th2 = np.array([x['mono/thres2'] for x in outputs]).mean()
        mean_th3 = np.array([x['mono/thres3'] for x in outputs]).mean()

        self.log('val/mono/abs_rel', mean_rel)
        self.log('val/mono/sq_rel', mean_sq_rel)
        self.log('val/mono/rmse', mean_rmse)
        self.log('val/mono/rmse_log', mean_rmse_log)
        self.log('val/mono/a1', mean_a1)
        self.log('val/mono/a2', mean_a2)
        self.log('val/mono/a3', mean_a3)

        self.log('val/mono/epe', mean_epe)
        self.log('val/mono/d1', mean_d1)
        self.log('val/mono/th1', mean_th1)
        self.log('val/mono/th2', mean_th2)
        self.log('val/mono/th3', mean_th3)
