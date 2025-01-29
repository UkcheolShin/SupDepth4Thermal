# Written by Ukcheol Shin (shinwc159[at]gmail.com)
# Reference: https://github.com/haofeixu/aanet
import torch
from models.network import AANetModel
from models.registry import MODELS
from models.trainers.stereo_depth.BaseModule import StereoDepthBaseModule

def filter_specific_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False

def filter_base_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True

@MODELS.register_module(name='AANet')
class AANet(StereoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # Network
        self.disp_net = AANetModel(max_disp=opt.model.max_disp,
                           num_downsample=opt.model.num_downsample,
                           feature_type=opt.model.feature_type,
                           no_feature_mdconv=opt.model.no_feature_mdconv,
                           feature_pyramid=opt.model.feature_pyramid,
                           feature_pyramid_network=opt.model.feature_pyramid_network,
                           feature_similarity=opt.model.feature_similarity,
                           aggregation_type=opt.model.aggregation_type,
                           num_scales=opt.model.num_scales,
                           num_fusions=opt.model.num_fusions,
                           num_stage_blocks=opt.model.num_stage_blocks,
                           num_deform_blocks=opt.model.num_deform_blocks,
                           no_intermediate_supervision=opt.model.no_intermediate_supervision,
                           refinement_type=opt.model.refinement_type,
                           mdconv_dilation=opt.model.mdconv_dilation,
                           deformable_groups=opt.model.deformable_groups,
                           pre_trained=opt.model.pre_trained,
                           ckpt_path=opt.model.ckpt_path)

        if opt.model.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.disp_net.apply(set_bn_eval)

        self.highest_loss_only = opt.model.highest_loss_only
        self.max_disp = opt.model.max_disp

    def get_optimize_param(self):
        # Learning rate for offset learning is set 0.1 times those of existing layers
        specific_params = list(filter(filter_specific_params, self.disp_net.named_parameters()))
        base_params     = list(filter(filter_base_params, self.disp_net.named_parameters()))

        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]

        optim_params = [
            {'params': specific_params, 'lr': self.optim_opt.learning_rate*0.1},
            {'params': base_params, 'lr': self.optim_opt.learning_rate},
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
        # only the last highest resolution output
        if self.highest_loss_only:
            pred_disp_pyramid = [pred_disp_pyramid[-1]]  

        disp_loss = 0

        # Loss weights
        if len(pred_disp_pyramid) == 5:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
        elif len(pred_disp_pyramid) == 4:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
        elif len(pred_disp_pyramid) == 3:
            pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
        elif len(pred_disp_pyramid) == 1:
            pyramid_weight = [1.0]  # highest loss only
        else:
            raise NotImplementedError

        mask = (gt_disp > 0) & (gt_disp < self.max_disp)

        for k in range(len(pred_disp_pyramid)):
            pred_disp = pred_disp_pyramid[k]
            weight = pyramid_weight[k]

            if pred_disp.size(-1) != gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = torch.nn.functional.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            curr_loss = torch.nn.functional.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                         reduction='mean')
            disp_loss += weight * curr_loss

        loss = disp_loss 
        return loss
        
