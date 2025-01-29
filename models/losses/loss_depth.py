import torch
from torch import nn
import torch.nn.functional as F

class SilogLoss(nn.Module):
    """
    Compute SiLog loss. 
    See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf 
    """
    def __init__(self, scale=10., variance_focus=0.85):
        super().__init__()
        self.scale = scale
        self.variance_focus = variance_focus

    def forward(self, pred, gt):
        batch_size, h, w = gt.size()
        if pred.nelement() != gt.nelement():
            pred = F.interpolate(pred, [h, w], mode='bilinear')
        pred = pred.view(batch_size, h, w)

        # let's only compute the loss on non-null pixels from the ground-truth depth-map
        non_zero_mask = (gt > 0) & (pred > 0)

        log_diff = torch.log(pred[non_zero_mask]) - \
                   torch.log(gt[non_zero_mask])
        silog1 = (log_diff ** 2).mean()
        silog2 = self.variance_focus * (log_diff.mean() ** 2)
        silog_loss = torch.sqrt(silog1 - silog2) * self.scale
        return silog_loss

def sup_depth_loss(pred, gt):
    # pred : b c h w
    # gt: b h w
    batch_size, h, w = gt.size()

    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='bilinear') # bilinear, nearest

    pred = pred.view(batch_size, h, w)

    max_depth = 100
    min_depth = 1e-3

    valid = (gt > min_depth) & (gt < max_depth)
    valid_gt = gt[valid]
    valid_pred = pred[valid]

    # align scale
    # valid_pred = valid_pred * \
    #     torch.median(valid_gt)/torch.median(valid_pred)

    sup_loss = F.smooth_l1_loss(valid_gt, valid_pred, reduction='mean')

    return sup_loss


def compute_smooth_loss(tgt_depth, tgt_img):
    def get_smooth_loss(disp, img):
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(
            torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(
            torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth, tgt_img)

    return loss

# for chamfer_distance
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence

class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss