# Written by Ukcheol Shin (shinwc159@gmail.com)
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
import random
import math

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP

class MonoStereoCRFDepth(nn.Module):
    def __init__(self, maxdisp, use_swin_backbone=True, \
                 use_concat_volume=False, use_3d_decoder=False, \
                 encoder_model='tiny', pre_trained=False, ckpt_path='',\
                 disp_head='classify'):
        super(MonoStereoCRFDepth, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.num_groups = 40
        self.use_3d_decoder = use_3d_decoder
        self.use_swin_backbone = use_swin_backbone
        self.encoder_model = encoder_model

        norm_cfg = dict(type='BN', requires_grad=True)

        window_size = 7

        if self.encoder_model == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif self.encoder_model == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif self.encoder_model == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=-1
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.feat_encoder = SwinTransformer(**backbone_cfg)
        self.costvolume_builter = self.build_2d_volume

        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, 512+6]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2]+12, embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1]+24, embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0]+48, embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0], mode=disp_head, max_disp=self.maxdisp)
        self.disp_head2 = DispHead(input_dim=crf_dims[1], mode=disp_head, max_disp=self.maxdisp//2)
        self.disp_head3 = DispHead(input_dim=crf_dims[2], mode=disp_head, max_disp=self.maxdisp//4)
        self.disp_head4 = DispHead(input_dim=crf_dims[3], mode=disp_head, max_disp=self.maxdisp//4)
        if disp_head == 'regression':
            self.scale_factor = [4, 8, 16, 32] # the output is a value between range 0-1, after upsample original resolution, then multiply 192
        elif disp_head == 'classify':
            self.scale_factor = [4, 2, 1, 1] # the output channel is 48, 48*4 ==> 192 disp channel

        if pre_trained == False:
            self.init_weights(pretrained=None)
        else:
            self.init_weights(pretrained=ckpt_path)
            
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.feat_encoder.init_weights(pretrained=pretrained)
        self.decoder.init_weights()

    def build_2d_volume(self, feat_l, feat_r, scale=4):
        # Bx D/4 x H/4 x W/4
        # candidate information
        if self.use_swin_backbone:
            l1c_volume = build_l1c_volume(feat_l, feat_r, self.maxdisp // scale)
        else:
            l1c_volume = build_l1c_volume(feat_l["gwc_feature"], feat_r["gwc_feature"], self.maxdisp // 4)

        # Bx 1 x H/4 x W/4
        # initial candidate
        if self.use_swin_backbone:
            init_disp  = l1c_volume.argmin(dim=1,keepdim=True)
            # 256+48+1 
            # volume = torch.cat((feat_l, init_disp, l1c_volume), 1)
        else:
            init_disp  = l1c_volume.argmin(dim=1,keepdim=True)
            # 320+48+1
            # volume = torch.cat((feat_l["gwc_feature"], init_disp, l1c_volume), 1)

        return l1c_volume

    def forward(self, left, right):
        # B x 320(64 128 128) x H/4 x W/4
        feats_left = self.feat_encoder(left)
        feats_right = self.feat_encoder(right)

        agg_ppm_left = self.decoder(feats_left)
        agg_ppm_right = self.decoder(feats_right)
        
        volume = self.costvolume_builter(agg_ppm_left, agg_ppm_right, 32)
        agg_volume_stereo = torch.cat((agg_ppm_left, volume), 1)
        e3 = self.crf3(feats_left[3], agg_volume_stereo)
        if self.training:
            d3 = self.disp_head4(e3, self.scale_factor[3])

        e3 = nn.PixelShuffle(2)(e3)
        volume = self.costvolume_builter(feats_left[2], feats_right[2], 16)
        feat = torch.cat((feats_left[2], volume), 1)
        e2 = self.crf2(feat, e3)
        if self.training:
            d2 = self.disp_head3(e2, self.scale_factor[2])

        e2 = nn.PixelShuffle(2)(e2)
        volume = self.costvolume_builter(feats_left[1], feats_right[1], 8)
        feat = torch.cat((feats_left[1], volume), 1)
        e1 = self.crf1(feat, e2)
        if self.training:
            d1 = self.disp_head2(e1, self.scale_factor[1])

        e1 = nn.PixelShuffle(2)(e1)
        volume = self.costvolume_builter(feats_left[0], feats_right[0], 4)
        feat = torch.cat((feats_left[0], volume), 1)

        e0 = self.crf0(feat, e1)
        d0 = self.disp_head1(e0, self.scale_factor[0])

        if self.training:
            disp_stereo =  [d3, d2, d1, d0] 
        else:
            disp_stereo =  d0 

        volume_mono = self.costvolume_builter(agg_ppm_left, agg_ppm_left, 32)
        agg_volume_mono = torch.cat((agg_ppm_left, volume_mono), 1)
        e3 = self.crf3(feats_left[3], agg_volume_mono)
        if self.training:
            d3 = self.disp_head4(e3, self.scale_factor[3])

        e3 = nn.PixelShuffle(2)(e3)
        volume = self.costvolume_builter(feats_left[2], feats_left[2], 16)
        feat = torch.cat((feats_left[2], volume), 1)
        e2 = self.crf2(feat, e3)
        if self.training:
            d2 = self.disp_head3(e2, self.scale_factor[2])

        e2 = nn.PixelShuffle(2)(e2)
        volume = self.costvolume_builter(feats_left[1], feats_left[1], 8)
        feat = torch.cat((feats_left[1], volume), 1)
        e1 = self.crf1(feat, e2)
        if self.training:
            d1 = self.disp_head2(e1, self.scale_factor[1])

        e1 = nn.PixelShuffle(2)(e1)
        volume = self.costvolume_builter(feats_left[0], feats_left[0], 4)
        feat = torch.cat((feats_left[0], volume), 1)

        e0 = self.crf0(feat, e1)
        d0 = self.disp_head1(e0, self.scale_factor[0])
        if self.training:
            disp_mono_l = [d3, d2, d1, d0] 
        else:
            disp_mono_l =  d0 

        return disp_mono_l, disp_stereo

    def forward_mono(self, left):
        # B x 320(64 128 128) x H/4 x W/4
        feats_left = self.feat_encoder(left)
        agg_ppm_left = self.decoder(feats_left)

        volume_mono = self.costvolume_builter(agg_ppm_left, agg_ppm_left, 32)
        agg_volume_mono = torch.cat((agg_ppm_left, volume_mono), 1)
        e3 = self.crf3(feats_left[3], agg_volume_mono)

        e3 = nn.PixelShuffle(2)(e3)
        volume = self.costvolume_builter(feats_left[2], feats_left[2], 16)
        feat = torch.cat((feats_left[2], volume), 1)
        e2 = self.crf2(feat, e3)

        e2 = nn.PixelShuffle(2)(e2)
        volume = self.costvolume_builter(feats_left[1], feats_left[1], 8)
        feat = torch.cat((feats_left[1], volume), 1)
        e1 = self.crf1(feat, e2)

        e1 = nn.PixelShuffle(2)(e1)
        volume = self.costvolume_builter(feats_left[0], feats_left[0], 4)
        feat = torch.cat((feats_left[0], volume), 1)

        e0 = self.crf0(feat, e1)
        d0 = self.disp_head1(e0, self.scale_factor[0])

        disp_mono_l =  d0 
        return disp_mono_l

class DispHead(nn.Module):
    def __init__(self, input_dim=128, mode='regression', max_disp=192):
        super(DispHead, self).__init__()
        self.mode = mode

        if self.mode == 'regression':
            self.head = self.regressor(in_channel=input_dim, out_channel=1)
            self.forward = self.forward_regression
        elif self.mode == 'classify':
            self.head = self.classifer(in_channel=input_dim, out_channel=48)
            self.forward = self.forward_classification
        else:
            print('Unsupported type!')

        self.max_disp = max_disp

    def convbn(self, in_channels, out_channels, kernel_size, stride, pad, dilation):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                             nn.BatchNorm2d(out_channels))

    def classifer(self, in_channel, out_channel):
        return nn.Sequential(self.convbn(in_channel, in_channel, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False))

    def regressor(self, in_channel, out_channel=1):
        return nn.Sequential(self.convbn(in_channel, in_channel, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
                                      nn.Sigmoid()) # 0-1 range

    def upsample(self, x, scale_factor=2, mode="bilinear", align_corners=False):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    def disparity_regression(self, x, maxdisp):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, maxdisp, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward_regression(self, x, scale):
        # x = self.relu(self.norm1(x))
        disp = self.head(x)
        if scale > 1:
            disp = self.upsample(disp, scale_factor=scale)
        return disp.squeeze(1) * self.max_disp

    def forward_classification(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.head(x)
        if scale > 1:
            x = self.upsample(x.unsqueeze(1), scale_factor=scale, mode='trilinear')
            x = torch.squeeze(x, 1)
        pred = F.softmax(x, dim=1)
        disp = self.disparity_regression(pred, self.max_disp)
        return disp
