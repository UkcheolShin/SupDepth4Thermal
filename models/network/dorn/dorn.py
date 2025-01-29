#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 21:06
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : dorn.py
https://github.com/liviniuk/DORN_depth_estimation_Pytorch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network.dorn.modules import ResNetBackbone, SceneUnderstandingModule, OrdinalRegressionLayer

def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
                    
class DeepOrdinalRegression(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0,
                 input_size=(256, 640), kernel_size=16, pyramid=[8, 12, 16],
                 batch_norm=False, num_layers=101,
                 discretization="SID", pretrained=False, num_channel=1):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = np.uint8(ord_num)
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization
        self.pretrained = pretrained

        self.backbone = ResNetBackbone(num_layers=num_layers, pretrained=pretrained, num_channel=num_channel)
        self.SceneUnderstandingModule = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid,
                                                                 batch_norm=batch_norm)
        self.regression_layer = OrdinalRegressionLayer()

    def forward(self, image):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        N, C, H, W = image.shape
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        # print("feat shape:", feat.shape)
        # feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=True)

        if self.training:
            prob = self.regression_layer(feat)
            return prob

        prob, label = self.regression_layer(feat)
        # print("prob shape:", prob.shape, " label shape:", label.shape)
        if self.discretization == "SID":
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        # print("depth min:", torch.min(depth), " max:", torch.max(depth),
        #       " label min:", torch.min(label), " max:", torch.max(label))
        return {"target": depth, "prob": prob, "label": label}

    def train(self, mode=True):
        """
            Override train() to keep BN and first two conv layers frozend.
        """
        super().train(mode)
        
        if self.pretrained:
            # Freeze BatchNorm layers
            for module in self.modules():
                if isinstance(module, nn.modules.BatchNorm2d):
                    module.eval()

            # Freeze first two conv layers
            self.backbone.backbone.conv1.eval()
            self.backbone.backbone.conv2.eval()
        
        return self
        
    def get_1x_lr_params(self):
        for k in self.backbone.parameters():
            if k.requires_grad:
                yield k

    def get_10x_lr_params(self):
        for module in [self.SceneUnderstandingModule, self.regression_layer]:
            for k in module.parameters():
                if k.requires_grad:
                    yield k