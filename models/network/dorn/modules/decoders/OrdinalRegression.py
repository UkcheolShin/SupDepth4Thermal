#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 17:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : oridinal_regression_layer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        # implementation according to the paper
        # A = x[:, ::2, :, :]
        # B = x[:, 1::2, :, :]
        #
        # # A = A.reshape(N, 1, ord_num * H * W)
        # # B = B.reshape(N, 1, ord_num * H * W)
        # A = A.unsqueeze(dim=1)
        # B = B.unsqueeze(dim=1)
        # concat_feats = torch.cat((A, B), dim=1)
        #
        # if self.training:
        #     prob = F.log_softmax(concat_feats, dim=1)
        #     ord_prob = x.clone()
        #     ord_prob[:, 0::2, :, :] = prob[:, 0, :, :, :]
        #     ord_prob[:, 1::2, :, :] = prob[:, 1, :, :, :]
        #     return ord_prob
        #
        # ord_prob = F.softmax(concat_feats, dim=1)[:, 0, ::]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1).reshape((N, 1, H, W))
        # return ord_prob, ord_label

        # reimplementation for fast speed.

        x = x.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(x, dim=1).view(N, C, H, W)
            return prob

        ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        ord_label = torch.sum((ord_prob > 0.5), dim=1)
        return ord_prob, ord_label


# class OrdinalRegressionLayer(nn.Module):
#     def __init__(self):
#         super(OrdinalRegressionLayer, self).__init__()

#     def forward(self, x):
#         """
#         :param x: N x 2K x H x W; N - batch_size, 2K - channels, K - number of discrete sub-intervals
#         :return:  labels - ordinal labels (corresponding to discrete depth values) of size N x 1 x H x W
#                   softmax - predicted softmax probabilities P (as in the paper) of size N x K x H x W
#         """
#         N, K, H, W = x.size()
#         K = K // 2 # number of discrete sub-intervals
        
#         odd = x[:, ::2, :, :].clone()
#         even = x[:, 1::2, :, :].clone()

#         odd = odd.view(N, 1, K * H * W)
#         even = even.view(N, 1, K * H * W)

#         paired_channels = torch.cat((odd, even), dim=1)
#         paired_channels = paired_channels.clamp(min=1e-8, max=1e8)  # prevent nans

#         softmax = nn.functional.softmax(paired_channels, dim=1)

#         softmax = softmax[:, 1, :]
#         softmax = softmax.view(-1, K, H, W)
#         labels = torch.sum((softmax > 0.5), dim=1).view(-1, 1, H, W)
#         return labels, softmax