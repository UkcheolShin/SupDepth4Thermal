#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-12-24 21:47
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : __init__.py.py
"""

from .backbones.resnet import ResNetBackbone
from .encoders.SceneUnderstandingModule import SceneUnderstandingModule
from .decoders.OrdinalRegression import OrdinalRegressionLayer
