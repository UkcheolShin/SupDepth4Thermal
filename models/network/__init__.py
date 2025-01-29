# The code for each network is migrated from their code base
# DORN: https://github.com/liviniuk/DORN_depth_estimation_Pytorch
# BTS: https://github.com/cleinc/bts
# MiDaS: https://github.com/isl-org/MiDaS
# AdaBins: https://github.com/shariqfarooq123/AdaBins
# NeWCRF: https://github.com/aliyun/NeWCRFs

# PSMNet: https://github.com/JiaRenChang/PSMNet
# GWCNet: https://github.com/xy-guo/GwcNet
# CFNet: https://github.com/gallenszl/CFNet
# AANet: https://github.com/haofeixu/aanet
# ACVNet: https://github.com/gangweiX/ACVNet

# monocular depth network
from .dorn.dorn import DeepOrdinalRegression
from .bts.bts import BtsModel
from .midas import DPTDepthModel, MidasNet, MidasNet_small
from .adabin import UnetAdaptiveBins
from .newcrf import NewCRFDepth

# stereo depth network
from .aanet import AANetModel
from .psmnet import PSMNetModel
from .acvnet import ACVNetModel
from .cfnet import CFNetModel
from .gwcnet import GWCNetModel_G, GWCNetModel_GC
from .ms_crf import MonoStereoCRFDepth
