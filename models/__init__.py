from .registry import MODELS

# Supervised Monocular Depth Network 
from .trainers.mono_depth.DORN import DORN
from .trainers.mono_depth.BTS import BTS
from .trainers.mono_depth.AdaBins import AdaBins
from .trainers.mono_depth.Midas import Midas
from .trainers.mono_depth.NewCRF import NewCRF

# Supervised Stereo Matching Network 
from .trainers.stereo_depth.PSMNet import PSMNet
from .trainers.stereo_depth.AANet import AANet
from .trainers.stereo_depth.GWCNet import GWCNet
from .trainers.stereo_depth.CFNet import CFNet
from .trainers.stereo_depth.ACVNet import ACVNet
from .trainers.stereo_depth.MonoStereoCRF import MonoStereoCRF
 