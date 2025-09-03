# Copyright (c) OpenMMLab. All rights reserved.
from .l2_loss import L2Loss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss
from .triplet_loss import TripletLoss
from .sim_clr_loss import SimCLRLoss

__all__ = ['L2Loss', 'TripletLoss', 'MultiPosCrossEntropyLoss', 'SimCLRLoss']
