# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models import LOSSES, weighted_loss

from pytorch_metric_learning.losses import NTXentLoss, CrossBatchMemory
from pytorch_metric_learning.utils import distributed as pml_dist
from mmcv.runner import get_dist_info

@weighted_loss
def NTLoss(pred, target, loss_function):
    loss = loss_function(pred, target)
    return loss

@LOSSES.register_module()
class SimCLRLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/
        master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Args:
        margin (float, optional): Margin for triplet loss. Default to 0.3.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.5, memory_size = 36):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.loss_function = NTXentLoss(temperature=temperature)
        rank, world_size = get_dist_info()
        if world_size > 1:
            self.loss_function = pml_dist.DistributedLossWrapper(self.loss_function)

    def forward(self, inputs, targets):
        loss_cls = self.loss_weight * NTLoss(inputs, targets, None, loss_function = self.loss_function, reduction='mean', avg_factor=None)
        return loss_cls

