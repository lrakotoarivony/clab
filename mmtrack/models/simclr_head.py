# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses.sim_clr_loss import SimCLRLoss
from mmcv.runner import get_dist_info

class SimCLRHead(nn.Module):
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

    def __init__(self, loss_weight=1.0, sequence_num = 2, temperature = 0.5, channel_start = 2048):
        super(SimCLRHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.loss_decode = SimCLRLoss(loss_weight = loss_weight, temperature = temperature)
        self.iter = 1
        self.T = sequence_num
        self.channel_start = channel_start

        self.channel = nn.Sequential(nn.Conv2d(self.channel_start,512 ,kernel_size = 3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def loss(self, embeddings):
        embeddings = self.channel(embeddings)
        embeddings = self.pool(embeddings)
        embeddings = torch.flatten(embeddings,1)

        embeddings = self.proj(embeddings)

        B = embeddings.shape[0] // (self.T + 1)

        rank, world_size = get_dist_info()
        targets = torch.arange(B)
        targets = torch.cat((targets,targets.repeat_interleave(self.T)))
        targets += self.iter * (B * world_size)
        targets += B * rank

        loss = self.loss_decode(embeddings, targets)
        losses = dict()
        losses['loss_simclr'] = [loss]
        return losses
