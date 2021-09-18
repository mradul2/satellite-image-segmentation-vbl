"""
Cross Entropy 2D for CondenseNet
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super(CrossEntropyLoss, self).__init__()
        if config == None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
                                            size_average=True, 
                                            reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)