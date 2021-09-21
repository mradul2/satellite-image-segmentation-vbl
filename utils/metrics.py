"""
This file will contain the metrics of the framework
"""
import numpy as np
import torch

class IoUAccuracy:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, config=None):
        self.num_classes = config.num_classes
        self.SMOOTH = 1e-6

    def evaluate(self, outputs, labels):
        accu = np.zeros((self.num_classes,), dtype=float)
        iou = np.zeros((self.num_classes,), dtype=float)

        output_cvt = torch.argmax(outputs, dim=1)

        np_outputs = output_cvt.cpu().detach().numpy()
        np_labels = labels.cpu().detach().numpy()

        np_outputs[np_labels == config.ignore_index] = config.ignore_index

        for x in range(num_classes):
            output_mask = (np_outputs == x)
            label_mask = (np_labels == x)
        
            intersection = (output_mask & label_mask).sum((1, 2))
            union = (output_mask | label_mask).sum((1, 2))
            total = label_mask.sum((1, 2))

            iou[x] = ((intersection + self.SMOOTH) / (union + self.SMOOTH)).mean()
            accu[x] = ((intersection + self.SMOOTH) / (total + self.SMOOTH)).mean()

        return iou, accu