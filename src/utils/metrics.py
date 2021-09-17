"""
This file will contain the metrics of the framework
"""
import numpy as np


class CityMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def evaluate(self, outputs, labels):
        accu = np.zeros((num_classes,), dtype=float)
        iou = np.zeros((num_classes,), dtype=float)

        output_cvt = torch.argmax(outputs, dim=1)

        np_outputs = output_cvt.cpu().detach().numpy()
        np_labels = labels.cpu().detach().numpy()

        np_outputs[np_labels == 19] = 19

        for x in range(num_classes):
            output_mask = (np_outputs == x)
            label_mask = (np_labels == x)
        
            intersection = (output_mask & label_mask).sum((1, 2))
            union = (output_mask | label_mask).sum((1, 2))
            total = label_mask.sum((1, 2))

            iou[x] = ((intersection + SMOOTH) / (union + SMOOTH)).mean()
            accu[x] = ((intersection + SMOOTH) / (total + SMOOTH)).mean()

        return iou, accu