"""
This file will contain the metrics of the framework
"""

# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": iu,
    }

import numpy as np
import torch

class IoUAccuracy:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, config=None):
        self.config = config
        self.num_classes = config.num_classes

    def evaluate(self, outputs, labels):
        output_cvt = torch.argmax(outputs, dim=1)
        np_outputs = output_cvt.cpu().detach().numpy()
        np_labels = labels.cpu().detach().numpy()

        score_dict = scores(np_labels, np_outputs, self.num_classes)

        iou = score_dict["Class IoU"]
        accu = score_dict["Pixel Accuracy"]

        return np_outputs.squeeze(), iou, accu