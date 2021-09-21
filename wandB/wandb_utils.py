import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def init_wandb(model, config) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        model (Torch Model): Model for Training
        args (TrainOptions,optional): TrainOptions class (refer options/train_options.py). Defaults to None.
    """

    wandb.login(key=config.wandb_api_key)
    wandb.init(
        name=config.experiment,
        project=config.wandb_project,
        id=config.wandb_id,
        dir="./",
    )

    wandb.watch(model, log="all")


def wandb_log(train_loss: float, val_loss: float, train_acc: float, val_acc: float, train_iou: float, val_iou: float, epoch: int):
    """
    Logs the accuracy and loss to wandb
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """
    wandb.log({
        'Training loss': train_loss,
        'Validation loss': val_loss,
        'Training Accuracy': train_acc,
        'Validation Accuracy': val_acc,
        'Training IOU': train_iou,
        'Validation IOU': val_iou
    }, step=epoch)


def wandb_save_summary(train_mean_accuracy: float, 
                       train_mean_iou: float,
                       train_loss: float,
                       valid_mean_accuracy: float,
                       valid_mean_iou: float,
                       valid_loss: float):
   
   
    """[summary]
    Args:

    """

    wandb.run.summary["Train mean_accuracy"] = train_mean_accuracy
    wandb.run.summary["Train mean_iou"] = train_mean_iou
    wandb.run.summary["Train loss"] = train_loss

    wandb.run.summary["Valid mean_accuracy"] = valid_mean_accuracy
    wandb.run.summary["Valid mean_iou"] = valid_mean_iou
    wandb.run.summary["Valid loss"] = valid_loss


# def wandb_log_conf_matrix(y_true: list, y_pred: list):
#     """
#     Logs the confusion matrix
#     Args:
#         y_true (list): ground truth labels
#         y_pred (list): predicted labels
#     """
#     num_classes = len(set(y_true))
#     wandb.log({'confusion_matrix': wandb.plots.HeatMap(list(np.arange(0, num_classes)), list(
#         np.arange(0, num_classes)), confusion_matrix(y_true, y_pred, normalize="true"), show_text=True)})


def save_model_wandb(save_path):
    """ 
    Saves model to wandb
    Args:
        save_path (str): Path to save the wandb model
    """

    wandb.save(os.path.abspath(save_path))