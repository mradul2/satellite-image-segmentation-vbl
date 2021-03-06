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

    wandb.init(
        config = config,
        project=config.wandb_project,
        dir="./",
    )

def wandb_log(train_loss, val_loss, train_acc, val_acc, train_iou, val_iou, epoch):
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
        'Loss/Training': train_loss,
        'Loss/Validation': val_loss,
        'MeanIoU/Training': train_iou[0:1].mean(),
        'MeanIoU/Validation': val_iou[0:1].mean(),
        'MeanAccuracy/Training': train_acc[0:1].mean(),
        'MeanAccuracy/Validation': val_acc[0:1].mean(),
    }, step=epoch)

    classes = ['un-classified', 'no-damage']

    for num in range(len(classes)):
        wandb.log({
            'Accuracy/Training/'+classes[num]: train_acc[num],
            'IoU/Training/'+classes[num]: train_iou[num],
            'Accuracy/Validation/'+classes[num]: val_acc[num],
            'IoU/Validation/'+classes[num]: val_iou[num],
        }, step=epoch)



def wandb_save_summary(valid_mean_accuracy: float,
                       valid_mean_iou: float,
                       valid_loss: float,
                       valid_output,
                       valid_X,
                       valid_y):
   
   
    """[summary]
    Args:

    """
    wandb.run.summary["Valid mean_accuracy"] = valid_mean_accuracy
    wandb.run.summary["Valid mean_iou"] = valid_mean_iou
    wandb.run.summary["Valid loss"] = valid_loss

    class_labels = {
        0: 'un-classified',
        1: 'no-damage',
    }

    for i in range(len(valid_output)):
        mask_img = wandb.Image(valid_X[i], masks={
        "predictions": {
            "mask_data": valid_output[i],
            "class_labels": class_labels
        },
        "ground_truth": {
            "mask_data": valid_y[i],
            "class_labels": class_labels
        }
        })
        wandb.log({"Validation Examples": mask_img})

    wandb.finish()


def wandb_log_conf_matrix(y_true: list, y_pred: list):
    """
    Logs the confusion matrix
    Args:
        y_true (list): ground truth labels
        y_pred (list): predicted labels
    """
    num_classes = 2
    wandb.log({'confusion_matrix': wandb.plots.HeatMap(list(np.arange(0, num_classes)), list(
        np.arange(0, num_classes)), confusion_matrix(y_true, y_pred, normalize="true"), show_text=True)})


def save_model_wandb(save_path):
    """ 
    Saves model to wandb
    Args:
        save_path (str): Path to save the wandb model
    """

    wandb.save(os.path.abspath(save_path))