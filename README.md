# Vision Beyond Limits

Solution directory contains: 
1. README File 
2. Source Code
3. Report
4. Segmented images of the validation split

Sample Notebook for Usage: [Notebook](https://www.kaggle.com/mradul2/vbl-main)

## Requirements
```bash 
pip install wandb
```
WandB is a experiment tracking tool and is used to log losses and metrics diring training, and can be used with the provided pipeline. 

Perovided solution experiment reports: [WandB](https://wandb.ai/mradul/vision-beyond-limits?workspace=user-mradul)

## Installation
```bash
git clone https://github.com/mradul2/vbl.git
```

## Usage

Download the data provided by the competition at your machine and pass the root of that directory in the config file to use this codebase pipeline with the following instructions:

Experiment settings such as type of model, environmental prefrences and various hyperparameters can be modified in the config file itself and some of the main prefrences can be provided via CLI itself:

Log into WandB using API Key:
```bash
wandb login
```
Run the script:
```bash
python3 main.py config.json [--model MODEL]
        [--wandb WANDB_SETTING] [--weighted WEIGHTED_TRAINING]
        [--cutmix CUTMIX_TRAINING] [--bs TRAIN_BATCH_SIZE]
        [--epoch NUMBER_OF_EPOCHS]

```

1. Default config file `config.json` can be used to train the model with the defualt settings and various elements can be changed accordingly inside it.
2. Available model architectures: 'enet' and 'unet'.
3. WandB tracking can be incorporated by passing 'true' in the argument. 
4. Training using class weights can be done by passing 'true' in the argument. 
5. Cutmix augmentation can be included in the training process by passing 'cutmix' argument as 'true'.
6. Other hyperparameters of the experiments such as learning rate, weight decay, momentum, gamma, number of epochs, batch sizes and splitting parameters can be changed in the config file itself. 

Both the models ENet and UNet were trained on CutMix Augmentation and using Class Weights also and the results are present in the report. 