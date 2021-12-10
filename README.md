# Vision Beyond Limits

## Requirements
```bash 
pip install wandb
```
WandB is a experiment tracking tool and is used to log losses and metrics diring training, and can be used with the provided pipeline. 

## Installation
```bash
git clone https://github.com/mradul2/vbl.git
```

## Usage

Download the data provided by the competition at your machine and pass the root of that directory in the config file to use this codebase pipeline with the following instructions:

Experiment settings such as type of model, environmental prefrences and various hyperparameters can be modified in the config file itself and some of the main prefrences can be provided via CLI itself:

```bash
python3 main.py config.json [--mode MODE] [--model MODEL]
        [--wandb WANDB_SETTING] [--weighted WEIGHTED_TRAINING]
        [--cutmix CUTMIX_TRAINING]

```

1. Default config file `config.json` can be used to train/test the model with the defualt settings and various elements can be changed accordingly inside it.
2. There are two different modes for running the pipeline: `train` and `test`. Train mode is used to train the model with the provided settings and test mode can be used to further evaluate the model on the pre-trained weights. 
3. Available model architectures: 'enet' and 'unet'.
4. WandB tracking can be incorporated by passing 'true' in the argument. 
5. Training using class weights can be done by passing 'true' in the argument. 
6. Cutmix augmentation can be included in the training process by passing 'cutmix' argument as 'true'.
7.