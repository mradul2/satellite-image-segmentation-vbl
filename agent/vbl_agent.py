import numpy as np
import shutil

import torch
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from models.enet import ENet
from models.unet import UNet
from models.lib import SeNet154_Unet_Double


from dataloader.vbl_loader import VBLDataLoader

from utils.metrics import IoUAccuracy
from utils.wandb import init_wandb, wandb_log, wandb_save_summary, save_model_wandb, wandb_log_conf_matrix

class VBLAgent():
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        self.config = config

        # Choose model from config file:
        if self.config.model == "unet":
            self.model = UNet(self.config)
        elif self.config.model == "enet":
            self.model = ENet(self.config)
        elif self.config.model == "SeNet154_Unet_Double":
            self.model = SeNet154_Unet_Double()
        else: 
            print("Incorrect Model provided!!!")
            exit()

        print("Training Configurations:")
        print(self.config)


        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            torch.cuda.manual_seed_all(self.config.seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)

        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)

        # Create an instance from the data loader
        self.dataloader = VBLDataLoader(self.config)

        # Create loss function according to the option of weighted loss 
        self.weight = []
        for cls in range(self.config.num_classes):
            temp = (self.dataloader.train_wts)[cls]
            temp = 1/np.log(1.02 + temp)
            (self.weight).append(temp)
        self.weight = torch.FloatTensor(self.weight).to(self.device)
        
        if self.config.weighted:
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index, weight = self.weight)
        else:
            # Create instance from the loss
            self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        # Define Scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                gamma=self.config.gamma)
        
        # initialize counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_loss = 100

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Initializing WandB
        if config.wandb:
            print("Initializing WandB Run...")
            try: 
                init_wandb(self.model, self.config)
                print("WandB initialized successfully")
                print("WandB Project: ", self.config.wandb_project)
            
            except:
                raise ValueError('WandB initialization unsuccessfull!')


    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except:
            print("Any pretrained checkpoint not present")


    def run(self):
        """
        This function will the operator
        :return:
        """
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'test':
                print("Testing Function called...")
                self.test()
            else:
                print("Training Function called...")
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch

            train_loss, train_accuracy, train_iou = self.train_one_epoch()            
            valid_loss, valid_accuracy, valid_iou, valid_output, valid_X, valid_y = self.validate()

            wandb_log(train_loss, valid_loss, train_accuracy, valid_accuracy, train_iou, valid_iou, self.current_epoch)

            self.scheduler.step()

            is_best = valid_loss > self.best_valid_loss
            if is_best:
                self.best_valid_mean_iou = valid_mean_iou

            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """

        # Set the model to be in training mode (for batchnorm)
        self.model.train()
        # Initialize your average meters
        train_loss = 0.0
        train_accuracy = np.zeros((self.config.num_classes,), dtype=float)
        train_iou = np.zeros((self.config.num_classes,), dtype=float)

        for batch in self.dataloader.train_loader:

            inputs = batch[0].float().to(self.device)
            labels = batch[1].float().to(self.device).long()

            outputs = self.model(inputs)

            metric = IoUAccuracy(self.config)
            np_output, iou, accu = metric.evaluate(outputs, labels)
            
            train_accuracy += accu
            train_iou += iou

            loss = self.loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(self.dataloader.train_loader)
        train_accuracy /= len(self.dataloader.train_loader)
        train_iou /= len(self.dataloader.train_loader)

        return train_loss, train_accuracy, train_iou

    def validate(self):
        """
        One epoch validation
        :return:
        """
        # set the model in training mode
        self.model.eval()

        valid_loss = 0.0
        valid_accuracy = np.zeros((self.config.num_classes,), dtype=float)
        valid_iou = np.zeros((self.config.num_classes,), dtype=float)

        valid_X = []
        valid_y = []
        valid_results = []

        for batch in self.dataloader.valid_loader:
            
            inputs = batch[0].float().to(self.device)
            labels = batch[1].float().to(self.device).long()

            outputs = self.model(inputs)

            metric = IoUAccuracy(self.config)
            np_output, iou, accu = metric.evaluate(outputs, labels)

            valid_results.append(np_output)            
            valid_accuracy += accu
            valid_iou += iou

            inputs = np.transpose(inputs[0].cpu().detach().numpy(), (2,1,0))
            valid_X.append(inputs)
            valid_y.append(labels[0].cpu().detach().numpy())
            
            loss = self.loss(outputs, labels)
            
            valid_loss += loss.item()

        valid_loss /= len(self.dataloader.valid_loader)
        valid_accuracy /= len(self.dataloader.valid_loader)
        valid_iou /= len(self.dataloader.valid_loader)

        return valid_loss, valid_accuracy, valid_iou, valid_results, valid_X, valid_y



    def final_summary(self):
        
        valid_loss, valid_accuracy, valid_iou, valid_output, valid_X, valid_y = self.validate()

        # wandb_log_conf_matrix(valid_y, valid_output)

        wandb_save_summary(valid_accuracy.mean(),
                           valid_iou.mean(),
                           valid_loss,
                           valid_output,
                           valid_X,
                           valid_y)


    def test(self):
        print("Testing Mode")
        try:
            self.load_checkpoint(self.config.checkpoint_dir + self.config.bestpoint_file)
        except:
            print("Pretrained Model not successfully Loaded")
            return

        print("Logging the final Metrics")
        self.final_summary()
        

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.test()
        self.save_checkpoint()
        self.dataloader.finalize()