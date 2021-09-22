import numpy as np
import shutil

import torch
from torch.backends import cudnn
from torch.autograd import Variable

from losses.crossentropy import CrossEntropyLoss
from models.enet import ENet 
from dataloader.vbl_loader import VBLDataLoader

from agents.base import BaseAgent

from utils.metrics import IoUAccuracy

from wandB.wandb_utils import init_wandb, wandb_log, wandb_save_summary, save_model_wandb

class VBLAgent(BaseAgent):
    """
    This class will be responsible for handling the whole process of our architecture.
    """

    def __init__(self, config):
        super().__init__(config)

        # define ENet model
        self.model = ENet(self.config)
        # Create an instance from the data loader
        self.dataloader = VBLDataLoader(self.config)
        # Create instance from the loss
        self.loss = CrossEntropyLoss(self.config)
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
                print("WandB Run: ", self.config.experiment)
            
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
            valid_loss, valid_accuracy, valid_iou, valid_output = self.validate()

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

        print("Training Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(train_loss))


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

        valid_output = []

        for batch in self.dataloader.valid_loader:
            
            inputs = batch[0].float().to(self.device)
            labels = batch[1].float().to(self.device).long()

            outputs = self.model(inputs)

            metric = IoUAccuracy(self.config)
            np_output, iou, accu = metric.evaluate(outputs, labels)

            valid_output.append(np_output)            
            valid_accuracy += accu
            valid_iou += iou
            
            loss = self.loss(outputs, labels)
            
            valid_loss += loss.item()

        valid_loss /= len(self.dataloader.valid_loader)
        valid_accuracy /= len(self.dataloader.valid_loader)
        valid_iou /= len(self.dataloader.valid_loader)

        return valid_loss, valid_accuracy, valid_iou, valid_output

        print("Validation Results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(valid_loss))


    def test(self):
        """ test the model using the provided pre trained weights 
            
        """
        self.model.eval()
            
        valid_accuracy = np.zeros((num_classes,), dtype=float)
        valid_iou = np.zeros((num_classes,), dtype=float)

        valid_results = []
            
        for batch in self.dataloader:
            
            inputs = batch[0].float().to(device)
            labels = batch[1].float().to(device).long()

            outputs = self.model(inputs)

            metric = IoUAccuracy(self.config)
            np_outputs, iou, accu = metric.evaluate(outputs, labels)
            
            valid_accuracy += accu
            valid_iou += iou
            valid_results.append(np_outputs)
            
        valid_accuracy /= len(val_DataLoader)
        valid_iou /= len(val_DataLoader)

        valid_results = np.array(val_results)

        print("Accuracy: ", valid_accuracy)
        print("IoU: ", valid_iou)

        return valid_accuracy, valid_iou, val_results


    def final_summary(self):
        self.load_checkpoint(self.config.checkpoint_dir + self.config.bestpoint_file)
        valid_loss, valid_accuracy, valid_iou, valid_output = self.validate()

        valid_X = []
        valid_y = []

        for batch in self.dataloader.valid_loader:
            image = batch[0][0]
            label = batch[1][0]

            valid_X.append(image)
            valid_y.append(label)

        wandb_save_summary(valid_accuracy.mean(),
                           valid_iou.mean(),
                           valid_loss,
                           valid_output,
                           valid_X,
                           valid_y)


    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.dataloader.finalize()

        if self.config.wandb:
            print("Logging final metrics in WandB...")
            self.final_summary()
            print("Saving Model in WandB...")

            save_model_wandb(self.config.checkpoint_dir + self.config.checkpoint_file)
            save_model_wandb(self.config.checkpoint_dir + self.config.bestpoint_file)
