import torch
import collections
import glob
import os
import re
import time

from models import *
from tools import *
from lossfunc import *

from torchvision.utils import make_grid
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchsummary import summary

class Trainer():
    def __init__(
        self, 
        batch_size = 128, 
        epochs     = 200, 
        lr         = 1e-3,
        w_decay    = 1e-4, 
        input_dim  = 1,
        hid_dim    = 64, 
        c_num      = 7):
        """
        Params:
            batch_size : number of images per batch
            epochs     : number of epochs
            lr         : learning rate
            inpit_dim  : number of channels of input images
            c_num      : number of classes
        """

        self.net = ResNet9(input_dim, hid_dim, c_num)

        summary(self.net, (1, 48, 48))

        self.net.to(DEVICE)
        #self.net.apply(init_weights)

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=w_decay)

        self.epochs = epochs
        self.ep_train_str = "epoch {:4d}: train_loss={:.3f}"
        self.ep_valid_str = ", valid_loss={:.3f}, valid_acc={:.3f} in {:.1f}s"
        train_loader, valid_loader, *rest = get_FEDdataset(batch_size)

        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, lr, epochs=epochs, steps_per_epoch=len(train_loader))

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.save_step = 10
        self.checkpoints_path = './models'
        """
        self.show_batch()
        """
    def show_batch(self):
        """
        Show batch of images in grid
        """
        for images, labels in self.train_loader:
            images = np.clip(images * 0.5 + 0.5, 0, 1)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1,2,0))
            plt.show()
            break      # comments to show every batch
    
    def train(self):
        """
        train the model
            - First, we detect if there exist checkpoints
            - If exists, we load the last checkpoint
            - Finally, we train the model
        """

        # Verify and load (if it exists) an available checkpoint
        chs = glob.glob(os.path.join(self.checkpoints_path, '*.pth'))
        if len(chs) == 0: ep_start = 1
        else:
            eps = [int(re.search('model-(.+?).pth', ch).group(1)) for ch in chs]
            ep_start = sorted(eps)[-1]
            self.load(ep_start)
            ep_start += 1

        # Training
        for ep in range(ep_start, self.epochs + 1):
            t_start = time.time()
            loss_ep = self.train_step(ep)
            print(self.ep_train_str.format(ep, loss_ep), end="")

            self.schedule.step()
            loss_ep_valid, acc_ep_valid = self.valid_step()
            t_end = time.time()
            print(self.ep_valid_str.format(loss_ep_valid, acc_ep_valid, (t_end-t_start)))

            # Saving every "self.save_step"
            if ep % self.save_step == 0:
                self.save(ep)

    def train_step(self, ep):
        """
        Train step: train the model with the whole training data

        Params:
            ep: epoch

        Return:
            the mean of the loss in every batch of the training data
        """
        self.net.train()

        loss_list = []
        for batch_num, batch in enumerate(self.train_loader, 0):
            batch_data, batch_labels = batch
            batch_labels_1hot = F.one_hot(batch_labels, num_classes=7)

            self.optimize_step(batch_data.to(DEVICE), batch_labels.to(DEVICE))
            loss_list.append(self.loss.item())

        return np.mean(loss_list)
        
    
    def valid_step(self):
        """
        Valid step: train the model with the whole valid data

        Returns:
            the mean of the loss and accuracy in every batch of the training data
        """
        self.net.eval()

        loss_it = []
        acc_it  = []

        for batch in self.valid_loader:
            batch_data, batch_labels = batch
            
            outputs = self.net(batch_data.to(DEVICE))
            loss = self.compute_loss(outputs, batch_labels.to(DEVICE))
            acc  = self.compute_accuracy(outputs, batch_labels.to(DEVICE))

            loss_it.append(loss.item())
            acc_it.append(acc.item())
        
        return np.mean(loss_it), np.mean(acc_it)

    def optimize_step(self, inputs, labels):
        """
        Optimization

        Params:
            inputs: a tensor that contains the batch of images
            labels: a tensor that contains the batch of labels
        """
        outputs = self.net(inputs)
        self.loss = self.compute_loss(outputs, labels)
        
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

    def compute_loss(self, outputs, labels):
        """
        Optimization

        Params:
            outputs: a tensor that contains the batch of logits (model's output)
            labels : a tensor that contains the batch of labels
        """
        loss = F.cross_entropy(outputs, labels)
        #loss = cross_entropy(outputs, labels)
        
        return loss

    def compute_accuracy(self, outputs, labels):
        """
        Compute the accuracy

        Params:
            outputs: a tensor that contains the batch of logits (model's output)
            labels : a tensor that contains the batch of labels
        """
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds==labels).item()/len(preds))

    def save(self, ep):
        """
        Save the model

        Params:
            ep: epoch
        """

        print('Model saved at epoch {:d}... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['model_sd'] = self.net.state_dict()   
        checkpoint['optimizer_sd']  = self.opt.state_dict()

        torch.save(checkpoint, save_path)
        print("Done.")

    def load(self, ep):
        """
        Load a checkpoint to the model

        Params:
            ep: epoch
        """
        print('Model Loaded at epoch ...'.format(ep), end='')
        file_model = 'model-{:d}.pth'.format(ep)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.net.load_state_dict(checkpoint['model_sd'])
        self.opt.load_state_dict(checkpoint['optimizer_sd'])

        print("Done.")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()