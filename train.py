import torch
import collections
import glob
import os
import re
import time

import settings

from models.base import *
from datasets.fer2013 import *
from lossfunc import *

from torchvision.utils import make_grid
import torch.nn.functional as F

DEVICE = settings.DEVICE

from torchsummary import summary


class Trainer:
    def __init__(
        self,
        batch_size=128,
        epochs=200,
        lr=1e-3,
        w_decay=1e-4,
        input_dim=1,
        hid_dim=64,
        c_num=7,
    ):
        """
        Params:
            batch_size : number of images per batch
            epochs     : number of epochs
            lr         : learning rate
            inpit_dim  : number of channels of input images
            c_num      : number of classes
        """

        self.net = settings.model(input_dim, hid_dim, c_num)

        summary(self.net, (1, 48, 48))

        self.net.to(DEVICE)
        # self.net.apply(init_weights)

        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=w_decay)

        self.epochs = settings.epochs
        self.ep_train_str = "epoch {:4d}: train_loss={:.3f}"
        self.ep_valid_str = ", valid_loss={:.3f}, valid_acc={:.3f} in {:.1f}s"
        train_loader, valid_loader, *rest = settings.get_dataset(batch_size)

        self.schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.save_step = settings.save_step
        self.checkpoints_path = settings.checkpoints_path

        self.loss = None

    def train(self):
        """
        train the model
            - First, we detect if there exist checkpoints
            - If exists, we load the last checkpoint
            - Finally, we train the model
        """

        # Verify and load (if it exists) an available checkpoint
        chs = glob.glob(os.path.join(self.checkpoints_path, "*.pth"))
        if len(chs) == 0:
            ep_start = 1
        else:
            eps = [int(re.search("model-(.+?).pth", ch).group(1)) for ch in chs]
            ep_start = sorted(eps)[-1]
            self.load(ep_start)
            ep_start += 1

        train_loss_history = []
        valid_loss_history = []
        valid_accu_history = []
        # Training
        for ep in range(ep_start, self.epochs + 1):
            t_start = time.time()
            # loss_ep = settings.train_step(self.net, ep, self.train_loader, self.loss, self.opt)
            # loss_ep = self.train_step(ep)
            loss_ep = settings.train_step(self, ep)
            train_loss_history.append(loss_ep)
            print(self.ep_train_str.format(ep, loss_ep), end="")

            self.schedule.step()
            # loss_ep_valid, acc_ep_valid = settings.valid_step(self.net, self.valid_loader)
            # loss_ep_valid, acc_ep_valid = self.valid_step()
            loss_ep_valid, acc_ep_valid = settings.valid_step(self)
            valid_loss_history.append(loss_ep_valid)
            valid_accu_history.append(acc_ep_valid)
            t_end = time.time()
            print(
                self.ep_valid_str.format(loss_ep_valid, acc_ep_valid, (t_end - t_start))
            )

            # Saving every "self.save_step"
            if ep % self.save_step == 0:
                self.save(ep)

        plt.plot(train_loss_history)
        plt.plot(valid_loss_history)
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("model_loss.png")
        plt.clf()
        plt.plot(valid_accu_history)
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["val"], loc="upper left")
        plt.savefig("model_accuracy.png")

    def save(self, ep):
        """
        Save the model

        Params:
            ep: epoch
        """

        print("Model saved at epoch {:d}... ".format(ep), end="")

        file_model = "model-{:d}.pth".format(ep)

        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        save_path = os.path.join(self.checkpoints_path, file_model)

        checkpoint = {}

        checkpoint["model_sd"] = self.net.state_dict()
        checkpoint["optimizer_sd"] = self.opt.state_dict()

        torch.save(checkpoint, save_path)

    def load(self, ep):
        """
        Load a checkpoint to the model

        Params:
            ep: epoch
        """
        print("Model Loaded at epoch ...".format(ep), end="")
        file_model = "model-{:d}.pth".format(ep)

        load_path = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.net.load_state_dict(checkpoint["model_sd"])
        self.opt.load_state_dict(checkpoint["optimizer_sd"])

        print("Done.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
