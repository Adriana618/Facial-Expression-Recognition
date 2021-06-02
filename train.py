import torch
import collections

from models import *
from tools import *
from lossfunc import *

from torchvision.utils import make_grid
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(
        self, 
        batch_size = 64, 
        epochs     = 1000, 
        lr         = 5e-3, 
        input_dim  = 1,
        hid_dim    = 32, 
        c_num      = 7,
    ):

        self.net = BaseModel(input_dim, hid_dim, c_num)
        self.net.to(DEVICE)
        self.net.apply(init_weights)

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=lr
        )
        self.epochs = epochs
        self.ep_train_str = "epoch {:4d}:\ttrain_loss={:.3f}"
        self.ep_valid_str = ", valid_loss={:.3f}, valid_acc={:.3f}"
        train_loader, valid_loader, clases = get_FEDdataset(batch_size)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        """
        self.show_batch()
        """
    def show_batch(self):
        """
            Plot a batch from the training set
        """
        for images, labels in self.train_loader:
            #print(torch.mean(images, dim=(2,3)).cpu().numpy())
            images = np.clip(images * 0.5 + 0.5, 0, 1)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=16).permute(1,2,0))
            plt.show()
            break
    
    def train(self):
        for ep in range(1, self.epochs + 1):
            self.train_step(ep)
            self.valid_step()

    def train_step(self, ep):
        self.net.train()

        loss_list = []
        for batch_num, batch in enumerate(self.train_loader, 0):
            batch_data, batch_labels = batch
            batch_labels_1hot = F.one_hot(batch_labels, num_classes=7)

            self.loss_step(batch_data.to(DEVICE), batch_labels.to(DEVICE))
            loss_list.append(self.loss.item())
        print(self.ep_train_str.format(ep, np.mean(loss_list)), end="")
    
    def valid_step(self):
        self.net.eval()

        loss_it = []
        acc_it  = []

        for batch in self.valid_loader:
            batch_data, batch_labels = batch
            
            outputs = self.net(batch_data.to(DEVICE))
            loss = self.compute_loss(outputs, batch_labels.to(DEVICE))
            acc  = self.accuracy(outputs, batch_labels.to(DEVICE))

            loss_it.append(loss.item())
            acc_it.append(acc.item())
        
        print(self.ep_valid_str.format(
            np.mean(loss_it), np.mean(acc_it)))

    def loss_step(self, inputs, labels):
        outputs = self.net(inputs)
        self.loss = self.compute_loss(outputs, labels)
        
        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

    def compute_loss(self, outputs, labels):
        loss = cross_entropy(outputs, labels)
        #loss = F.cross_entropy(outputs, labels)
        return loss

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds==labels).item()/len(preds))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()