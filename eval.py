import torch
import collections
import glob
import os
import re

from models import *
from tools import *
from lossfunc import *

from torchvision.utils import make_grid
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Eval():
    def __init__(
        self, 
        batch_size = 64, 
        input_dim  = 1,
        hid_dim    = 32, 
        c_num      = 7,
    ):

        self.net = BaseModel(input_dim, hid_dim, c_num)
        self.net.to(DEVICE)

        self.ep_eval_str = "test_acc={:.3f}"
        *rest, test_loader, classes = get_FEDdataset(batch_size)

        self.test_loader = test_loader
        self.checkpoints_path = './models'
    
    def step(self):
        chs = glob.glob(os.path.join(self.checkpoints_path, '*.pth'))

        assert (len(chs) != 0)
        
        eps = [int(re.search('model-(.+?).pth', ch).group(1)) for ch in chs]
        self.load(sorted(eps)[-1])
        self.eval_step()
    
    def eval_step(self):
        self.net.eval()

        loss_it = []
        acc_it  = []

        for batch in self.test_loader:
            batch_data, batch_labels = batch
            
            outputs  = self.net(batch_data.to(DEVICE))
            accuracy = self.accuracy(outputs, batch_labels.to(DEVICE))

            acc_it.append(accuracy.item())
        
        print(self.ep_eval_str.format(np.mean(acc_it)))

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds==labels).item()/len(preds))

    def load(self, ep):
        print('Model Loaded at epoch ...'.format(ep), end='')
        file_model = 'model-{:d}.pth'.format(ep)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.net.load_state_dict(checkpoint['model_sd'])
        print("Done.")

if __name__ == '__main__':
    eval = Eval()
    eval.step()