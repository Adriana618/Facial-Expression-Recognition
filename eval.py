import torch
import collections
import glob
import os
import re

from models.base import *
import settings
from datasets.fer2013 import *
from lossfunc import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
import torch.nn.functional as F


class Eval:
    def __init__(
        self, batch_size=64, input_dim=1, hid_dim=64, c_num=7,
    ):

        self.net = settings.model(input_dim, hid_dim, c_num)
        self.net.to(settings.DEVICE)

        self.ep_eval_str = "test_acc={:.3f}"
        *rest, test_loader, classes = settings.get_dataset(batch_size)

        self.test_loader = test_loader
        self.classes = classes
        self.checkpoints_path = settings.checkpoints_path

    def step(self):
        chs = glob.glob(os.path.join(self.checkpoints_path, "*.pth"))

        assert len(chs) != 0

        eps = [int(re.search("model-(.+?).pth", ch).group(1)) for ch in chs]
        self.load(sorted(eps)[-1])
        self.eval_step()

    def eval_step(self):
        self.net.eval()

        loss_it = []
        acc_it = []

        y_pred = []
        y_true = []

        for batch in self.test_loader:
            batch_data, batch_labels = batch

            outputs = self.net(batch_data.to(settings.DEVICE))

            accuracy = self.accuracy(outputs, batch_labels.to(settings.DEVICE))

            if settings.confussion_matrix:
                y_pred.extend((torch.max(outputs, dim=1)[1]).cpu().numpy())
                y_true.extend(batch_labels)

            acc_it.append(accuracy.item())
        if settings.confussion_matrix:
            cf_matrix = confusion_matrix(y_true, y_pred)
            # print(cf_matrix)
            df_cm = pd.DataFrame(
                cf_matrix / np.sum(cf_matrix) * 100,
                index=[i for _, i in self.classes.items()],
                columns=[i for _, i in self.classes.items()],
            )
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig("confussion_matrix_{}-{}.png".format(settings.model_name, settings.dataset))
        print(self.ep_eval_str.format(np.mean(acc_it)))

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def load(self, ep):
        print("Model Loaded at epoch ...{}".format(ep), end="")
        file_model = "model-{:d}.pth".format(ep)

        load_path = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.net.load_state_dict(checkpoint["model_sd"])


if __name__ == "__main__":
    eval = Eval()
    eval.step()
