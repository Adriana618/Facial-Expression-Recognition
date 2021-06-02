from PIL import Image
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import csv

class MUG(Dataset):
    
    def __init__(self, data_path, transforms=None):
        images,labels = self.getSet(data_path)
        self.X = images
        self.y = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        """Obtiene un par que contiene la imagen y su etiqueta

        Args:
            i (int): indice del elemento a obtener

        Returns:
            Si no se aplica ninguna transformación, la data por defecto es una imagen PIL
        """
        data = Image.open(self.X[i])
        if self.transforms is not None:
            data = self.transforms(data)
        label = int(self.y[i])
        return (data, label)

    def getSet(self, data_path):
        file_path = os.path.join(data_path)
        file = open(file_path,'r')
        lines = file.readlines()
        images = []
        labels = []
        for line in lines:
            tokens = line.split(' ')
            images.append(tokens[0])
            labels.append(tokens[1][0])
        return images,labels