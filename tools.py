import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

torch.manual_seed(20)

def makemtrx(lst, n=48):
    for i in range(0, 48 * 48, n):
        yield lst[i:i + n]

def showimg(data):
    pixel = [int(i) for i in data[1].split(' ')]
    pixel = np.array(list(makemtrx(pixel)))
    plt.imshow(pixel, cmap='gray')
    plt.xlabel(f'Expression Class: {data[0]}')
    plt.plot()


class FERDataset(Dataset):
    def __init__(self, images, labels, transf):
        self.images = images
        self.labels = labels
        self.transf = transf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        data = [int(m) for m in self.images[i].split(' ')]
        data = np.asarray(data).astype(np.uint8).reshape(48,48,1)
        data = self.transf(data)
        label = self.labels[i]

        return (data, label)

class MyRotationTransform:
    """
        transforms.Compose(
            ...
            MyRotationTransform(angles=[0, 90, 180, 270 ]),
            ...
        )
    """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def get_FEDdataset(batch_size):
    df = pd.read_csv('dataset/fer2013.csv')
        
    classes = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    df_train = pd.concat([
        df[df.Usage == 'Training'],
        df[df.Usage == 'PublicTest']],
        ignore_index = True).drop(['Usage'], axis=1)

    df_test  = df[
        df.Usage == 'PrivateTest'].drop([
                    'Usage'], axis=1).reset_index().drop(['index'], 1)
        
    train_images = df_train.iloc[:,1]
    train_labels = df_train.iloc[:,0]
    test_images = df_test.iloc[:,1]
    test_labels = df_test.iloc[:,0]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), inplace=True)
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
    )

    train_data = FERDataset(train_images, train_labels, train_transform)
    test_data = FERDataset(test_images, test_labels, test_transform)

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_data, batch_size, num_workers=4, pin_memory=True 
    )

    return (train_loader, test_loader, classes)