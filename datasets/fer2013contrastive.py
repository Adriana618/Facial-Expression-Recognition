import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random

torch.manual_seed(20)


def makemtrx(lst, n=48):
    for i in range(0, 48 * 48, n):
        yield lst[i : i + n]


def showimg(data):
    pixel = [int(i) for i in data[1].split(" ")]
    pixel = np.array(list(makemtrx(pixel)))
    plt.imshow(pixel, cmap="gray")
    plt.xlabel(f"Expression Class: {data[0]}")
    plt.plot()


class FERDataset(Dataset):
    def __init__(self, images, labels, transf, index, train=True):
        self.images = images
        self.labels = labels
        self.transf = transf
        self.train = train
        if self.train:
            self.index = index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        anchor_image = [int(m) for m in self.images[i].split(' ')]
        anchor_image = np.asarray(anchor_image).astype(np.uint8).reshape(48,48,1)
        anchor_image = self.transf(anchor_image)
        anchor_label = self.labels[i]
        if self.train:
            positive_list = self.index[self.index!=i][self.labels[self.index!=i]==anchor_label]
            negative_list = self.index[self.index!=i][self.labels[self.index!=i]!=anchor_label]

            positive_item = random.choice(positive_list)
            positive_image = [int(m) for m in self.images[positive_item].split(' ')]
            positive_image = np.asarray(positive_image).astype(np.uint8).reshape(48,48,1)
            positive_image = self.transf(positive_image)
            p_label = self.labels[positive_item]
            
            negative_item = random.choice(negative_list)
            negative_image = [int(m) for m in self.images[negative_item].split(' ')]
            negative_image = np.asarray(negative_image).astype(np.uint8).reshape(48,48,1)
            negative_image = self.transf(negative_image)
            n_label = self.labels[negative_item]

            return (anchor_image, positive_image, negative_image, anchor_label)

        return (anchor_image, anchor_label)


def get_dataset(batch_size):
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

    #df_train = pd.concat([
    #    df[df.Usage == 'Training'],
    #    df[df.Usage == 'PublicTest']],
    #    ignore_index = True).drop(['Usage'], axis=1)

    df_train  = df[
        df.Usage == 'Training'].drop([
                    'Usage'], axis=1).reset_index().drop(['index'], 1)

    df_valid  = df[
        df.Usage == 'PublicTest'].drop([
                    'Usage'], axis=1).reset_index().drop(['index'], 1)
    df_test  = df[
        df.Usage == 'PrivateTest'].drop([
                    'Usage'], axis=1).reset_index().drop(['index'], 1)
        
    train_images = df_train.iloc[:,1]
    train_labels = df_train.iloc[:,0]
    train_index = df_train.index.values
    valid_images = df_valid.iloc[:,1]
    valid_labels = df_valid.iloc[:,0]
    valid_index = df_valid.index.values

    test_images = df_test.iloc[:,1]
    test_labels = df_test.iloc[:,0]
    test_index = df_test.index.values

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), inplace=True),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
    )

    train_data = FERDataset(train_images, train_labels, train_transform, train_index)
    valid_data = FERDataset(valid_images, valid_labels, eval_transform, valid_index, train=False)
    test_data = FERDataset(test_images, test_labels, eval_transform, test_index)

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_data, batch_size, num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_data, batch_size, num_workers=2, pin_memory=True 
    )

    return (train_loader, valid_loader, test_loader, classes)