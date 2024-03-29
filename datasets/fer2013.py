import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

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
    def __init__(self, images, labels, transf):
        self.images = images
        self.labels = labels
        self.transf = transf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        data = [int(m) for m in self.images[i].split(" ")]
        data = np.asarray(data).astype(np.uint8).reshape(48, 48, 1)
        data = self.transf(data)
        label = self.labels[i]

        return (data, label)


def get_dataset(batch_size):
    df = pd.read_csv("dataset/fer2013.csv")

    classes = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }

    df_train = (
        df[df.Usage == "Training"]
        .drop(["Usage"], axis=1)
        .reset_index()
        .drop(["index"], 1)
    )

    df_valid = (
        df[df.Usage == "PublicTest"]
        .drop(["Usage"], axis=1)
        .reset_index()
        .drop(["index"], 1)
    )
    df_test = (
        df[df.Usage == "PrivateTest"]
        .drop(["Usage"], axis=1)
        .reset_index()
        .drop(["index"], 1)
    )

    train_images = df_train.iloc[:, 1]
    train_labels = df_train.iloc[:, 0]
    valid_images = df_valid.iloc[:, 1]
    valid_labels = df_valid.iloc[:, 0]

    test_images = df_test.iloc[:, 1]
    test_labels = df_test.iloc[:, 0]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(48, padding=4, padding_mode="reflect"),
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
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    train_data = FERDataset(train_images, train_labels, train_transform)
    valid_data = FERDataset(valid_images, valid_labels, eval_transform)
    test_data = FERDataset(test_images, test_labels, eval_transform)

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    valid_loader = DataLoader(valid_data, batch_size, num_workers=2, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size, num_workers=2, pin_memory=True)

    return (train_loader, valid_loader, test_loader, classes)
