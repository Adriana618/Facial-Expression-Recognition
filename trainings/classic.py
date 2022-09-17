import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_step(net, ep, train_loader, loss, opt):
    """
    Train step: train the model with the whole training data

    Params:
        ep: epoch

    Return:
        the mean of the loss in every batch of the training data
    """
    net.train()

    loss_list = []
    for batch_num, batch in enumerate(train_loader, 0):
        batch_data, batch_labels = batch
        batch_labels_1hot = F.one_hot(batch_labels, num_classes=7)

        loss = optimize_step(
            batch_data.to(DEVICE), batch_labels.to(DEVICE), net, opt, loss
        )
        loss_list.append(loss.item())

    return np.mean(loss_list)


def valid_step(net, valid_loader):
    """
    Valid step: train the model with the whole valid data

    Returns:
        the mean of the loss and accuracy in every batch of the training data
    """
    net.eval()

    loss_it = []
    acc_it = []

    for batch in valid_loader:
        batch_data, batch_labels = batch

        outputs = net(batch_data.to(DEVICE))
        loss = compute_loss(outputs, batch_labels.to(DEVICE))
        acc = compute_accuracy(outputs, batch_labels.to(DEVICE))

        loss_it.append(loss.item())
        acc_it.append(acc.item())

    return np.mean(loss_it), np.mean(acc_it)


def optimize_step(inputs, labels, net, opt, loss):
    """
    Optimization

    Params:
        inputs: a tensor that contains the batch of images
        labels: a tensor that contains the batch of labels
    """
    outputs = net(inputs)
    loss = compute_loss(outputs, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss


def compute_loss(outputs, labels):
    """
    Optimization

    Params:
        outputs: a tensor that contains the batch of logits (model's output)
        labels : a tensor that contains the batch of labels
    """
    loss = F.cross_entropy(outputs, labels)
    # loss = cross_entropy(outputs, labels)

    return loss


def compute_accuracy(outputs, labels):
    """
    Compute the accuracy

    Params:
        outputs: a tensor that contains the batch of logits (model's output)
        labels : a tensor that contains the batch of labels
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
