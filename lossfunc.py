import torch.nn as nn
import torch
import torch.nn.functional as F

def cross_entropy(outputs, targets):
    targets_1hot = F.one_hot(targets, num_classes=7)
    targets_1hot = targets_1hot.type(torch.cuda.FloatTensor)

    log_softmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(-targets_1hot * log_softmax(outputs), dim=1))
    
def mean_absolute_error(outputs, targets):
    targets_1hot = F.one_hot(targets, num_classes=7)
    targets_1hot = targets_1hot.type(torch.cuda.FloatTensor)

    return torch.mean(torch.abs(outputs-targets_1hot))