import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def DiceBCE_loss(inputs, targets):
    return DiceBCELoss().forward(inputs, targets)

def Dice_loss(inputs, targets):
    return DiceLoss().forward(inputs, targets)

def BCE_loss(inputs, targets):
    criterion = nn.BCELoss()
    return criterion(inputs, targets)

def DiceBCE_CE_loss(inputs, targets, pred_labels, true_labels, classification_class=2):
    """
    A custom loss which incorporates both segmentation (DiceBCE) and classification loss (Cross Entropy).
    """
    classification_criterion = nn.CrossEntropyLoss()

    idx = true_labels != classification_class

    classification_loss = classification_criterion(pred_labels, true_labels)
    segmentation_loss = DiceBCELoss().forward(inputs[idx, :], targets[idx, :]) 

    return classification_loss + segmentation_loss