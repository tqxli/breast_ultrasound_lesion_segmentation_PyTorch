import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, alpha = 0.5):
        self.alpha = alpha
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
        Dice_BCE = (1-self.alpha) * BCE + self.alpha * dice_loss
        
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

class DiceBCE_CE_JointLoss(nn.Module):
    """
    A custom loss for combining segmentation DiceBCE loss with an additional classification Cross Entropy loss.

    Parameters:
        beta: a scalar that controls ratio between the two losses.
    """
    def __init__(self, beta=0.1):
        super(DiceBCE_CE_JointLoss, self).__init__()
        self.beta = beta
    
    def forward(self, inputs, targets, pred_labels, true_labels, classification_class=2):
        classification_criterion = nn.BCELoss()

        idx = true_labels != classification_class

        classification_loss = classification_criterion(pred_labels, true_labels)
        segmentation_loss = DiceBCE_loss(inputs[idx, :], targets[idx, :]) 

        return self.beta * classification_loss + (1-self.beta) * segmentation_loss 


def DiceBCE_loss(inputs, targets):
    return DiceBCELoss().forward(inputs, targets)

def Dice_loss(inputs, targets):
    return DiceLoss().forward(inputs, targets)

def BCE_loss(inputs, targets):
    criterion = nn.BCELoss()
    return criterion(inputs, targets)

def DiceBCE_CE_loss(inputs, targets, pred_labels, true_labels, classification_class=2):
    return DiceBCE_CE_JointLoss().forward(inputs, targets, pred_labels, true_labels, classification_class)
