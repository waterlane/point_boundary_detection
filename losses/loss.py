import torch.nn.functional as F

def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    
    # dice
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return bce + dice