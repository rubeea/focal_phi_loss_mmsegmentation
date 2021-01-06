import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .cross_entropy_loss import _expand_onehot_labels


def tversky_val(inputs, targets, alpha, beta, smooth):
    
    inputs = inputs.sigmoid()
    targets= targets.type_as(inputs)
    # flatten label and prediction tensors
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # print(inputs.shape)
    # print(targets.shape)
    # True Positives, False Positives & False Negatives

    #removing .sum() to keep the tensor dimensions to [2,2,256,256] compatible with weight dimensions
    # to use .sum with TP,FP and FN, weights must also be summed using .sum() in weighted reduction loss
    

    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    
    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return Tversky


@LOSSES.register_module()
class ComboLoss(nn.Module):
    """ComboLoss.

    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        alpha (float, optional): False positives penalization factor. Defaults to 0.5.
        beta (float, optional): False negatives penalization factor. Defaults to 0.5.
        gamma (float, optional): Gamma modifier for focal loss. Defaults to 1.0.
        smooth (float, optional): Smoothing factor. Defaults to 1.0.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 reduction='mean',
                 alpha=0.5,
                 beta=0.5,
                 falpha=1,
                 fgamma=2,
                 smooth=1,
                 loss_weight=1.0):
        super(ComboLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.falpha = falpha
        self.fgamma = fgamma
        self.smooth = smooth

    def forward(self,
                inputs,
                targets,
                weight=None,
                reduction_override=None,
                ignore_index=255):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # print(inputs.shape) #[2,2,256,256] => [N,C,H,W]
        # print(targets.shape) #[2,256,256] => [N,H,W]

        if inputs.dim() != targets.dim():
            assert (inputs.dim() == 2 and targets.dim() == 1) or (
                inputs.dim() == 4 and targets.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
            label, weight = _expand_onehot_labels(targets, weight, inputs.shape,
                                                ignore_index) #use imported version from cross_entropy_loss
        
        
        #calculate focal loss
        
        pred_sigmoid = inputs.sigmoid()
        label = label.type_as(inputs)
        pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
        focal_weight = (self.falpha * label + (1 - self.falpha) *
                          (1 - label)) * pt.pow(self.fgamma)
  
        fl = F.binary_cross_entropy_with_logits(
              inputs, label, reduction='none') * focal_weight

        #fl = weight_reduce_loss(fl, weight, reduction, None)
        print(type(fl))     
        tversky_val = tversky_val(inputs, label, self.alpha, self.beta, self.smooth)
        
        tversky_loss = (1. - tversky_val)
        weight= weight.sum()
        #tversky_loss = weight_reduce_loss(tversky_loss, weight, reduction, None)
        print(type(tversky_loss))
            
            
        combo_loss= 7*fl-(torch.log(tversky_loss))
        loss = self.loss_weight * combo_loss
        return loss
