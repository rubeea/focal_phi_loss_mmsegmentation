import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import reduce_loss
from .cross_entropy_loss import _expand_onehot_labels

def tversky_loss(inputs, targets, alpha, beta, gamma, smooth, use_focal=False, reduction='mean'):
    
    inputs = inputs.sigmoid()
    targets= targets.type_as(inputs)
    
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
   

    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    if use_focal:
        loss = (1. - Tversky) ** gamma #focal tversky loss
    else:
        loss= (1. - Tversky) #tversky_loss
    loss = reduce_loss(loss, reduction=reduction)
    return loss


@LOSSES.register_module()
class TverskyLoss(nn.Module):
    """TverskyLoss.

    Args:
        use_focal (bool, optional): Whether the prediction uses focal version
            of Tversky loss. Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        alpha (float, optional): False positives penalization factor. Defaults to 0.5.
        beta (float, optional): False negatives penalization factor. Defaults to 0.5.
        gamma (float, optional): Gamma modifier for focal loss. Defaults to 1.0.
        smooth (float, optional): Smoothing factor. Defaults to 1.0.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_focal=False,
                 reduction='mean',
                 alpha=0.5,
                 beta=0.5,
                 gamma=1,
                 smooth=1,
                 loss_weight=1.0):
        super(TverskyLoss, self).__init__()
        self.use_focal = use_focal
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self,
                inputs,
                targets,
                weight=None,
                reduction_override=None,
                avg_factor=None,
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
        loss = self.loss_weight * tversky_loss(inputs, label, 
                                               self.alpha, self.beta, self.gamma, self.smooth, self.use_focal,
                                               reduction= reduction 
                                                )
        return loss
