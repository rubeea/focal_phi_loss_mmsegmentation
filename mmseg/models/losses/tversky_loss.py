import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss
from .cross_entropy_loss import _expand_onehot_labels

def tversky(inputs, targets, alpha, beta, smooth):
      
        # print(inputs.shape) [2,2,256,256] => [N,C,H,W]
        # print(targets.shape) [2,256,256] => [N,H,W]
        
        if inputs.dim() != targets.dim():
            assert (inputs.dim() == 2 and targets.dim() == 1) or (
                inputs.dim() == 4 and targets.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
            label, weight = _expand_onehot_labels(target, weight, x.shape,
                                                ignore_index) #use imported version from cross_entropy_loss
                                                
        inputs = F.sigmoid(inputs)
        targets= label.type_as(x)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        return Tversky, weight

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
        self.reduction= reduction
        self.loss_weight = loss_weight
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.smooth=smooth

    def forward(self,
                inputs,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        tversky_val, weight= tversky(inputs,targets,self.alpha,self.beta,self.smooth) 
        
        
        if self.use_focal:
          focal_tversky_loss= (1 - tversky_val)**self.gamma
          loss = weight_reduce_loss(focal_tversky_loss, weight, reduction, None)
          loss = self.loss_weight * loss
          
        else:
          tversky_loss= (1-tversky_val)
          loss = weight_reduce_loss(tversky_loss, weight, reduction, None)
          loss = self.loss_weight * loss
        return loss