import torch
import torch.nn as nn
from ..builder import LOSSES
import torch.nn.functional as F
from .cross_entropy_loss import _expand_onehot_labels
from .utils import weight_reduce_loss

@LOSSES.register_module
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean',loss_weight=1.0):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight=loss_weight

    def forward(self, x, target, weight=None, ignore_index=255):
       
        # print(x.shape) [2,2,256,256] => [N,C,H,W]
        # print(target.shape) [2,256,256] => [N,H,W]

        #Since prediction input (x) does not have same dimensions as target label so converting target label to the desired format for BCE
        if x.dim() != target.dim():
            assert (x.dim() == 2 and target.dim() == 1) or (
                x.dim() == 4 and target.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
            label, weight = _expand_onehot_labels(target, weight, x.shape,
                                                ignore_index) #use imported version from cross_entropy_loss
        
#         print(label.shape) convert label into float because BCE accepts float values
#         my version
#         BCE_loss = F.binary_cross_entropy_with_logits(x, label.float(), reduction='none')
#         pt = torch.exp(-BCE_loss) # prevents nans when probability 0
#         fl = self.alpha * (1-pt)**self.gamma * BCE_loss
#         fl= self.loss_weight * self._reduce(fl) 
#         return fl

        #mmdet version
        pred_sigmoid = x.sigmoid()
        label = label.type_as(x)
        pt = (1 - pred_sigmoid) * label + pred_sigmoid * (1 - label)
        focal_weight = (self.alpha * label + (1 - self.alpha) *
                        (1 - label)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            x, label, reduction='none') * focal_weight
        loss = weight_reduce_loss(loss, weight, 'mean', None)
        fl= self.loss_weight * loss 
        return fl
        

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
