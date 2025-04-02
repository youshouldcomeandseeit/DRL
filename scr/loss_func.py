import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function,
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions.
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss.

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask, epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1.).cuda(), targets)
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt ** self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        loss = loss[mask]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class AsymmetricLossOptimized(nn.Module):
    '''ASL loss as described in the paper "Asymmetric Loss For Multi-Label Classification"
    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # 分别计算正负例的概率
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # 非对称裁剪
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)  # 给 self.xs_neg 加上 clip 值

        # 先进行基本交叉熵计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # 以下 4 行相当于做了个并行操作
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        self.loss = torch.masked_select(self.loss, mask == 1)

        return -self.loss.mean()


class DRLoss(nn.Module):
    r"""DRLoss is proposed by us in the paper "Dynamically Robust Loss: Unlocking Noisy-Label Learning in Named Entity Recognition "
        """
    def __init__(self,beta=0.,robust=True):
        super(DRLoss, self).__init__()
        self.beta = beta + 1e-7
        self.exp_clamp = -80.
        self.robust = robust

    def forward(self, x, y, mask):
        x = x.to(torch.float64)
        x = x.reshape(x.shape[0],-1,x.shape[-1])
        y = y.reshape(y.shape[0], -1, y.shape[-1])
        mask = mask.reshape(x.shape)
        x = (1 - 2 * y) * x

        if self.robust:
            # x = torch.where(torch.abs(x) > 80.,x, (1. - torch.exp(-self.beta * x)) / self.beta)
            x = ((1. - torch.exp(-self.beta * x.clamp(min=self.exp_clamp))) / self.beta)

        x_neg = (x - (y * 1e12)).masked_fill(mask == 0, -1e12)
        x_pos = (x - ((1 - y) * 1e12)).masked_fill(mask == 0, -1e12)
        plogit_class = torch.logsumexp(x_pos, dim=-2)
        plogit_sample = torch.logsumexp(x_pos, dim=-1)
        nlogit_class = torch.logsumexp(x_neg, dim=-2)
        nlogit_sample = torch.logsumexp(x_neg, dim=-1)

        class_loss = torch.nn.functional.softplus(nlogit_class) + torch.nn.functional.softplus(plogit_class)
        sample_loss = torch.nn.functional.softplus(nlogit_sample) + torch.nn.functional.softplus(plogit_sample)

        return sample_loss.mean(), class_loss.mean()







class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter,
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss.

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2
    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin).clamp(min=1e-8)
        pred_neg = torch.sigmoid(logits).clamp(min=1e-8)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1 - targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        loss = loss[mask]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss





class Symmetric_CELoss(nn.Module):
    ''' Symmetric_CELoss as described in the paper "Symmetric cross entropy for robust learning with noisy labels"
    '''
    def __init__(self, alpha=1, beta=1):
        super(Symmetric_CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, labels):

        bce = self.cross_entropy(logits, labels)

        # RCE
        pred = torch.sigmoid(logits)
        pred = torch.clamp(pred, min=1e-8, max=1.0)
        pos_label = torch.clamp(labels.float().to(pred.device), min=1e-4, max=1.0)
        neg_label = torch.clamp((1- labels).float().to(pred.device), min=1e-4, max=1.0)

        rce = - pred * torch.log(pos_label) - (1 - pred) * torch.log(neg_label)

        loss = self.alpha * bce + self.beta * rce.mean()

        return loss
class BCEFocalLoss(torch.nn.Module):
    '''
            BCEFocalLoss as described in the paper " Focal loss for dense object detection."
        '''
    def __init__(self, gamma=2, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predict, target,mask):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        loss = -  (1 - pt) ** self.gamma * target * torch.log(pt.clamp(min=1e-8)) -  pt ** self.gamma * (1 - target) * torch.log((1 - pt).clamp(min=1e-8))
        loss = torch.masked_select(loss,mask==1)
        loss = loss[loss < 10.] # nan issue
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class BCELossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=0, reduction='mean'):
        super(BCELossWithLabelSmoothing, self).__init__()
        self.alpha = alpha

        self.reduction = reduction

    def forward(self, predict, target,mask):

        label_count = torch.sum(target,dim=-1).unsqueeze(-1).expand(-1,-1,-1,predict.size(dim=-1)) > 0
        true = torch.where(label_count > 0, target * (1. - self.alpha), target)
        # smooth_label = target * (1 - self.alpha) + self.alpha / num_classes
        loss = BCELoss(predict,true,mask)

        return loss

def BCELoss(predict, target, mask=None):
    pred = torch.masked_select(predict, mask == 1)
    true = torch.masked_select(target, mask == 1)
    loss = F.binary_cross_entropy_with_logits(pred, true.float())
    return loss



def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights



class GHMC(nn.Module):
    '''GHMC as described in the paper " Gradient harmonized single-stage detector."
    '''
    def __init__(
            self,
            bins=10,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, pred, target, label_weight, *args, **kwargs):
        """ Args:
        pred [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary class target for each sample.
        label_weight [batch_num, class_num]:
            the value is 1 if the sample is valid and 0 if ignored.
        """
        if not self.use_sigmoid:
            raise NotImplementedError
        # the target should be binary class label
        if pred.dim() != target.dim():
            target, label_weight = _expand_binary_labels(target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid

            num_in_bin = inds.sum().item()

            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight


class Boundary_smoothing(nn.Module):
    '''
    Boundary_smoothing as described in the paper "  Boundary smoothing for named entity recognition."
            '''
    def __init__(self, sb_epsilon=0.1, sb_size=1, reduction='mean'):
        super(Boundary_smoothing, self).__init__()
        self.sb_epsilon = sb_epsilon
        self.sb_size = sb_size
        self.reduction = reduction
        self.sb_adj_factor = 1.
    def _spans_from_surrounding(self,indices,mask):
        b, s, e, l = indices
        sur_spans = []
        length = mask.shape[1]
        if 0 <= e + self.sb_size < length and mask[b, s, e + self.sb_size, l] == 1:
            sur_spans.append((b, s, e + self.sb_size, l))
        if 0 <= e - self.sb_size < length and mask[b, s, e - self.sb_size, l] == 1:
            sur_spans.append((b, s, e - self.sb_size, l))
        if 0 <= s + self.sb_size < length and mask[b, s + self.sb_size, e, l] == 1:
            sur_spans.append((b, s + self.sb_size, e, l))
        if 0 <= s - self.sb_size < length and mask[b, s - self.sb_size, e, l] == 1:
            sur_spans.append((b, s - self.sb_size, e, l))
        return sur_spans

    def forward(self, predict, target, mask):
        boundary2label_id = torch.where(target == 1,target-self.sb_epsilon,target)
        indices = list(zip(torch.where(target == 1)[0].tolist(), torch.where(target == 1)[1].tolist(),
                           torch.where(target == 1)[2].tolist(), torch.where(target == 1)[3].tolist(), ))
        for dist in range(1, self.sb_size + 1):
            eps_per_span = self.sb_epsilon / (self.sb_size * dist * 4)
            for indice in indices:
                sur_spans = self._spans_from_surrounding(indice, mask)
                for b,s,e,l in sur_spans:
                    boundary2label_id[b,s,e,l] += (eps_per_span * self.sb_adj_factor)
                boundary2label_id[indice] += eps_per_span * (dist * 4 - len(sur_spans))

        loss = BCELoss(predict,boundary2label_id,mask)
        return loss






