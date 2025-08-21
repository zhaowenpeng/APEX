import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from kornia.losses import BinaryFocalLossWithLogits
from kornia.filters import sobel
from torch.autograd import Function
import kornia.filters as kf
import cv2
import math
try:
    import sys
    sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
    from bilateralfilter import bilateralfilter, bilateralfilter_batch
except ImportError:
    pass
from kornia.losses import ssim_loss
from monai.losses import DiceLoss

import segmentation_models_pytorch as smp
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, frac=0.5, threshold=0.5, eps=1e-10):
        super().__init__()
        self.frac = frac
        self.threshold = threshold
        self.eps = eps

    def forward(self, output, target):
        loss_bce = nn.BCELoss()(output, target)
        batch = output.shape[0]
        output = output.reshape(batch, -1)
        target = target.reshape(batch, -1)
        output_t = torch.where(output > self.threshold, torch.ones_like(output), torch.zeros_like(output))
        inter = (output_t * target).sum(-1)
        union = (output_t + target).sum(-1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        loss_dice = 1 - (dice.sum() / batch)
        return self.frac * loss_bce + loss_dice

class CustomDiceCELoss(nn.Module):
    def __init__(self, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = smp.losses.DiceLoss(
            mode='binary',
            ignore_index=ignore_index,
            from_logits=True
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        
    def forward(self, y_pred, y_true):
        valid_mask = (y_true != self.ignore_index).float()
        
        y_true_valid = y_true.clone()
        y_true_valid[y_true == self.ignore_index] = 0
        
        ce_loss = self.ce(y_pred, y_true_valid) * valid_mask
        ce_loss = ce_loss.sum() / (valid_mask.sum() + 1e-8)
        
        dice_loss = self.dice(y_pred, y_true)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

class BCE_SSIM_DICE(nn.Module):
    def __init__(self, size_average=True, issigmoid=False, 
                 dice_weight=1.0, ssim_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.ssim_weight = ssim_weight
        self.dice_weight = dice_weight
        self.size_average = size_average
        self.issigmoid = issigmoid
        
        self.bce = nn.BCELoss(reduction='mean' if size_average else 'none')
        
        self.dice = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=False,
            squared_pred=False,
            jaccard=False,
            reduction='mean' if size_average else 'none'
        )
        
    def forward(self, pmask, rmask):
        if pmask.dim() == 3:
            pmask = pmask.unsqueeze(1)
        if rmask.dim() == 3:
            rmask = rmask.unsqueeze(1)
            
        if self.issigmoid:
            pmask_prob = torch.sigmoid(pmask)
        else:
            pmask_prob = pmask
            
        loss_bce = self.bce(pmask_prob, rmask)
        if not self.size_average and loss_bce.dim() > 0:
            loss_bce = torch.mean(loss_bce, dim=[1,2,3])
            
        loss_ssim = ssim_loss(pmask_prob, rmask, window_size=11, 
                                  reduction='mean' if self.size_average else 'none')
            
        loss_dice = self.dice(pmask_prob, rmask)
        
        total_loss = (self.bce_weight * loss_bce + 
                      self.ssim_weight * loss_ssim + 
                      self.dice_weight * loss_dice)
        
        return total_loss

class BinaryNCE_MAE_Loss(torch.nn.Module):
    def __init__(self, nce_scale=100.0, mae_scale=1.0):
        super(BinaryNCE_MAE_Loss, self).__init__()
        self.nce_scale = nce_scale
        self.mae_scale = mae_scale
        
    def forward(self, pred, target):
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
            
        pred_probs = torch.sigmoid(pred)
        
        mae = torch.abs(pred_probs - target).mean()
        
        pred_log_probs = F.logsigmoid(pred)
        pred_log_neg_probs = F.logsigmoid(-pred)
        
        pos_targets = target
        neg_targets = 1.0 - target
        
        numerator = -(pos_targets * pred_log_probs + neg_targets * pred_log_neg_probs)
        
        denominator = -(pred_log_probs + pred_log_neg_probs)
        denominator = torch.clamp(denominator, min=1e-8)
        
        nce = (numerator / denominator).mean()
        
        combined_loss = self.nce_scale * nce + self.mae_scale * mae
        
        return combined_loss

class BinarySCELoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(BinarySCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        labels = labels.float()
        
        ce = self.cross_entropy(pred, labels)

        pred_probs = torch.sigmoid(pred)
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1.0)
        
        labels_smooth = torch.clamp(labels, min=1e-4, max=1.0)
        inv_labels_smooth = torch.clamp(1 - labels, min=1e-4, max=1.0)
        
        rce = -(pred_probs * torch.log(labels_smooth) + 
                (1 - pred_probs) * torch.log(inv_labels_smooth))
        
        if rce.dim() > 1:
            rce = rce.mean(dim=list(range(1, rce.dim())))

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class EdgeTanimotoLoss(nn.Module):
    def __init__(self, smooth=1e-4, p=2, reduction='mean'):
        super(EdgeTanimotoLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                     [-2, 0, 2], 
                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                     [0, 0, 0], 
                                     [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def detect_edges(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        device = x.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        
        edge = torch.sqrt(gx**2 + gy**2)
        
        edge = edge / (torch.max(edge) + 1e-8)
        
        return edge
    
    def forward(self, predict, target):
        predict_probs = torch.sigmoid(predict)
        
        original_shape = predict.shape
        if len(original_shape) == 2:
            predict_probs = predict_probs.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            predict_probs = predict_probs.unsqueeze(1)
            target = target.unsqueeze(1)
        
        predict_edges = self.detect_edges(predict_probs)
        target_edges = self.detect_edges(target.float())
        
        predict_edges = predict_edges.view(predict_edges.shape[0], -1)
        target_edges = target_edges.view(target_edges.shape[0], -1)
        
        num = torch.sum(torch.mul(predict_edges, target_edges), dim=1) + self.smooth
        den = torch.sum(predict_edges.pow(self.p) + target_edges.pow(self.p), dim=1) + self.smooth

        num_1 = torch.sum(torch.mul(1-predict_edges, 1-target_edges), dim=1) + self.smooth
        den_1 = torch.sum((1-predict_edges).pow(self.p) + (1-target_edges).pow(self.p), dim=1) + self.smooth

        loss = 1 - 0.5*num_1/(den_1-num_1+ self.smooth) - 0.5*num/(den-num+ self.smooth)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception(f'Unexpected reduction method {self.reduction}')

class JSD_Mult_SmoothLoss(nn.Module):
    def __init__(self, smoothing=0.1, threshold=0.8, 
                 consistency_weight=1.0, ce_weight=1.0, ignore_index=255):
        super(JSD_Mult_SmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.threshold = threshold
        self.consistency_weight = consistency_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        
        self.default_weight = torch.ones(3)
        
    def forward(self, labels, pred1, pred2, pred3, weight=None, apply_mask=True):
        device = labels.device
        if weight is None:
            weight = self.default_weight.to(device)
        else:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            weight = weight.to(device)
            
        if labels.dim() == 4 and labels.size(1) == 1:
            labels = labels.squeeze(1)
            
        mask_valid = (labels != self.ignore_index).float()
        
        weight_softmax = F.softmax(weight, dim=0)
        
        pred1_binary = pred1.squeeze(1)
        pred2_binary = pred2.squeeze(1)
        pred3_binary = pred3.squeeze(1)
        
        if self.smoothing > 0:
            smooth_target = labels.float() * (1 - self.smoothing) + (1 - labels.float()) * self.smoothing
            loss1 = F.binary_cross_entropy_with_logits(
                pred1_binary * weight_softmax[0], smooth_target, reduction='none')
            loss2 = F.binary_cross_entropy_with_logits(
                pred2_binary * weight_softmax[1], smooth_target, reduction='none')
            loss3 = F.binary_cross_entropy_with_logits(
                pred3_binary * weight_softmax[2], smooth_target, reduction='none')
        else:
            loss1 = F.binary_cross_entropy_with_logits(
                pred1_binary * weight_softmax[0], labels.float(), reduction='none')
            loss2 = F.binary_cross_entropy_with_logits(
                pred2_binary * weight_softmax[1], labels.float(), reduction='none')
            loss3 = F.binary_cross_entropy_with_logits(
                pred3_binary * weight_softmax[2], labels.float(), reduction='none')
        
        ce_loss = (loss1 + loss2 + loss3) * mask_valid if apply_mask else (loss1 + loss2 + loss3)
        ce_loss = ce_loss.mean()
        
        probs = [torch.sigmoid(pred) for pred in [pred1_binary, pred2_binary, pred3_binary]]
        probs = [torch.clamp(p, 1e-6, 1 - 1e-6) for p in probs]
        probs_expanded = []
        for prob in probs:
            pos_prob = prob.unsqueeze(1)
            neg_prob = (1 - prob).unsqueeze(1)
            probs_expanded.append(torch.cat([neg_prob, pos_prob], dim=1))
        
        weighted_probs = [w * p for w, p in zip(weight_softmax, probs_expanded)]
        mixture_label = torch.stack(weighted_probs).sum(dim=0)
        mixture_label = torch.clamp(mixture_label, 1e-6, 1-1e-6)
        
        if self.threshold is not None:
            try:
                max_probs = torch.amax(
                    mixture_label * mask_valid.unsqueeze(1) if apply_mask else mixture_label, 
                    dim=(-3, -2, -1), keepdim=True
                )
            except AttributeError:
                _, max_probs = torch.max(
                    mixture_label * mask_valid.unsqueeze(1) if apply_mask else mixture_label, 
                    dim=-3, keepdim=True
                )
                _, max_probs = torch.max(max_probs, dim=-2, keepdim=True)
                _, max_probs = torch.max(max_probs, dim=-1, keepdim=True)
                
            confidence_mask = max_probs.ge(self.threshold).float()
        else:
            confidence_mask = torch.ones(mixture_label.shape[0], 1, 1, 1).to(device)
        
        logp_mixture = mixture_label.log()
        consistency_terms = []
        
        for prob in probs_expanded:
            kl_div = F.kl_div(logp_mixture, prob, reduction='none')
            masked_kl = kl_div * confidence_mask
            consistency_term = torch.sum(masked_kl, dim=1)
            if apply_mask:
                consistency_term = consistency_term * mask_valid
            consistency_terms.append(consistency_term)
        
        consistency_loss = torch.mean(sum(consistency_terms))
        
        total_loss = self.ce_weight * ce_loss + self.consistency_weight * consistency_loss
        
        return total_loss, ce_loss, consistency_loss, mixture_label

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _iou(pred, target, size_average = True):
    Iand1 = torch.sum(target * pred, dim=[1,2,3])
    Ior1 = torch.sum(target, dim=[1,2,3]) + torch.sum(pred, dim=[1,2,3]) - Iand1
    IoU = 1- (Iand1 / (Ior1 + 1e-8))

    if size_average==True:
        IoU = IoU.mean()
    return IoU

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)

class BCE_SSIM_IOU(nn.Module):
    def __init__(self, size_average=True, issigmoid=False):
        super().__init__()
        self.ce = torch.nn.BCELoss(reduction='mean' if size_average else 'none')
        self.ssim = SSIM(window_size=11, size_average=size_average)
        self.iou = IOU(size_average=size_average)
        self.size_average = size_average
        self.issigmoid = issigmoid
    def forward(self, pmask, rmask):
        if self.issigmoid:
            pmask=torch.sigmoid(pmask)
        loss_ce = self.ce(pmask, rmask)
        if not self.size_average:
            loss_ce = torch.mean(loss_ce, dim=[1,2,3])
        loss_ssim = 1-self.ssim(pmask, rmask)
        loss_iou = self.iou(pmask, rmask)
        loss = loss_ce + loss_ssim + loss_iou
        return loss

class SoftBootstrappingLoss(nn.Module):
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

class HardBootstrappingLoss(nn.Module):
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

class BinaryDynamicSoftBootstrappingLoss(nn.Module):
    def __init__(self, reduce=True, epsilon=1e-7):
        super().__init__()
        self.reduce = reduce
        self.epsilon = epsilon

    def forward(self, input, target, B):
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        input_const = input.detach()
        bootstrap = -torch.sum(torch.sigmoid(input_const) * F.logsigmoid(input), dim=1)
        
        loss = (1 - B) * bce + B * bootstrap
        if self.reduce:
            return loss.mean()
        return loss

class MixupBinaryDynamicSoftBootstrappingLoss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, y_a, y_b, B, lam, index, output_x1, output_x2):
        bce_a = F.binary_cross_entropy_with_logits(pred, y_a, reduction='none')
        bce_b = F.binary_cross_entropy_with_logits(pred, y_b, reduction='none')
        
        bce_a = bce_a.mean(dim=[1,2,3])
        bce_b = bce_b.mean(dim=[1,2,3])
        
        soft_a = -torch.sum(torch.sigmoid(output_x1) * F.logsigmoid(pred), dim=1)
        soft_b = -torch.sum(torch.sigmoid(output_x2) * F.logsigmoid(pred), dim=1)
        
        loss = lam * ((1-B) * bce_a + B * soft_a) + \
               (1-lam) * ((1-B[index]) * bce_b + B[index] * soft_b)
        
        return loss.mean()

class DenseEnergyLossFunction(Function):
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.cuda()

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear', align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class CEandEnergy(torch.nn.Module):
    def __init__(self, ignore_index=255, densecrfloss=1e-7, sigma_rgb=15.0,
                 sigma_xy = 100, rloss_scale=0.5):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
        self.energy = DenseEnergyLoss(weight=densecrfloss, sigma_rgb=sigma_rgb,
                                      sigma_xy=sigma_xy, scale_factor=rloss_scale)
    def forward(self, ori_img, pred, seg_label, croppings):
        seg_label = seg_label.unsqueeze(1)
        pred_probs = torch.softmax(pred, dim=1)
        ori_img = ori_img.float()
        croppings = croppings.float()

        dloss = self.energy(ori_img, pred_probs, croppings, seg_label)
        dloss = dloss.cuda()

        celoss = self.ce(pred, seg_label.squeeze().long().cuda())

        return dloss+celoss

class DenseEnergyLossFunction_t(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.cuda()

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch_opencv(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None

class DenseEnergyLoss_t(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss_t, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear', align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction_t.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )

class CEandEnergy_t(torch.nn.Module):
    def __init__(self, ignore_index=255, densecrfloss=1e-7, sigma_rgb=15.0,
                 sigma_xy = 100, rloss_scale=0.5):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
        self.energy = DenseEnergyLoss_t(weight=densecrfloss, sigma_rgb=sigma_rgb,
                                        sigma_xy=sigma_xy, scale_factor=rloss_scale)
    def forward(self, ori_img, pred, seg_label, croppings):
        seg_label = seg_label.unsqueeze(1)
        pred_probs = torch.softmax(pred, dim=1)
        ori_img = ori_img.float()
        croppings = croppings.float()

        dloss = self.energy(ori_img, pred_probs, croppings, seg_label)
        dloss = dloss.cuda()

        celoss = self.ce(pred, seg_label.squeeze().long().cuda())

        return dloss+celoss

def bilateralfilter_batch_opencv(images, segmentations, output, N, K, H, W, sigma_rgb, sigma_xy):
    try:
        from cv2 import ximgproc
    except ImportError:
        raise ImportError("OpenCV extended module required. Install opencv-contrib-python package.")
    
    images = np.reshape(images, (N, H, W, -1))
    segmentations = np.reshape(segmentations, (N, K, H, W))
    
    filtered_result = np.zeros_like(segmentations)
    for n in range(N):
        guide = images[n, :, :, :3] if images.shape[3] >= 3 else images[n]
        guide = guide.astype(np.float32)
        
        for k in range(K):
            seg = segmentations[n, k].astype(np.float32)
            if np.max(guide) > 1.0:
                guide_normalized = guide / 255.0
            else:
                guide_normalized = guide
                
            filtered = ximgproc.jointBilateralFilter(
                joint=guide_normalized,
                src=seg,
                d=15,
                sigmaColor=sigma_rgb,
                sigmaSpace=sigma_xy
            )
            filtered_result[n, k] = filtered
    
    np.copyto(output, filtered_result.flatten())
    
class GradientDomainTanimotoLoss(nn.Module):
    def __init__(self, smooth=1e-4, p=2, reduction='mean', normalize=True):
        super(GradientDomainTanimotoLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.normalize = normalize
        
        self.sobel_x = torch.tensor([[-1, 0, 1], 
                                     [-2, 0, 2], 
                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], 
                                     [0, 0, 0], 
                                     [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def compute_gradient_magnitude(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        device = x.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        if self.normalize:
            batch_max = torch.max(grad_magnitude.view(grad_magnitude.size(0), -1), dim=1, keepdim=True)[0]
            batch_max = batch_max.unsqueeze(2).unsqueeze(3).expand_as(grad_magnitude)
            grad_magnitude = grad_magnitude / (batch_max + 1e-8)
        
        return grad_magnitude
    
    def forward(self, predict, target):
        predict_probs = torch.sigmoid(predict)
        
        original_shape = predict.shape
        if len(original_shape) == 2:
            predict_probs = predict_probs.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif len(original_shape) == 3:
            predict_probs = predict_probs.unsqueeze(1)
            target = target.unsqueeze(1)
        
        predict_grad = self.compute_gradient_magnitude(predict_probs)
        target_grad = self.compute_gradient_magnitude(target.float())
        
        predict_grad_flat = predict_grad.view(predict_grad.shape[0], -1)
        target_grad_flat = target_grad.view(target_grad.shape[0], -1)
        
        intersection = torch.sum(predict_grad_flat * target_grad_flat, dim=1)
        
        p_norm = torch.sum(predict_grad_flat.pow(self.p), dim=1)
        t_norm = torch.sum(target_grad_flat.pow(self.p), dim=1)
        
        tanimoto_foreground = (intersection + self.smooth) / (p_norm + t_norm - intersection + self.smooth)
        
        predict_grad_inv = 1 - predict_grad_flat
        target_grad_inv = 1 - target_grad_flat
        
        intersection_inv = torch.sum(predict_grad_inv * target_grad_inv, dim=1)
        p_norm_inv = torch.sum(predict_grad_inv.pow(self.p), dim=1)
        t_norm_inv = torch.sum(target_grad_inv.pow(self.p), dim=1)
        
        tanimoto_background = (intersection_inv + self.smooth) / (p_norm_inv + t_norm_inv - intersection_inv + self.smooth)
        
        tanimoto_loss = 1 - 0.5 * (tanimoto_foreground + tanimoto_background)
        
        if self.reduction == 'mean':
            return tanimoto_loss.mean()
        elif self.reduction == 'sum':
            return tanimoto_loss.sum()
        elif self.reduction == 'none':
            return tanimoto_loss
        else:
            raise ValueError(f'Unsupported reduction method: {self.reduction}')