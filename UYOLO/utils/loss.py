# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou,MPDIoU,mix_iou,inner_iou,BMPDIoU
from utils.torch_utils import is_parallel
from models.yolo import DETECT_NUM
# DETECT_NUM =19 #21  + 3 # 

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()
    
#xutao loss-----------------------------------------------------------------------------------------------------------
class SegmentationLoss(nn.Module):
    def __init__(self, device, loss_weight):
        super(SegmentationLoss, self).__init__()
        # self.overall_weight = torch.tensor(loss_weight).to(device)
        # self.cross_entropy = F.binary_cross_entropy_with_logits(input,target)
        # #self.cross_entropy = nn.BCEWithLogitsLoss(self.overall_weight, reduction='mean')
        # self.a = torch.tensor(0.1).to(device)  # hardcoded

    def forward(self, x, target):
        # 将预测结果进行 sigmoid 激活
       # x = target
        # 将预测结果和目标标签转换为二维形状
        #x = x.argmax(1)
        x = x.view(-1)  # 将预测结果展平为一维向量

        target = target.view(-1)  # 将目标标签展平为一维向量

        # 计算二分类交叉熵损失
        #target = 3 * target
        loss = F.binary_cross_entropy_with_logits(x, target.float())
        #loss = torch.sigmoid(loss) * self.a

        return loss

#---------------------------------------------------------------------------------------------------------------------
    
#dice loss-------------------------------------------------------------------------------------------------------------
class Dice_Loss(nn.Module):
    def __init__(self, device):
        super(Dice_Loss, self).__init__()
        self.device = device
        self.smooth = 1.0

    def forward(self, x, target):
        # 确保输入值在0到1之间
        x = torch.sigmoid(x)

        # 计算Dice系数的分子和分母
        intersection = torch.sum(x * target)
        union = torch.sum(x) + torch.sum(target)

        # 计算Dice系数
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 计算Dice Loss
        loss = 1.0 - dice_coefficient

        return loss

class Dice_Masked_Loss(nn.Module):
    def __init__(self, device, loss_weight):
        super(Dice_Masked_Loss, self).__init__()
        self.overall_weight = torch.tensor(loss_weight).to(device)
        self.dice_loss = Dice_Loss(device)
        self.a = torch.tensor(1).to(device) #hardcode
        self.b = torch.tensor(0.1).to(device)

    def forward(self, x, target):
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        mask = target < 3   # (B, H, W)
        masked_target = target[mask]  # (N,)

        # repeat the channel dim for C times of x's channel dim
        # (B, H, W, C)
        mask_input = mask.unsqueeze(-1).repeat(1, 1, 1, x_perm.shape[-1])
        masked_x = x_perm[mask_input].view(-1, x_perm.shape[-1])  # (N, C)

        dice_loss = self.dice_loss(masked_x, masked_target)

        pred = F.softmax(x_perm, dim=1)
        pred = torch.clamp(pred, min=1e-5, max=1.0)
        label_one_hot = F.one_hot(target, 3).float().to(target.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.a * dice_loss + self.b * rce.mean()

        return loss
#------------------------------------------------------------------------------------------------------------------------

class CEL_Masked_Loss(nn.Module):
    def __init__(self, device, loss_weight):
        super(CEL_Masked_Loss, self).__init__()
        self.overall_weight = torch.tensor(loss_weight).to(device)
        self.cross_entropy = nn.CrossEntropyLoss(
            self.overall_weight, reduction='mean')
        self.a = torch.tensor(1).to(device) #hardcode
        self.b = torch.tensor(0.1).to(device)
    def forward(self, x, target):
        """Masked Cross Entropy Loss

        Args:
            x (torch.Tensor): (B, C, H, W)
            target (torch.Tensor): (B, H, W)

        Returns:
            torch.Tensor: [scalar]
        """
        # x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        # mask = target < 3   # (B, H, W)
        # masked_target = target[mask]  # (N,)

        # # repeat the channel dim for C times of x's channel dim
        # # (B, H, W, C)
        # mask_input = mask.unsqueeze(-1).repeat(1, 1, 1, x_perm.shape[-1])
        # # print(x_perm.size())
        # masked_x = x_perm[mask_input].view(-1, x_perm.shape[-1])  # (N, C)
        # # masked_target = (masked_target>100).long()
        # loss = self.cross_entropy(masked_x, masked_target)

        # return loss
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        mask = target < 3   # (B, H, W)
        masked_target = target[mask]  # (N,)

        # repeat the channel dim for C times of x's channel dim
        # (B, H, W, C)
        mask_input = mask.unsqueeze(-1).repeat(1, 1, 1, x_perm.shape[-1])
        # print(x_perm.size())
        masked_x = x_perm[mask_input].view(-1, x_perm.shape[-1])  # (N, C)
        #masked_target = (masked_target>100).long()
        ce_loss = self.cross_entropy(masked_x, masked_target)
        pred = F.softmax(x_perm, dim=1)
        pred = torch.clamp(pred, min=1e-5, max=1.0)
        label_one_hot = F.one_hot(target, 3).float().to(target.device)#hpy=3
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) #最小设为 1e-4，即 A 取 -4
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        loss = self.a * ce_loss + self.b * rce.mean()
        return loss

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        loss_weight = [1, 1]
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        CELmask = CEL_Masked_Loss(device, [h['seg_weight1'], h['seg_weight2'], h['seg_weight3']])
        #CELmask = SegmentationLoss(device, [h['seg_weight1']])
        #CELmask = Dice_Masked_Loss(device, [h['seg_weight1'], h['seg_weight2'], h['seg_weight3']])
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[24] if is_parallel(model) else model.model[24]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.CELmask, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, CELmask, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, masks, targets, seg_choose=1):  # predictions, targets, model
        '''
        p: precision, railseg
        targets: det_gt, seg_gt 
        '''

        device = targets.device
        lcls, lbox, lobj, lseg= torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p[0], targets)  # targets

        # Losses
        for i, pi in enumerate(p[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, GIoU=True)  # iou(prediction, target)
                #obj_sz = (tobj.size()[2],tobj.size()[3])
                iou = BMPDIoU(pbox.T, tbox[i],x1y1x2y2=False)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        
        rail_p = p[1]
        rail_gt = masks
        lseg += self.CELmask(rail_p, rail_gt)

        # metric = SegmentationMetric(2)
        # nb, _, height, width = targets[1].shape
        # pad_w, pad_h = shapes[0][1][1]
        # pad_w = int(pad_w)
        # pad_h = int(pad_h)
        # _,lane_line_pred=torch.max(p[1], 1)
        # _,lane_line_gt=torch.max(targets[2], 1)
        # lane_line_pred = lane_line_pred[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # lane_line_gt = lane_line_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]
        # metric.reset()
        # metric.addBatch(lane_line_pred.cpu(), lane_line_gt.cpu())
        # IoU = metric.IntersectionOverUnion()
        # liou_ll = 1 - IoU


        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lseg *= (self.hyp['seg'] * seg_choose)
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + lseg) * bs, torch.cat((lbox, lobj, lcls, lseg)).detach()
        #return (lseg) * bs, torch.cat((lseg)).detach()
        #return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # c                                                                                                                                                                                     lass

        return tcls, tbox, indices, anch                                                                                                                
