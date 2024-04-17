# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def converter(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().data.numpy().flatten()
    return data.flatten()


def fast_hist(label_pred, label_true, num_classes):
    #pdb.set_trace()
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred,
                       minlength=num_classes**2)
    hist = hist.reshape(num_classes, num_classes)
    return hist


class Metric_mIoU():
    def __init__(self, class_num):
        self.class_num = class_num
        self.hist = np.zeros((self.class_num, self.class_num))

    def update(self, predict, target):
        # target = (target > 150).long()
        predict, target = converter(predict), converter(target)

        self.hist += fast_hist(predict, target, self.class_num)

    def reset(self):
        self.hist = np.zeros((self.class_num, self.class_num))

    def get_miou(self):
        miou = np.diag(self.hist) / (np.sum(self.hist, axis=1) + np.sum(
            self.hist, axis=0) - np.diag(self.hist))
        miou = np.nanmean(miou)
        return miou

    def get_acc(self):
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def get(self, opt="IoU"):
        if opt=="IoU":
            return self.get_miou()
        elif opt == "acc":
            return self.get_acc()
        else:
            assert ModuleNotFoundError

def to_python_float(t):
    if hasattr(t, "item"):
        return t.item()
    else:
        return t[0]


class EvaluationMetrics():
    def __init__(self, class_num, class_id_to_name):
        self.SMOOTH = 1e-6
        self.metrics = {"IoU": self.seg_masked_iou,
                        "Accuracy": self.seg_masked_accuracy}
        self.class_num = class_num
        self.class_id_to_name = class_id_to_name
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape, "Segmentation pred shape must match the target shape."
        return {name: metric(pred, target) for name, metric in self.metrics.items()}

    def summary(self, inputs: dict):
        assert inputs.keys() == self.metrics.keys()
        result = {key : None for key in inputs.keys()}
        for key, value in inputs.items():
            if key == "IoU":
                iou = torch.stack([torch.stack(val) for val in value])               
                iou_mean = iou.mean(0)
                iou_stat = {self.class_id_to_name[i]: to_python_float(
                    iou_mean[i]) for i in range(len(iou_mean))}
                result[key] = iou_stat
            elif key == "Accuracy":
                # Accuracy all classes
                acc = [item[0] for item in value]
                acc = torch.stack([torch.stack(val) for val in acc])
                acc_mean = acc.mean(0)

                # Global accuracy
                global_acc = [item[1] for item in value]
                global_acc = torch.stack(global_acc)
                global_acc_mean = global_acc.mean(0)

                acc_stat = {self.class_id_to_name[i]: to_python_float(
                    acc_mean[i]) for i in range(len(acc_mean))}
                acc_stat["global"] = global_acc_mean
                result[key] = acc_stat
        return result

    def seg_masked_iou(self, pred: torch.Tensor, target: torch.Tensor):
        mask = target >= 0
        masked_pred = pred[mask]
        masked_target = target[mask]

        iou_score = []
        for i in range(self.class_num):
            pred_i = masked_pred == i
            target_i = masked_target == i
            intersection = torch.logical_and(pred_i, target_i).sum()
            union = torch.logical_or(pred_i, target_i).sum()
            iou_score.append(intersection / (union + self.SMOOTH))  # Avoid 0/0
        return iou_score

    def seg_masked_accuracy(self, pred: torch.Tensor, target: torch.Tensor):
        mask = target >= 0
        masked_pred = pred[mask]
        masked_target = target[mask]

        acc_score = []
        for i in range(self.class_num):
            pred_i = masked_pred == i
            target_i = masked_target == i
            acc_score.append((pred_i == target_i).sum() /
                             (pred_i.numel() + self.SMOOTH))

        global_acc_score = 1 - \
            torch.logical_xor(masked_pred, masked_target).sum() / \
            masked_pred.numel()
        return acc_score, global_acc_score

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.3, 0.7, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95, IoU]
    return (x[:, :5] * w).sum(1)


def ap_per_class(tp,
                 conf,
                 pred_cls,
                 target_cls,
                 plot=False,
                 save_dir='.',
                 names=(),
                 eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros(
        (nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0],
                              left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                              left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:,
                                                                           j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec,
                                        mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items()
             if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict\
    
    pr_file = Path(save_dir) / 'PR_curve.csv'
    s = '' if pr_file.exists() else (('%20s,' * len(names) % tuple(names.values())).rstrip(',') +'\n')  # add header
    with open(pr_file, 'a') as f:
        f.write(s)
        for i in zip(px, *py):
            ap_ = sum(i[1:]) / 8.
            f.write(('%20.5g,' * len(list(i) + [ap_]) % tuple(list(i) + [ap_])).rstrip(',') + '\n')
        f.close()
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px,
                      f1,
                      Path(save_dir) / 'F1_curve.png',
                      names,
                      ylabel='F1')
        plot_mc_curve(px,
                      p,
                      Path(save_dir) / 'P_curve.png',
                      names,
                      ylabel='Precision')
        plot_mc_curve(px,
                      r,
                      Path(save_dir) / 'R_curve.png',
                      names,
                      ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(
            mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])
        #iou = MPDIoU(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1],
                                            return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0],
                                            return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (
                (self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1
            )  # normalize columns
            array[array <
                  0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99
                      ) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore'
                )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(
                    array,
                    annot=self.nc < 30,
                    annot_kws={
                        "size": 8
                    },
                    cmap='Blues',
                    fmt='.2f',
                    square=True,
                    xticklabels=names +
                    ['background FP'] if labels else "auto",
                    yticklabels=names +
                    ['background FN'] if labels else "auto").set_facecolor(
                        (1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1,
             box2,
             x1y1x2y2=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2)**
                    2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter
                    )  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter
                    )  # iou = inter / (area1 + area2 - inter)

def MPDIoU(box1,
           box2,
           x1y1x2y2=True,
           eps=1e-7):
    box2 = box2.T
    w,h = 640,640
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        #box1 top-left=(b1_x1,b1_y1) bottom-right=(b1_x2,b1_y2)
        #box1 top-left=(b2_x1,b2_y1) bottom-right=(b2_x2,b2_y2)
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    #交集
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    #并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    squared1 = (b2_x1-b1_x1)*(b2_x1-b1_x1)+(b2_y1-b1_y1)*(b2_y1-b1_y1)
    squared2 = (b2_x2-b1_x2)*(b2_x2-b1_x2)+(b2_y2-b1_y2)*(b2_y2-b1_y2)
    #squared1 = (0.8*(b2_x1-b1_x1))*(0.8*(b2_x1-b1_x1))+(0.8*(b2_y1-b1_y1))*(0.8*(b2_y1-b1_y1))
    #squared2 = (0.8*(b2_x2-b1_x2))*(0.8*(b2_x2-b1_x2))+(0.8*(b2_y2-b1_y2))*(0.8*(b2_y2-b1_y2))
    #squared3 = (b2_x2-b1_x2)*(b2_x2-b1_x2)+(b2_y1-b1_y1)*(b2_y1-b1_y1)
    #squared4 = (b2_x1-b1_x1)*(b2_x1-b1_x1)+(b2_y2-b1_y2)*(b2_y2-b1_y2)
    sqr_imgdiag = w*w + h*h
    mpd_iou = inter/union - squared1/sqr_imgdiag - squared2/sqr_imgdiag
    #- squared3/sqr_imgdiag - squared4/sqr_imgdiag
    return mpd_iou

def BMPDIoU(box1,
           box2,
           x1y1x2y2=True,
           eps=1e-7):
    box2 = box2.T
    w,h = 640,640
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        #box1 top-left=(b1_x1,b1_y1) bottom-right=(b1_x2,b1_y2)
        #box1 top-left=(b2_x1,b2_y1) bottom-right=(b2_x2,b2_y2)
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    #交集
    inter = (1.2*(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)) * \
            (1.2*(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0))
    #并集
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    w1, h1 = 1.2*w1,1.2*h1
    w2, h2 = 1.2*w2,1.2*h2
    union = w1 * h1 + w2 * h2 - inter + eps
    #squared1 = (b2_x1-b1_x1)*(b2_x1-b1_x1)+(b2_y1-b1_y1)*(b2_y1-b1_y1)
    #squared2 = (b2_x2-b1_x2)*(b2_x2-b1_x2)+(b2_y2-b1_y2)*(b2_y2-b1_y2)
    squared1 = (1.2*(b2_x1-b1_x1))*(1.2*(b2_x1-b1_x1))+(1.2*(b2_y1-b1_y1))*(1.2*(b2_y1-b1_y1))
    squared2 = (1.2*(b2_x2-b1_x2))*(1.2*(b2_x2-b1_x2))+(1.2*(b2_y2-b1_y2))*(1.2*(b2_y2-b1_y2))
    #squared3 = (b2_x2-b1_x2)*(b2_x2-b1_x2)+(b2_y1-b1_y1)*(b2_y1-b1_y1)
    #squared4 = (b2_x1-b1_x1)*(b2_x1-b1_x1)+(b2_y2-b1_y2)*(b2_y2-b1_y2)
    sqr_imgdiag = w*w + h*h
    bmpd_iou = inter/union - squared1/sqr_imgdiag - squared2/sqr_imgdiag
    #- squared3/sqr_imgdiag - squared4/sqr_imgdiag
    return bmpd_iou

def mix_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    bbox_iou_value = bbox_iou(box1,box2, x1y1x2y2=False, GIoU=True)  # 计算bbox_iou
    mpd_iou_value = MPDIoU(box1, box2, x1y1x2y2=True)  # 计算mpd_iou
    bmpd_iou_value = BMPDIoU(box1, box2, x1y1x2y2=True)

    mix_iou_value = 0.6 * bbox_iou_value + 0.4 * mpd_iou_value 
    #+ 0.2*bmpd_iou_value

    return mix_iou_value

#inner-iou+mpd------------------------------------------------------------------------------------
def inner_iou(box1,
             box2,
             feat_sz,
             xywh=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             SIoU=False,
             EIoU=False,
             WIoU=False,
             MPDIoU=False,
             alpha=1,
             scale=False,
             monotonous=False,
             ratio=1.0,
             eps=1e-7):
    """
    计算bboxes iou
    Args:
        feat_sz: 特征图大小
        box1: predict bboxes
        box2: target bboxes
        xywh: 将bboxes转换为xyxy的形式
        GIoU: 为True时计算GIoU LOSS (yolov5自带)
        DIoU: 为True时计算DIoU LOSS (yolov5自带)
        CIoU: 为True时计算CIoU LOSS (yolov5自带，默认使用)
        SIoU: 为True时计算SIoU LOSS (新增)
        EIoU: 为True时计算EIoU LOSS (新增)
        WIoU: 为True时计算WIoU LOSS (新增)
        MPDIoU: 为True时计算MPDIoU LOSS (新增)
        alpha: AlphaIoU中的alpha参数，默认为1，为1时则为普通的IoU，如果想采用AlphaIoU，论文alpha默认值为3，此时设置CIoU=True则为AlphaCIoU
        scale: scale为True时，WIoU会乘以一个系数
        monotonous: 3个输入分别代表WIoU的3个版本，None: origin v1, True: monotonic FM v2, False: non-monotonic FM v3
        ratio: Inner-IoU对应的是尺度因子，通常取范围为[0.5，1.5],原文中VOC数据集对应的Inner-CIoU和Inner-SIoU设置在[0.7，0.8]之间有较大提升，
        数据集中大目标多则设置<1，小目标多设置>1
        eps: 防止除0
    Returns:
        iou
    """
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
 
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
 
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
 
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        wise_scale = WIoU_Scale(1 - (inter / union), monotonous=monotonous)
 
    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter / (union + eps), alpha)  # alpha iou
    feat_h, feat_w = feat_sz
 
    # Inner-IoU
    if xywh:
        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
    else:
        x1, y1, x2, y2 = b1_x1, b1_y1, b2_x1, b2_y1
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_ * ratio, x1 + w1_ * ratio, \
                                                             y1 - h1_ * ratio, y1 + h1_ * ratio
        inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_ * ratio, x2 + w2_ * ratio, \
                                                             y2 - h2_ * ratio, y2 + h2_ * ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                  (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1 * ratio * h1 * ratio + w2 * ratio * h2 * ratio - inner_inter + eps
    inner_iou = inner_inter / inner_union
 
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU or MPDIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c_area = cw * ch + eps  # convex area
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                    b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                return inner_iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                return inner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw, s_ch = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps, (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1, sin_alpha_2 = torch.abs(s_cw) / sigma, torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x, rho_y = (s_cw / cw) ** 2, (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w, omiga_h = torch.abs(w1 - w2) / torch.max(w1, w2), torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return inner_iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)  # SIou
            elif WIoU:
                if scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - inner_iou) * torch.exp(
                        (rho2 / c2)), inner_iou  # WIoU v3 https://arxiv.org/abs/2301.10051
                return inner_iou, torch.exp((rho2 / c2))  # WIoU v1
            return inner_iou - rho2 / c2  # DIoU
        elif MPDIoU:
            d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
            d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
            mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
            return inner_iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow - torch.pow((c_area - union) / c_area + eps,
                                                                                   alpha)  # MPDIoU
        # c_area = cw * ch + eps  # convex area
        return inner_iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean
    """
    iou_mean = 1.
    _momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True
 
    def __init__(self, iou, monotonous=False):
        self.iou = iou
        self.monotonous = monotonous
        self._update(self)
 
    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()
 
    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


    



# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1,
                label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px,
            py.mean(1),
            linewidth=3,
            color='blue',
            label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px,
                  py,
                  save_dir='mc_curve.png',
                  names=(),
                  xlabel='Confidence',
                  ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1,
                    label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1,
                color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px,
            y,
            linewidth=3,
            color='blue',
            label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
