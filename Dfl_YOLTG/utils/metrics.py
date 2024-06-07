# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general

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

        self.hist += fast_hist(predict.astype(np.int64), target.astype(np.int64), self.class_num)

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




def fitness(x):
    # Model fitness as a weighted combination of wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwws
    w = [0.0, 0.0, 0.1, 0.9, 0.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95 ,IOU]
    return (x[:, :5] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
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
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    pr_file = Path(save_dir) / 'PR_curve.csv'
    s = '' if pr_file.exists() else (('%20s,' * len(names) % tuple(names.values())).rstrip(',') +'\n')  # add header
    with open(pr_file, 'a') as f:
        f.write(s)
        for i in zip(px, *py):
             ap_ = sum(i[1:]) / 4
             f.write(('%20.5g,' * len(list(i) + [ap_]) % tuple(list(i) + [ap_])).rstrip(',') + '\n')
        f.close()
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')
# def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
#     """ Compute the average precision, given the recall and precision curves.
#     Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
#     # Arguments
#         tp:  True positives (nparray, nx1 or nx10).
#         conf:  Objectness value from 0-1 (nparray).
#         pred_cls:  Predicted object classes (nparray).
#         target_cls:  True object classes (nparray).
#         plot:  Plot precision-recall curve at mAP@0.5
#         save_dir:  Plot save directory
#     # Returns
#         The average precision as computed in py-faster-rcnn.
#     """

#     # Sort by objectness
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

#     # Find unique classes
#     unique_classes = np.unique(target_cls)
#     nc = unique_classes.shape[0]  # number of classes, number of detections

#     # Create Precision-Recall curve and compute AP for each class
#     px, py = np.linspace(0, 1, 1000), []  # for plotting
#     ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         n_l = (target_cls == c).sum()  # number of labels
#         n_p = i.sum()  # number of predictions

#         if n_p == 0 or n_l == 0:
#             continue
#         else:
#             # Accumulate FPs and TPs
#             fpc = (1 - tp[i]).cumsum(0)
#             tpc = tp[i].cumsum(0)

#             # Recall
#             recall = tpc / (n_l + 1e-16)  # recall curve
#             r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

#             # Precision
#             precision = tpc / (tpc + fpc)  # precision curve
#             p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

#             # AP from recall-precision curve
#             for j in range(tp.shape[1]):
#                 ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
#                 if plot and j == 0:
#                     py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

#     # Compute F1 (harmonic mean of precision and recall)
#     f1 = 2 * p * r / (p + r + 1e-16)
#     if plot:
#         plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
#         plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
#         plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
#         plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

#     i = f1.mean(0).argmax()  # max F1 index
#     return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
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
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))




# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
