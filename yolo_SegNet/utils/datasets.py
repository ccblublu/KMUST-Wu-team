# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool,Pool
from pathlib import Path
from threading import Thread
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags,ImageOps
from torch.utils.data import Dataset,DataLoader,distributed
from tqdm import tqdm

import pickle
from copy import deepcopy
#from pycocotools import mask as maskUtils
from torchvision.utils import save_image
from torchvision.ops import roi_pool, roi_align, ps_roi_pool, ps_roi_align

from utils.general import (check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, 
    resample_segments, clean_str,xyxy2xywhn,LOGGER)
from utils.torch_utils import torch_distributed_zero_first
from utils.plots import plot_images
# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
HASH_TABLE = defaultdict(list)
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break
def compute_overlap(a, b, over_threshold=0.5):
    """
    Parameters
    ----------
    a: (N, 4)  ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = area

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    overlap = intersection / ua
    index = overlap > over_threshold
    return index

class GtBoxBasedCrop(object):
    """
        Crop around the gt bbox.
        Note:
            Here 'img_shape' is change to the shape of img_cropped.
    """

    def __init__(self,crop_size, ignore_label=None, im_nums=4,):
        self.crop_size = crop_size  # (w, h)
        self.ignore_label = ignore_label
        self.subim_nums = im_nums
        
        

    def __call__(self, mask, img, labels):

        self.label = labels
        self.ori_mask = mask
        self.ori_image = img
        self.h, self.w, _ = self.ori_image.shape
        self.crop_size = min(self.h, self.w, self.crop_size)
        self.ignore_label_index = (self.label[:,0] != self.ignore_label).astype(int)
        rest_index = np.ones(len(labels),int)
        rest_index *= self.ignore_label_index

        mask_out = []
        img_out = []
        labels_out = []
        for _ in range(self.subim_nums):
            if rest_index.sum() == 0:
                    rest_index = np.ones(len(labels),int)
            mask_, img_, labels_, rest_index = self._crop_patchv2(rest_index)
            mask_out.append(mask_)
            img_out.append(img_)
            labels_out.append(labels_)
        
    
        return mask_out, img_out, labels_out
    
    def _crop_patchv2(self, rest_index):
        px = py = self.crop_size
        obj_num = len(rest_index)
        if obj_num == 0:
            raise ValueError('No labels!')
            #  return mask, img, labels, labels

        # select_gt_id = np.random.randint(0, obj_num)
        select_gt_id = np.random.choice(np.where(rest_index != 0)[0])
        HASH_TABLE[INDEX_TEST].append(select_gt_id)
        # while rest_index[select_gt_id] == 0:
        #     select_gt_id = np.random.randint(0, obj_num)
        # for select_gt_id in range(obj_num):
        labels = np.copy(self.label)
        # labels = np.copy(all_labels)
        labels[:,1:] = xywhn2xyxy(labels[:, 1:], h=self.h, w=self.w)
        # all_labels[:, 1:] = xywhn2xyxy(all_labels[:, 1:], h=h, w=w)
        x1, y1, x2, y2 = labels[select_gt_id, 1:]
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(x2,self.w), min(y2,self.h)    # 确保x1,y1,x2,y2不能超出原图，否则下面CROP会出错

        # print("obj_num:",obj_num)
        # print("select_gt:",gt_labels[select_gt_id])
        # print("H,W,px,py,PATCH:",H,W,px,py,x1,y1,x2,y2,)
        # print("nx,ny:",max(x2 - px, 0),min(x1 + 1, W - px + 1),max(y2 - py, 0),min(y1 + 1, H - py + 1))

        if x2 - x1 > px:    # 大于px,py的边截取至px,py
            nx = np.random.randint(x1, x2 - px + 1)
        else:   # 小于等于px的边从max(x2 - px, 0)到min(x1 + 1, W - px + 1)之间取左上角x
            nx = np.random.randint(max(x2 - px, 0), min(x1 + 1, self.w - px + 1))
        if y2 - y1 > py:
            ny = np.random.randint(y1, y2 - py + 1)
        else:
            ny = np.random.randint(max(y2 - py, 0), min(y1 + 1, self.h - py + 1))
        patch_coord = np.zeros((1, 5), dtype="int")
        patch_coord[0, 1] = nx
        patch_coord[0, 2] = ny
        patch_coord[0, 3] = nx + px
        patch_coord[0, 4] = ny + py
        index = compute_overlap(patch_coord[:,1:], labels[:, 1:], 0.4)
        index = np.squeeze(index, axis=0)
        index[select_gt_id] = True
        rest_index *= (1 - index.astype(int))
        patch_img = self.ori_image[ny: ny + py, nx: nx + px, :]
        patch_mask = self.ori_mask[ny: ny + py, nx: nx + px]
        labels = labels[index, :]
        labels[:, 1] = np.maximum(labels[:, 1] - patch_coord[0, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2] - patch_coord[0, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3] - patch_coord[0, 1], px - 1)
        labels[:, 4] = np.minimum(labels[:, 4] - patch_coord[0, 2], py - 1)
        labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w=px, h=py)
            # all_patch.append(patch)
            # all_gt_bboxes.append(gt_bboxes)
            # all_gt_labels.append(gt_labels)
        return patch_mask, patch_img, labels, rest_index
    
    # def __call__(self, mask, img, labels):
    #     rest_labels = labels.copy()
    #     mask_out = []
    #     img_out = []
    #     labels_out = []
    #     for _ in range(self.subim_nums):
    #         if rest_labels.shape[0]==0:
    #             rest_labels = labels.copy()
    #         mask_, img_, labels_, rest_labels = self._crop_patch(mask, img, rest_labels, labels)
    #         mask_out.append(mask_)
    #         img_out.append(img_)
    #         labels_out.append(labels_)
    #     return mask_out, img_out, labels_out

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str

    

    def _crop_patch(self, mask, img, labels0, all_labels):

        h, w, c = img.shape
        self.crop_size = min(h, w, self.crop_size)
        px = py = self.crop_size
        obj_num = labels0.shape[0]
        if obj_num == 0:
            raise ValueError('No labels!')
            #  return mask, img, labels, labels
        ignore_label_index = (labels0[:,0]==self.ignore_label)
        select_gt_id = np.random.randint(0, obj_num)
        while ignore_label_index[select_gt_id]:
            select_gt_id = np.random.randint(0, obj_num)
        # for select_gt_id in range(obj_num):
        labels_ = np.copy(labels0)
        labels = np.copy(all_labels)
        labels[:,1:] = xywhn2xyxy(all_labels[:, 1:], h=h, w=w)
        # all_labels[:, 1:] = xywhn2xyxy(all_labels[:, 1:], h=h, w=w)
        x1, y1, x2, y2 = labels_[select_gt_id, 1:]
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(x2,w), min(y2,h)    # 确保x1,y1,x2,y2不能超出原图，否则下面CROP会出错

        # print("obj_num:",obj_num)
        # print("select_gt:",gt_labels[select_gt_id])
        # print("H,W,px,py,PATCH:",H,W,px,py,x1,y1,x2,y2,)
        # print("nx,ny:",max(x2 - px, 0),min(x1 + 1, W - px + 1),max(y2 - py, 0),min(y1 + 1, H - py + 1))

        if x2 - x1 > px:    # 大于px,py的边截取至px,py
            nx = np.random.randint(x1, x2 - px + 1)
        else:   # 小于等于px的边从max(x2 - px, 0)到min(x1 + 1, W - px + 1)之间取左上角x
            nx = np.random.randint(max(x2 - px, 0), min(x1 + 1, w - px + 1))

        if y2 - y1 > py:
            ny = np.random.randint(y1, y2 - py + 1)
        else:
            # try:
            ny = np.random.randint(max(y2 - py, 0), min(y1 + 1, h - py + 1))
            # except ValueError:
            #     print(y1,y2,py, h)
                # raise ValueError
        patch_coord = np.zeros((1, 5), dtype="int")
        patch_coord[0, 1] = nx
        patch_coord[0, 2] = ny
        patch_coord[0, 3] = nx + px
        patch_coord[0, 4] = ny + py
        index = compute_overlap(patch_coord[:,1:], labels[:, 1:], 0.4)
        index = np.squeeze(index, axis=0)
        index[select_gt_id] = True
        rest_labels = all_labels[~index]
        patch_img = img[ny: ny + py, nx: nx + px, :]
        patch_mask = mask[ny: ny + py, nx: nx + px]
        labels = labels[index, :]

        labels[:, 1] = np.maximum(labels[:, 1] - patch_coord[0, 1], 0)
        labels[:, 2] = np.maximum(labels[:, 2] - patch_coord[0, 2], 0)
        labels[:, 3] = np.minimum(labels[:, 3] - patch_coord[0, 1], px - 1)
        labels[:, 4] = np.minimum(labels[:, 4] - patch_coord[0, 2], py - 1)
        labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w=px, h=py)
            # all_patch.append(patch)
            # all_gt_bboxes.append(gt_bboxes)
            # all_gt_labels.append(gt_labels)
        return patch_mask, patch_img, labels, rest_labels
    
def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=False,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            #print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def img2mask_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'masks' + os.sep  # /images/, /labels/ substrings
    return ['png'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

class LoadImagesAndLabels(Dataset):  # for training/testing    加载数据集
    cach_version = 0.6
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path        
        self.albumentations = Albumentations() if augment else None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            self.mask_files = [x.replace('images','masks') for x in self.img_files]
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        self.mask_files = img2mask_paths(cache.keys())
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = [0,1,2,3]  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.mask_files = [self.mask_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            if stride ==0:
                stride=1
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int_) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs,self.img_npy,self.masks,self.mask_npy = [None] * n,[None] * n,[None] * n,[None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
                self.mask_cache_dir = Path(Path(self.mask_files[0]).parent.as_posix() + '_npy')
                self.mask_npy = [self.mask_cache_dir / Path(f).with_suffix('.npy').name for f in self.mask_files]
                self.mask_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                        np.save(self.mask_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size + self.mask_npy[i].stat().st_size
                else:
                    self.mask[i],self.imgs[i],self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes + self.masks[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB{cache_images})'
            pbar.close()
            
    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):  # is segment
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):   #每个step训练的时候，都会通过__getirem__获取一批训练数据

        index = self.indices[index]  # linear, shuffled, or image_weight

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # if random.random() < 0.8: #这一行很奇怪
            # Load mosaic
            img, mask , labels = load_mosaic(self, index)#load mosaic4
            shapes = None
            # else:
            #     mask, img, (h0, w0), (h, w) , labels= load_image(self, index)
            #     shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, mask, labels  = mixup(img, mask, labels, *load_mosaic(self, random.randint(0, self.n - 1)))
            # r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            # img = (img * r + img2 * (1 - r)).astype(np.uint8)
            # labels = np.concatenate((labels, labels2), 0)
        else:
            #mask, img, (h0, w0), (h, w) = load_image(self, index)
            mask, img, (h0, w0), (h, w) = load_image(self, index)
           # segments = self.segments[index].copy() 
                # mask, img, (h0, w0), (h, w) = load_image(self, index)
            # Letterbox
            #shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            shape = self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            
            mask, _, _ = letterbox(mask, shape, color=(0,0,0), auto=False, scaleup=self.augment) # TODO check
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, mask, labels = random_perspective(img, mask, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if self.augment:
            # Albumentations
#            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                mask = np.flipud(mask)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # mask = mask.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        
        return torch.from_numpy(img), torch.from_numpy(mask).long(), labels_out, self.img_files[index], shapes
    
    # @staticmethod
    # def collate_fn(batch):
    #     img, mask, label, path, shapes = zip(*batch)  # transposed
    #     for i, l in enumerate(label):
    #         l[:, 0] = i  # add target image index for build_targets()
    #     return torch.stack(img, 0), torch.stack(mask, 0), torch.cat(label, 0), path, shapes
    #     #return torch.stack(img, 0), mask, torch.cat(label, 0), path, shapes
    @staticmethod
    def collate_fn(batch):
        img, mask, label, path, shapes = zip(*batch)  # transposed
        # if len(img) == 1 and len(img[0].shape) == 4:
        #     return torch.cat(img, 0), torch.cat(mask, 0), torch.cat(label, 0), path, shapes
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.stack(mask, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, mask, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, mask4, label4, path4, shapes4 = [], [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

# Ancillary functions --------------------------------------------------------------------------------------------------
# def load_image(self, index):
#     # loads 1 image from dataset index 'i', returns im, original hw, resized hw
#     img = self.imgs[index]   #有
#     mask = self.masks[index]
#     path_img = self.img_files[index]   #有
#     img = cv2.imread(path_img)  # BGR  有
#     assert img is not None, f'Image Not Found {path_img}'   #有
#     path_mask = self.mask_files[index]
#     mask = cv2.imread(path_mask,0)  # BGR
#     h0, w0 = img.shape[:2]  # orig hw   有
#     assert mask is not None, f'Image Not Found {path_img}'
#     h0, w0 = img.shape[:2]  # orig hw   有
#     r = self.img_size / max(h0, w0)  # ratio    有
#     if r != 1:  # if sizes are not equal  有
#                 img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 mask = mask/255
#     return mask, img, (h0, w0), img.shape[:2],labels # im, hw_original, hw_resized   有
def load_image(self, index):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    #random.seed(seed)
    #i = random.randint(0,self.n - 1)
    im = self.imgs[index]
    mask = self.masks[index]
    labels1 = self.labels[index].copy() #就copy了一下
    # if im is None:  # not cached in ram
    #     npy_img = self.img_npy[index]
    #     npy_mask = self.mask_npy[index]
    #     if npy_img and npy_img.exists():  # load npy
    #         im = np.load(npy_img)
    #     if npy_mask and npy_mask.exists():  # load npy
    #         mask = np.load(npy_mask)
    #         h0, w0 = im.shape[:2]  # orig hw
    #         r = self.img_size / max(h0, w0)  # ratio/
    #         # r = self.img_size / min(h0, w0)  # ratio
    #         if r != 1:  # if sizes are not equal
    #             im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
    #             #im = cv2.resize(im, (640,640),
    #                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    #             mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
    #             #mask = cv2.resize(mask, (640,640),
    #                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
    #             # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
    #         return mask, im, (h0, w0), im.shape[:2],labels1  # im, hw_original, hw_resized
        # else:  # read image
    path_img = self.img_files[index]   #
    im = cv2.imread(path_img)  # BGR
    assert im is not None, f'Image Not Found {path_img}'
    path_mask = self.mask_files[index]
    mask = cv2.imread(path_mask,0)  # BGR
    h0, w0 = im.shape[:2]  # orig hw
    assert mask is not None, f'Image Not Found {path_img}'
    h0, w0 = im.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                mask = mask/255
    return mask, im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

# def load_image(self, i, seed=None):
#     # loads 1 image from dataset index 'i', returns im, original hw, resized hw
#     #random.seed(seed)
#     #i = random.randint(0,self.n - 1)
#     im = self.imgs[i]
#     mask = self.masks[i]
#     if im is None:  # not cached in ram
#         npy_img = self.img_npy[i]
#         npy_mask = self.mask_npy[i]
#         if npy_img and npy_img.exists():  # load npy
#             im = np.load(npy_img)
#         if npy_mask and npy_mask.exists():  # load npy
#             mask = np.load(npy_mask)
#             h0, w0 = im.shape[:2]  # orig hw
#             r = self.img_size / max(h0, w0)  # ratio/
#             # r = self.img_size / min(h0, w0)  # ratio
#             if r != 1:  # if sizes are not equal
#                 im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
#                 #im = cv2.resize(im, (640,640),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
#                 #mask = cv2.resize(mask, (640,640),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
#             return mask, im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#         else:  # read image
#             path_img = self.img_files[i]   #
#             im = cv2.imread(path_img)  # BGR
#             assert im is not None, f'Image Not Found {path_img}'
#             path_mask = self.mask_files[i]
#             mask = cv2.imread(path_mask, 0)  # BGR
#             # mask = Image.open(path_mask)
#             # mask = np.load(npy_mask)
#             h0, w0 = im.shape[:2]  # orig hw
#             assert mask is not None, f'Image Not Found {path_img}'
#             h0, w0 = im.shape[:2]  # orig hw
#             r = self.img_size / max(h0, w0)  # ratio
#             # r = self.img_size / min(h0, w0)  # ratio
#             if r != 1:  # if sizes are not equal
#                 im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
#                                 interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#                 # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
#             return mask, im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
#     else:
#         return self.masks[i][...,1], self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def load_crop_image(self, j):
 
    process_n = 4
    # loads 1 image from dataset index 'j', returns im, original hw, resized hw
    #random.seed(seed)
    #j = random.randint(0,self.n - (self.n)*0.5)
    labels0 = self.labels[j].copy()
    while labels0.shape[0] == 0:
        j = random.randint(0, self.n - 1)
        labels0 = self.labels[j].copy()
    im0 = self.imgs[j]
    mask0 = self.masks[j]
    labels4, segments4 = [], []
    yc = xc = s = self.img_size  # mosaic center x, y
    #yc = xc = s = 480
    if im0 is None:  # not cached in ram
        npy_img = self.img_npy[j]
        npy_mask = self.mask_npy[j]
        if npy_img and npy_img.exists():  # load npy
            im0 = np.load(npy_img)
    if npy_mask and npy_mask.exists():  # load npy
            mask0 = np.load(npy_mask)
    else:  # read image
        path_img = self.img_files[j]
        im0 = cv2.imread(path_img)
        path_mask = self.mask_files[j]
        mask0 = cv2.imread(path_mask, 0)
        #raise ValueError("no such files!")
    global INDEX_TEST
    INDEX_TEST = j
        #raise ValueError("no such files!")
    #process = GtBoxBasedCrop(320, im_nums=process_n)
    process = GtBoxBasedCrop(640, im_nums=process_n)
    masks, ims, labels_all= process(mask0, im0, labels0)
    h = w = self.img_size
    #h = w = 426
    if process_n == 4:
        for i, (mask, img, labels) in enumerate(zip(masks, ims, labels_all)):
            segments = self.segments[j].copy()
    # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                mask4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)  # base image with 4 tiles
                #img4 = np.full((960, 960, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                #mask4 = np.full((960, 960), 0, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            mg = cv2.resize(img, (640, 640))
            mask = cv2.resize(mask,(640,640))
            #y1a,y1b,y2a,y2b,x1a,x1b,x2a,x2b = int(0.75*y1a),int(0.75*y1b),int(0.75*y2a),int(0.75*y2b),int(0.75*x1a),int(0.75*x1b),int(0.75*x2a),int(0.75*x2b) 
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b] 
            padw = x1a - x1b
            padh = y1a - y1b

    # Labels

            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        labels4[:, 1:] = xyxy2xywhn(labels4[:, 1:], w=2 * s, h=2 * s)
    
    elif process_n == 9:
        for i, (mask, img, labels) in enumerate(zip(masks, ims, labels_all)):
        # Load image
        # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                mask9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)
            segments = self.segments[j].copy() 
            
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            mask9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        labels4[:, 1:] = xyxy2xywhn(labels4[:, 1:], w=2 * s, h=2 * s)
    h0, w0 = img4.shape[:2]  # orig hw
    r = self.img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        im = cv2.resize(img4, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        mask = cv2.resize(mask4, (int(w0 * r), int(h0 * r)),
                        interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
    return mask, im, (h0, w0), im.shape[:2], labels4  # im, hw_original, hw_resized

# def load_mosaic(self, index):
#     # loads images in a 4-mosaic

#     labels4, segments4 = [], []
#     s = self.img_size
#     yc, xc = (s,s)#[int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
#     indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
#     for i, index in enumerate(indices):
#         # Load image
#         mask,img, _, (h, w),labels = load_image(self, index)
#         #segments = self.segments[index].copy()

#         # place img in img4
#         if i == 0:  # top left
#             img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#             mask4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)
#             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#         elif i == 1:  # top right
#             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#         elif i == 2:  # bottom left
#             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#         elif i == 3:  # bottom right
#             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#         mask4[y1a:y2a,x1a:x2a] = mask[y1b:y2b, x1b:x2b]

#         padw = x1a - x1b
#         padh = y1a - y1b

#         # Labels
#         labels, segments = self.labels[index].copy(), self.segments[index].copy()
#         if labels.size:
#             labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
#             segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
#         labels4.append(labels)
#         segments4.extend(segments)

#     # Concat/clip labels
#     labels4 = np.concatenate(labels4, 0)
#     for x in (labels4[:, 1:], *segments4):
#         np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
#     # img4, labels4 = replicate(img4, labels4)  # replicate

#     # Augment
#     img4, mask4,labels4, segments4 = copy_paste(img4, mask4,labels4, segments4, probability=self.hyp['copy_paste'])
#     img4, mask4,labels4 = random_perspective(img4, mask4,labels4, segments4,
#                                        degrees=self.hyp['degrees'],
#                                        translate=self.hyp['translate'],
#                                        scale=self.hyp['scale'],
#                                        shear=self.hyp['shear'],
#                                        perspective=self.hyp['perspective'],
#                                        border=self.mosaic_border)  # border to remove

#     return img4, mask4,labels4
def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc = xc = s = self.img_size
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        labels, segments = self.labels[index].copy(), self.segments[index].copy() #这里已经有lables了，函数中也没有对其进行处理过吧
        mask, img, _, (h, w) = load_image(self, index) # 这写的是啥... self是啥啊？ 为什么要返回labels？
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            mask4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b] 
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels     
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, mask4, labels4, segments4 = copy_paste(img4, mask4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, mask4, labels4  = random_perspective(img4, mask4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, mask4 ,labels4

def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        mask, img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
 
    img9, labels9 = random_perspective(img9, labels9,segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9,labels9


def load_samples(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks


# def copy_paste(img, mask,labels, segments, probability=0.5):
#     # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
#     n = len(segments)
#     if probability and n:
#         h, w, c = img.shape  # height, width, channels
#         im_new = np.zeros(img.shape, np.uint8)
#         for j in random.sample(range(n), k=round(probability * n)):
#             l, s = labels[j], segments[j]
#             box = w - l[3], l[2], w - l[1], l[4]
#             ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
#             if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
#                 labels = np.concatenate((labels, [[l[0], *box]]), 0)
#                 segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
#                 cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

#         result = cv2.bitwise_and(src1=img, src2=im_new)
#         result = cv2.flip(result, 1)  # augment segments (flip left-right)
#         i = result > 0  # pixels to replace
#         # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
#         img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
#         mask[i] =0
#     return img, mask,labels, segments
def copy_paste(img, mask, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)
        # mask_new = np.full_like(mask, -1, np.uint8)
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug
        mask[i] = 0
    return img, mask, labels, segments


def remove_background(img, labels, segments):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    h, w, c = img.shape  # height, width, channels
    im_new = np.zeros(img.shape, np.uint8)
    img_new = np.ones(img.shape, np.uint8) * 114
    for j in range(n):
        cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        
        i = result > 0  # pixels to replace
        img_new[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img_new, labels, segments


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0,w-1), l[2].astype(int).clip(0,h-1), l[3].astype(int).clip(0,w-1), l[4].astype(int).clip(0,h-1) 
            
            #print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue
            
            sample_labels.append(l[0])
            
            mask = np.zeros(img.shape, np.uint8)
            
            cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
            sample_masks.append(mask[box[1]:box[3],box[0]:box[2],:])
            
            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            #print(box)
            sample_images.append(mask[box[1]:box[3],box[0]:box[2],:])

    return sample_labels, sample_images, sample_masks


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border 
    return im, ratio, (dw, dh)


def random_perspective(im, mask, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=(0, 0, 0))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

    # Visualize
    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, mask, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area
    

def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels
    

def pastein(image, labels, sample_labels, sample_images, sample_masks):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)   
        
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area     
        else:
            ioa = np.zeros(1)
        
        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels)-1)
            #print(len(sample_labels))
            #print(sel_ind)
            #print((xmax-xmin, ymax-ymin))
            #print(image[ymin:ymax, xmin:xmax].shape)
            #print([[sample_labels[sel_ind], *box]])
            #print(labels.shape)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
            r_w = int(ws*r_scale)
            r_h = int(hs*r_scale)
            
            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    #print(sample_labels[sel_ind])
                    #print(sample_images[sel_ind].shape)
                    #print(temp_crop.shape)
                    box = np.array([xmin, ymin, xmin+r_w, ymin+r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])
                              
                    image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

    return labels

class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        import albumentations as A

        self.transform = A.Compose([
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.01),
            A.RandomGamma(gamma_limit=[80, 120], p=0.01),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.01),],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            #logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco')
    Arguments
        path:           Path to images directory
        weights:        Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file
    
def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in img_formats, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any([len(x) > 8 for x in l]):  # is segment
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 5, f'labels require 5 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all()#, f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    l = l[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        # with open('./vaiod_label.txt', 'w') as f:
        #     f.writelines(im_file)
        nc = 1
        msg = f'{prefix}WARNING: {im_file}'#: ignoring corrupt image/label: {e}
        return [None, None, None, None, nm, nf, ne, nc, msg]
    

def mixup(im, mask, labels, im2, mask2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    mask = cv2.bitwise_and(src1=mask, src2=mask2)
    return im, mask, labels

def load_segmentations(self, index):
    key = '/work/handsomejw66/coco17/' + self.img_files[index]
    #print(key)
    # /work/handsomejw66/coco17/
    return self.segs[key]
if __name__ == "__main__":
    s = GtBoxBasedCrop(320)
    img = np.load('/home/xt/Desktop/yolov7+seg/data/dataSets/images/')
    mask = np.load('/home/xt/Desktop/yoloxt/9.20/data/dataSets/masks')
     
    gt_bboxes = []
    gt_labels = []
    with open('/home/xt/Desktop/yolov7+seg/data/dataSets/labels', "r") as f:
        tmp = f.readlines()
        f.close()
    for f in tmp:
        sub = list(map(float,f.strip().split(" ")))
        gt_labels.append(int(sub[0]))
        gt_bboxes.append(sub[1:])
    labels = np.concatenate((np.expand_dims(np.array(gt_labels), 1), np.array(gt_bboxes)), axis=1)
    mask_1, img_1, labels_1 = s(mask, img, labels)
    img_1 =img_1.transpose(2,0,1)
    n = len(labels_1)
    labels_1 = np.repeat(np.expand_dims(labels_1, 0), 9, axis=0).reshape(-1, 5)
    batch_id = np.zeros((len(labels_1), 1))
    for i in range(9):
        batch_id[i * n: i * n + n]=i
    
    plot_images(np.repeat(np.expand_dims((img_1), 0),9,axis=0), np.concatenate((batch_id, labels_1), axis=-1), np.repeat(np.expand_dims(mask_1,0), 9, axis=0))

