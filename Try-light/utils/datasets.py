# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

from cProfile import label
from collections import defaultdict
from distutils.fancy_getopt import FancyGetopt
from email.policy import default
import glob
import hashlib
import json
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from numba import jit
from train import CROP_ONLINE


from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from utils.plots import plot_images

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

HASH_TABLE = defaultdict(list)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


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


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
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


class InfiniteDataLoader(dataloader.DataLoader):
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


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

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
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        #TODO: 采用滑窗形式裁剪为小图

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
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
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def img2mask_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'masks' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.png' for x in img_paths]

class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

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
        """
        首先读取图像路径，转换合适的格式，根据图像路径，替换其中的images和图片后缀，转换成label路径
        读取coco128/labels/train.cache文件,没有则创建，cache存储字典{图片路径:label路径，图片大小}
        """

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            self.mask_files = [x.replace('images', 'masks') for x in self.img_files]
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
            # with open('./select.txt', 'w') as f:
            #     for x in cache['msgs']:
            #         f.write(x + '\n')
            #     f.close()
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        self.mask_files = img2mask_paths(cache.keys())
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int8)  # batch index.
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
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

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int_) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy, self.masks, self.mask_npy= [None] * n, [None] * n, [None] * n, [None] * n  
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
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))#NUM_THREADS
            # results = lambda x: load_image(*x), zip(repeat(self), range(n))#NUM_THREADS
            pbar = tqdm(enumerate(results), total=n)#
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[1])
                        np.save(self.mask_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size + self.mask_npy[i].stat().st_size
                else:
                    self.masks[i], self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes + self.masks[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):

        index = self.indices[index]  # linear, shuffled, or image_weight

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, mask, labels = load_mosaic(self, index)#load mosaic4
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, mask, labels = mixup(img, mask, labels, *load_mosaic(self, random.randint(0, self.n - 1)))
        
        else:
            # Load image
            if not self.augment and CROP_ONLINE:
                mask, img, shape, labels = load_val_image(self, index)
                img = img.transpose((0, 3, 1, 2))[:,::-1]  # HWC to CHW, BGR to RGB
                # mask = mask.transpose((2, 0, 1))
                img = np.ascontiguousarray(img)
                mask = np.ascontiguousarray(mask)
                trans = torch.nn.MaxPool2d(2, stride=2)
                img = trans(torch.from_numpy(img).to(float))
                mask = trans(torch.from_numpy(mask).to(float))

                return img, mask.long(), torch.from_numpy(labels), self.img_files[index], shape#self.img_size
            else:
                if self.mosaic and random.random() < self.hyp['CropOnline']:
                     mask, img, (h0, w0), (h, w), labels = load_crop_image(self, index)
                     segments = self.segments[index].copy() #None, just for struct
                else:
                    mask, img, (h0, w0), (h, w) = load_image(self, index)
                    labels, segments = self.labels[index].copy(), self.segments[index].copy() 
                # mask, img, (h0, w0), (h, w) = load_image(self, index)
            # Letterbox
            #shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            shape = self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            mask, _, _ = letterbox(mask, shape, color=(0,0,0), auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # labels = self.labels[index].copy()
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
            img, labels = self.albumentations(img, labels)
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

    @staticmethod
    def collate_fn(batch):
        img, mask, label, path, shapes = zip(*batch)  # transposed
        if len(img) == 1 and len(img[0].shape) == 4:
            return torch.cat(img, 0), torch.cat(mask, 0), torch.cat(label, 0), path, shapes
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
def load_image(self, i, seed=None):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    #random.seed(seed)
    #i = random.randint(0,self.n - 1)
    im = self.imgs[i]
    mask = self.masks[i]
    if im is None:  # not cached in ram
        npy_img = self.img_npy[i]
        npy_mask = self.mask_npy[i]
        if npy_img and npy_img.exists():  # load npy
            im = np.load(npy_img)
        if npy_mask and npy_mask.exists():  # load npy
            mask = np.load(npy_mask)
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio/
            # r = self.img_size / min(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                #im = cv2.resize(im, (640,640),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
                #mask = cv2.resize(mask, (640,640),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
            return mask, im, (h0, w0), im.shape[:2] # im, hw_original, hw_resized
        else:  # read image
            path_img = self.img_files[i]   #
            im = cv2.imread(path_img)  # BGR
            assert im is not None, f'Image Not Found {path_img}'
            path_mask = self.mask_files[i]
            mask = cv2.imread(path_mask, 0)  # BGR
            # mask = Image.open(path_mask)
            # mask = np.load(npy_mask)
            h0, w0 = im.shape[:2]  # orig hw
            assert mask is not None, f'Image Not Found {path_img}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            # r = self.img_size / min(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                # mask = mask.resize(size=(int(h0 * r), int(w0 * r)))
            return mask, im, (h0, w0), im.shape[:2] # im, hw_original, hw_resized
    else:
        return self.masks[i][...,1], self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

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
    #im0 = apply_grid_mask(im0,24,48)
    global INDEX_TEST
    INDEX_TEST = j
        #raise ValueError("no such files!")
    process = GtBoxBasedCrop(160, im_nums=process_n)
    #process = GtBoxBasedCrop(640, im_nums=process_n)
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

            img = cv2.resize(img, (640, 640))
            mask = cv2.resize(mask,(640,640))
            #img = apply_grid_mask(img,24,48)
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

def load_val_image(self, i, overlap=20,seed=None):
    # def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    #random.seed(seed)
    #i = random.randint(0,self.n - 1)
    im = self.imgs[i]
    mask = self.masks[i]
    labels = self.labels[i].copy()
    img_size = self.img_size * 2
    if im is None:  # not cached in ram
        npy_img = self.img_npy[i]
        npy_mask = self.mask_npy[i]
        if npy_img and npy_img.exists():
            im = np.load(npy_img)
            mask = np.load(npy_mask)
            h0, w0 = im.shape[:2]  # orig hw

            rows, cols =  int(np.ceil(h0 / (img_size - overlap))), int(np.ceil(w0 / (img_size - overlap)))
            im_out = np.full((rows*cols, img_size, img_size, 3), 112)
            mask_out = np.full((rows * cols, img_size, img_size), 0)
            im_out = cut_image(im, im_out, img_size, rows, cols, overlap)
            mask_out = cut_image(mask, mask_out, img_size, rows, cols, overlap)
            if len(labels):
                labels[:,1:] = xywhn2xyxy(labels[:,1:], w0, h0)
                i_ = np.zeros(len(labels))
                labels = np.insert(labels, 0, i_, 1)
                # h_s, w_s = np.arange(0, h0, img_size - overlap), np.arange(0, w0, img_size - overlap)
                # h, w = np.meshgrid(h_s,w_s)
                labels= cut_labels(labels, h0, w0, img_size, rows, cols, overlap)

            return mask_out, im_out, (rows, cols, img_size, h0, w0), np.concatenate(labels) #im.shape[:2]  # im, hw_original, hw_resized
        else:  # read image
            assert im is not None, f'Image Not Found !'
    else:
        return self.masks[i][...,1], self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


@jit(nopython=True)
def cut_image(im0, im, img_size, rows, cols, overlap):
    h0, w0 = im0.shape[:2]
    for r in range(rows):
        for c in range(cols):
            i = 0
            x_b, y_b = c * (img_size - overlap), r * (img_size - overlap)
            x_t, y_t = min(w0, x_b + img_size), min(h0, y_b + img_size)
            im[i, 0: (y_t - y_b), 0: (x_t - x_b)] = im0[y_b: y_t, x_b: x_t]
            i += 1
    return im

def cut_labels(labels, h0, w0, img_size, rows, cols, overlap):
    labels_out = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            x_b, y_b = c * (img_size - overlap), r * (img_size - overlap)
            x_t, y_t = min(w0, x_b + img_size), min(h0, y_b + img_size)
            coord = np.array([[x_b, y_b, x_t, y_t]])
            index = compute_overlap(coord, labels[:, 2:], 0.4).squeeze(0)
            chosen = labels[index].copy()
            if len(chosen):
                chosen[:, 2:] = clamp_coord(chosen[:, 2:], (x_b, y_b), img_size)
            chosen[:, 0] = i
            labels_out.append(chosen)
            i += 1
    return labels_out

def clamp_coord(labels, coord, size):
        labels[:, 0] = np.maximum(labels[:, 0] - coord[0], 0)
        labels[:, 1] = np.maximum(labels[:, 1] - coord[1], 0)
        labels[:, 2] = np.minimum(labels[:, 2] - coord[0], size - 1)
        labels[:, 3] = np.minimum(labels[:, 3] - coord[1], size - 1)
        return xyxy2xywhn(labels, w=size, h=size)

def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size

    # x_coords = [160, 160, 480, 480]
    # y_coords = [160, 480, 160, 480]
    # yc, xc = random.choice(list(zip(x_coords, y_coords)))
    yc, xc = (int(random.uniform(-x, 2*s + x)) for x in self.mosaic_border)  # mosaic center x, y
    #yc, xc = (int(random.uniform(max(160, -x), min(480, 2*s + x))) for x in self.mosaic_border)
    #yc = xc = s = self.img_size

    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        # if random.random() < self.hyp['CropOnline']:
        #     mask, img, _, (h, w), labels = load_crop_image(self, index)
        #     segments = self.segments[index].copy() #None, just for struct
            

        mask, img, _, (h, w) = load_image(self, index)
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
            #mask, img, _, (h, w),labels= load_image(self, index)
            #segments = self.segments[index].copy()
            
            
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
    img4, mask4, labels4, segments4 = copy_paste(img4, mask4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, mask4, labels4 = random_perspective(img4, mask4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, mask4, labels4


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


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
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


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


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
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
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


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats

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

def compute_overlap(a, b, over_threshold=0.5):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
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
if __name__ == "__main__":
    s = GtBoxBasedCrop(160)
    #img = np.load('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/images_npy/ENV_1_20201224_1_17101_0.npy')
    #mask = np.load('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/masks_npy/ENV_1_20201224_1_17101_0.npy')
    img = np.load('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/images_npy/')
    mask = np.load('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/masks_npy/')
     
    gt_bboxes = []
    gt_labels = []
    #with open('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/labels/ENV_1_20201224_1_17101_0.txt', "r") as f:
    with open('/home/zzd/ZZD/ZZDpy/yoloR/datasets/VOC2007/labels/', "r") as f:
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


def apply_grid_mask(img, d1, d2, rotate=1, ratio=0.2, mode=0, prob=0.8):
    if np.random.rand() > prob:  # Apply GridMask data augmentation with a certain probability
        return img
    h = 640
    w = 640
    c = 3 # Get the number of channels in the image

    # 1.5 * h, 1.5 * w works fine with squared images
    # But with rectangular input, the mask might not be able to recover back to the input image shape
    # A square mask with an edge length equal to the diagonal of the input image
    # will be able to cover all the image spots after rotation. This is also the minimum square.
    hh = math.ceil((math.sqrt(h * h + w * w)))
    d = np.random.randint(d1, d2)
    # ratio represents the amount of preserved original image information,
    # controlling the size of the filled areas by adjusting the ratio
    l = math.ceil(d * (1 - math.sqrt(1 - ratio)))
    maskk = np.ones((hh, hh,c), np.float32)  # Create a mask, clearly larger than the original image, with a size equal to the diagonal of the original image, for convenient rotation of the mask

    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    # This method is equivalent to filling the cross, and then taking the inverse of the mask will be the filled black rectangle (black border in Figure 4)
    for i in range(-1, hh // d + 1):
        s = d * i + st_h
        t = s + l  # You can see the meaning of l here
        s = max(min(s, hh), 0)  # Cannot exceed the boundaries of the mask
        t = max(min(t, hh), 0)
        maskk[s:t, :] = 0
    for i in range(-1, hh // d + 1):
        s = d * i + st_w
        t = s + l
        s = max(min(s, hh), 0)  # Cannot exceed the boundaries of the mask
        t = max(min(t, hh), 0)
        maskk[:, s:t] = 0
    r = np.random.randn(rotate)
    maskk = Image.fromarray(np.uint8(maskk))
    maskk = maskk.rotate(r)  # Mask rotation
    maskk = np.asarray(maskk)
    maskk = maskk[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]  # Crop the mask to the size of the image
    if mode == 1:
        maskk = 1 - maskk

    #print(np.sum(maskk) / (h * w))  # Verify if the ratio is accurate, which represents the amount of preserved original image information
    #maskk = torch.from_numpy(maskk).float()
    #maskk = maskk.expand_as(img)
    maskk = cv2.resize(maskk,(640,640))
    img = cv2.resize(img,(640,640))
    img = img * maskk  # Multiply the mask in the image, keeping the original image information for the values with 1 in the mask and deleting the corresponding parts for the values with 0

    return img



class MobileWindowModule(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=4):
        super(MobileWindowModule, self).__init__()
        self.block_size = block_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.concat = nn.Conv2d(out_channels * (block_size ** 2), out_channels, kernel_size=1)
        self.normal_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.normal_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.normal_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

    def mobile_window_branch(self, x):
        block_size = self.block_size
        block_height, block_width = x.size(2) // block_size, x.size(3) // block_size
        blocks = []
        for i in range(block_size):
            for j in range(block_size):
                block = x[:, :, i * block_height : (i + 1) * block_height, j * block_width : (j + 1) * block_width]
                block_upsampled = self.upsample(block)
                block_features = F.relu(self.conv1(block))
                block_features = F.relu(self.conv2(block_features))
                block_downsampled = self.downsample(block_features)
                block_final = block + block_downsampled
                blocks.append(block_final)
        output_window = torch.cat(blocks, dim=1)
        output_window = self.concat(output_window)
        return output_window

    def normal_branch(self, x):
        conv1_output = F.relu(self.normal_conv1(x))
        conv2_output = F.relu(self.normal_conv2(x))
        conv3_output = F.relu(self.normal_conv3(x))
        output = conv1_output + conv2_output + conv3_output
        return output

    def forward(self, x):
        # Branch 1: Mobile window
        output_window = self.mobile_window_branch(x)

        # Branch 2: Normal convolution
        output_normal = self.normal_branch(x)

        # Merge branches
        output = output_window + output_normal

        return output
    





    