#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import PIL.Image
from tqdm import tqdm

import labelme


def main():
    #/media/chen/177e252a-7948-4f62-8f05-e7e0974777bd/DATA_all/Incremental_learning/mask_2/masks/RTM_4_0125.png
    output_dir = '/home/xt/Desktop/gridmask/datsset' 
    input_dir = '/home/xt/Desktop/gridmask/json'
    # '/media/chen/177e252a-7948-4f62-8f05-e7e0974777bd/DATA_all/add_json'
    # '/media/chen/299D817A2D97AD94/cp/CV/labelme/examples/semantic_segmentation/need_label_mask'
    # '/media/chen/177e252a-7948-4f62-8f05-e7e0974777bd/DATA_all/ENV_datasets'
    labels = '/home/xt/Desktop/gridmask/labels/labels.txt'
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(osp.join(output_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks_npy'), exist_ok=True)
    os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)
    # os.makedirs(osp.join(output_dir, 'SegmentationClassVisualization'),
    #             exist_ok=True)
    print('Creating dataset:', output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    # # colormap = labelme.utils.label_colormap(255)
    # xml_files = [
    #     xml_file.stem for xml_file in Path(xml_files).iterdir()
    #     if xml_file.suffix == ".xml"
    # ]
    # label_files = [
    #     xml_file.stem for xml_file in Path(input_dir).iterdir()
    #     if xml_file.suffix == ".npy"
    # ]
    # glob.glob(Path(osp.join(xml_files, '*.xml')).basename())
    # print(xml_files)
    label_files = glob.glob(osp.join(input_dir, '*.json'))
    for label_file in tqdm(label_files):
        # print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(output_dir, 'JPEGImages',
                                    base + '.jpg')
            out_lbl_file = osp.join(output_dir, 'masks_npy',
                                    base + '.npy')
            out_png_file = osp.join(output_dir, 'masks', base + '.png')
            out_viz_file = osp.join(
                output_dir,
                'SegmentationClassVisualization',
                base + '.jpg',
            )

            data = json.load(f)
            img_names = data['imagePath']
            if img_names.endswith('JPG'):
                img_names = img_names.replace('JPG', 'jpg')
            img_file = osp.join(osp.dirname(label_file), img_names)
            try:
                img = np.asarray(PIL.Image.open(img_file))
            except Exception as e:
                img_file = img_file.replace('jpg', 'JPG')
                img = np.asarray(PIL.Image.open(img_file))
            # PIL.Image.fromarray(img).save(out_img_file)

            lbl, _ = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            
            # labelme.utils.lblsave(out_png_file, lbl)
            mask = PIL.Image.fromarray(lbl.astype(np.uint8), mode="L")
            mask.save(out_png_file)

            np.save(out_lbl_file, lbl)

            # viz = labelme.utils.draw_label(lbl,
            #                                img,
            #                                class_names,
            #                                colormap=colormap)
            # PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()