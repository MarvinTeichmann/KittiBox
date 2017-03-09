from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc

sys.path.insert(1, '../incl')
from scipy.misc import imread, imresize

from utils.data_utils import (annotation_jitter, annotation_to_h5)
from utils.annolist import AnnotationLib as AnnoLib

import threading

from collections import namedtuple

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


tf.app.flags.DEFINE_boolean(
    'save', False, ('Whether to save the run. In case --nosave (default) '
                    'output will be saved to the folder TV_DIR_RUNS/debug, '
                    'hence it will get overwritten by further runs.'))

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')


fake_anno = namedtuple('fake_anno_object', ['rects'])


def _rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.y1 < r.y2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return anno


def _rescale_boxes2(current_shape, rect_list, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in rect_list:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.y1 < r.y2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return rect_list


def read_kitti_anno(label_file):
    """ Reads a kitti annotation file.

    Args:
    label_file: Path to file

    Returns:
      Lists of rectangels: Cars and don't care area.
    """
    labels = [line.rstrip().split(' ') for line in open(label_file)]
    rect_list = []
    for label in labels:
        if not (label[0] == 'Car' or label[0] == 'Van' or
                label[0] == 'DontCare'):
            continue
        if label[0] == 'DontCare':
            class_id = -1
        else:
            class_id = 1
        object_rect = AnnoLib.AnnoRect(
            x1=float(label[4]), y1=float(label[5]),
            x2=float(label[6]), y2=float(label[7]))
        assert object_rect.x1 < object_rect.x2
        assert object_rect.y1 < object_rect.y2
        object_rect.classID = class_id
        rect_list.append(object_rect)

    return rect_list


def _load_idl_tf(idlfile, hypes, jitter=False, random_shuffel=True):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = AnnoLib.parse(idlfile)
    annos = []
    for anno in annolist:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
        annos.append(anno)
    random.seed(0)
    if hypes['data']['truncate_data']:
        annos = annos[:10]
    for epoch in itertools.count():
        if random_shuffel:
            random.shuffle(annos)
        for anno in annos:
            im = imread(anno.imageName)
            if im.shape[2] == 4:
                im = im[:, :, :3]
            if im.shape[0] != hypes["image_height"] or \
               im.shape[1] != hypes["image_width"]:
                if epoch == 0:
                    anno = _rescale_boxes(im.shape, anno,
                                          hypes["image_height"],
                                          hypes["image_width"])
                im = imresize(
                    im, (hypes["image_height"], hypes["image_width"]),
                    interp='cubic')
            if jitter:
                jitter_scale_min = 0.9
                jitter_scale_max = 1.1
                jitter_offset = 16
                im, anno = annotation_jitter(
                    im, anno, target_width=hypes["image_width"],
                    target_height=hypes["image_height"],
                    jitter_scale_min=jitter_scale_min,
                    jitter_scale_max=jitter_scale_max,
                    jitter_offset=jitter_offset)

            boxes, flags = annotation_to_h5(hypes,
                                            anno,
                                            hypes["grid_width"],
                                            hypes["grid_height"],
                                            hypes["rnn_len"])

            yield {"image": im, "boxes": boxes, "flags": flags,
                   "rects": anno.rects, "anno": anno}


def _load_kitti_txt(kitti_txt, hypes, jitter=False, random_shuffel=True):
    """Take the idlfile and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    base_path = os.path.realpath(os.path.dirname(kitti_txt))
    files = [line.rstrip() for line in open(kitti_txt)]
    if hypes['data']['truncate_data']:
        files = files[:10]
        random.seed(0)
    for epoch in itertools.count():
        if random_shuffel:
            random.shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), \
                "File does not exist: %s" % gt_image_file

            rect_list = read_kitti_anno(gt_image_file)

            anno = fake_anno(rect_list)

            im = scp.misc.imread(image_file)
            if im.shape[2] == 4:
                im = im[:, :, :3]
            if im.shape[0] != hypes["image_height"] or \
               im.shape[1] != hypes["image_width"]:
                if epoch == 0:
                    anno = _rescale_boxes(im.shape, anno,
                                          hypes["image_height"],
                                          hypes["image_width"])
                im = imresize(
                    im, (hypes["image_height"], hypes["image_width"]),
                    interp='cubic')
            if jitter:
                jitter_scale_min = 0.9
                jitter_scale_max = 1.1
                jitter_offset = 16
                im, anno = annotation_jitter(
                    im, anno, target_width=hypes["image_width"],
                    target_height=hypes["image_height"],
                    jitter_scale_min=jitter_scale_min,
                    jitter_scale_max=jitter_scale_max,
                    jitter_offset=jitter_offset)

            pos_list = [rect for rect in anno.rects if rect.classID == 1]
            anno = fake_anno(pos_list)

            boxes, flags = annotation_to_h5(hypes,
                                            anno,
                                            hypes["grid_width"],
                                            hypes["grid_height"],
                                            hypes["rnn_len"])

            yield {"image": im, "boxes": boxes, "flags": flags,
                   "rects": pos_list}


def _make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v


def _load_data_gen(hypes, phase, jitter):
    grid_size = hypes['grid_width'] * hypes['grid_height']

    data_file = hypes["data"]['%s_idl' % phase]
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, data_file)

    data = _load_idl_tf(data_file, hypes,
                        jitter={'train': jitter, 'val': False}[phase])

    for d in data:
        output = {}
        rnn_len = hypes["rnn_len"]
        flags = d['flags'][0, :, 0, 0:rnn_len, 0]
        boxes = np.transpose(d['boxes'][0, :, :, 0:rnn_len, 0], (0, 2, 1))
        assert(flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 4))

        output['image'] = d['image']
        confs = [[_make_sparse(int(detection), d=hypes['num_classes'])
                  for detection in cell] for cell in flags]
        output['confs'] = np.array(confs)
        output['boxes'] = boxes
        output['flags'] = flags

        yield output


def test_new_kitti():
    idlfile = "/home/mifs/mttt2/cvfs/DATA/KittiBox/train_3.idl"
    kitti_txt = "/home/mifs/mttt2/cvfs/DATA/KittiBox/train.txt"

    with open('../hypes/kittiBox.json', 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    hypes["rnn_len"] = 1
    hypes["image_height"] = 200
    hypes["image_width"] = 800

    gen1 = _load_kitti_txt(kitti_txt, hypes, random_shuffel=False)
    gen2 = _load_idl_tf(idlfile, hypes, random_shuffel=False)

    print('testing generators')

    for i in range(20):
        data1 = gen1.next()
        data2 = gen2.next()
        rects1 = data1['rects']
        rects2 = data2['rects']

        assert len(rects1) <= len(rects2)

        if not len(rects1) == len(rects2):
            print('ignoring flags')
            continue
        else:
            print('comparing flags')
            assert(np.all(data1['image'] == data2['image']))
            # assert(np.all(data1['boxes'] == data2['boxes']))
            if np.all(data1['flags'] == data2['flags']):
                print('same')
            else:
                print('diff')


if __name__ == '__main__':
    test_new_kitti()
