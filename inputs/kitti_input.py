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

from scipy.misc import imread, imresize

import tensorflow as tf

from utils.data_utils import (annotation_jitter, annotation_to_h5)
from utils.annolist import AnnotationLib as AnnoLib
from utils.rect import Rect

import threading

from collections import namedtuple
fake_anno = namedtuple('fake_anno_object', ['rects'])


def read_kitti_anno(label_file, detect_truck):
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
                label[0] == 'Truck' or label[0] == 'DontCare'):
            continue
        notruck = not detect_truck
        if notruck and label[0] == 'Truck':
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


def _rescale_boxes(current_shape, anno, target_height, target_width):
    x_scale = target_width / float(current_shape[1])
    y_scale = target_height / float(current_shape[0])
    for r in anno.rects:
        assert r.x1 < r.x2
        r.x1 *= x_scale
        r.x2 *= x_scale
        assert r.x1 < r.x2
        r.y1 *= y_scale
        r.y2 *= y_scale
    return anno


def _generate_mask(hypes, ignore_rects):

    width = hypes["image_width"]
    height = hypes["image_height"]
    grid_width = hypes["grid_width"]
    grid_height = hypes["grid_height"]

    mask = np.ones([grid_height, grid_width])

    if not hypes['use_mask']:
        return mask

    for rect in ignore_rects:
        left = int((rect.x1+2)/width*grid_width)
        right = int((rect.x2-2)/width*grid_width)
        top = int((rect.y1+2)/height*grid_height)
        bottom = int((rect.y2-2)/height*grid_height)
        for x in range(left, right+1):
            for y in range(top, bottom+1):
                mask[y, x] = 0

    return mask


def _load_kitti_txt(kitti_txt, hypes, jitter=False, random_shuffel=True):
    """Take the txt file and net configuration and create a generator
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

            rect_list = read_kitti_anno(gt_image_file,
                                        detect_truck=hypes['detect_truck'])

            anno = AnnoLib.Annotation()
            anno.rects = rect_list

            im = scp.misc.imread(image_file)
            if im.shape[2] == 4:
                im = im[:, :, :3]
            if im.shape[0] != hypes["image_height"] or \
               im.shape[1] != hypes["image_width"]:
                if True:
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
            pos_anno = fake_anno(pos_list)

            boxes, confs = annotation_to_h5(hypes,
                                            pos_anno,
                                            hypes["grid_width"],
                                            hypes["grid_height"],
                                            hypes["rnn_len"])

            mask_list = [rect for rect in anno.rects if rect.classID == -1]
            mask = _generate_mask(hypes, mask_list)

            boxes = boxes.reshape([hypes["grid_height"],
                                   hypes["grid_width"], 4])
            confs = confs.reshape(hypes["grid_height"], hypes["grid_width"])

            yield {"image": im, "boxes": boxes, "confs": confs,
                   "rects": pos_list, "mask": mask}


def _make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v


def create_queues(hypes, phase):
    """Create Queues."""
    hypes["rnn_len"] = 1
    dtypes = [tf.float32, tf.float32, tf.float32, tf.float32]
    grid_size = hypes['grid_width'] * hypes['grid_height']
    shapes = ([hypes['image_height'], hypes['image_width'], 3],
              [hypes['grid_height'], hypes['grid_width']],
              [hypes['grid_height'], hypes['grid_width'], 4],
              [hypes['grid_height'], hypes['grid_width']])
    capacity = 30
    q = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes)
    return q


def _processe_image(hypes, image):
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    augment_level = hypes['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)
        image = tf.image.random_hue(image, max_delta=0.15)

    image = tf.minimum(image, 255.0)
    image = tf.maximum(image, 0)

    return image


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""

    # Creating Placeholder for the Queue
    x_in = tf.placeholder(tf.float32)
    confs_in = tf.placeholder(tf.float32)
    boxes_in = tf.placeholder(tf.float32)
    mask_in = tf.placeholder(tf.float32)

    # Creating Enqueue OP
    enqueue_op = q.enqueue((x_in, confs_in, boxes_in, mask_in))

    def make_feed(data):
        return {x_in: data['image'],
                confs_in: data['confs'],
                boxes_in: data['boxes'],
                mask_in: data['mask']}

    def thread_loop(sess, enqueue_op, gen):
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    data_file = hypes["data"]['%s_file' % phase]
    data_dir = hypes['dirs']['data_dir']
    data_file = os.path.join(data_dir, data_file)

    gen = _load_kitti_txt(data_file, hypes,
                          jitter={'train': hypes['solver']['use_jitter'],
                                  'val': False}[phase])

    data = gen.next()
    sess.run(enqueue_op, feed_dict=make_feed(data))
    t = threading.Thread(target=thread_loop,
                         args=(sess, enqueue_op, gen))
    t.daemon = True
    t.start()


def test_new_kitti():
    idlfile = "/home/mifs/mttt2/cvfs/DATA/KittiBox/train_3.idl"
    kitti_txt = "/home/mifs/mttt2/cvfs/DATA/KittiBox/train.txt"

    with open('hypes/kittiBox.json', 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    hypes["rnn_len"] = 1
    hypes["image_height"] = 200
    hypes["image_width"] = 800

    gen1 = _load_kitti_txt(kitti_txt, hypes, random_shuffel=False)
    gen2 = _load_kitti_txt(idlfile, hypes, random_shuffel=False)

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


def inputs(hypes, q, phase):

    if phase == 'val':
        image, confidences, boxes, mask = q.dequeue()
        image = tf.expand_dims(image, 0)
        confidences = tf.expand_dims(confidences, 0)
        boxes = tf.expand_dims(boxes, 0)
        mask = tf.expand_dims(mask, 0)
        return image, (confidences, boxes, mask)
    elif phase == 'train':
        image, confidences, boxes, mask = q.dequeue_many(hypes['batch_size'])
        image = _processe_image(hypes, image)
        return image, (confidences, boxes, mask)
    else:
        assert("Bad phase: {}".format(phase))
