#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Marvin Teichmann


"""
Detects Cars in an image using KittiBox.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiBox weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')
from utils import train_utils as kittibox_utils

try:
    # Check whether setup was done correctly
    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiBox.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiBox.')


default_run = 'KittiBox_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiBox_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return

    import zipfile
    download_name = tv_utils.download(weights_url, runs_dir)

    logging.info("Extracting KittiBox_pretrained.zip")

    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiBox')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image
    image = scp.misc.imread(input_image)
    image = scp.misc.imresize(image, (hypes["image_height"],
                                      hypes["image_width"]),
                              interp='cubic')
    feed = {image_pl: image}

    # Run KittiBox model on image
    pred_boxes = prediction['pred_boxes_new']
    pred_confidences = prediction['pred_confidences']

    (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes,
                                                     pred_confidences],
                                                    feed_dict=feed)

    # Apply non-maximal suppression
    # and draw predictions on the image
    output_image, rectangles = kittibox_utils.add_rectangles(
        hypes, [image], np_pred_confidences,
        np_pred_boxes, show_removed=False,
        use_stitching=True, rnn_len=1,
        min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))

    threshold = 0.5
    accepted_predictions = []
    # removing predictions <= threshold
    for rect in rectangles:
        if rect.score >= threshold:
            accepted_predictions.append(rect)

    print('')
    logging.info("{} Cars detected".format(len(accepted_predictions)))

    # Printing coordinates of predicted rects.
    for i, rect in enumerate(accepted_predictions):
        logging.info("")
        logging.info("Coordinates of Box {}".format(i))
        logging.info("    x1: {}".format(rect.x1))
        logging.info("    x2: {}".format(rect.x2))
        logging.info("    y1: {}".format(rect.y1))
        logging.info("    y2: {}".format(rect.y2))
        logging.info("    Confidence: {}".format(rect.score))

    # save Image
    if FLAGS.output_image is None:
        output_name = input_image.split('.')[0] + '_rects.png'
    else:
        output_name = FLAGS.output_image

    scp.misc.imsave(output_name, output_image)
    logging.info("")
    logging.info("Output image saved to {}".format(output_name))


if __name__ == '__main__':
    tf.app.run()
