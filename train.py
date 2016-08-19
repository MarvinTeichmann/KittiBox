#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the MediSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

import time

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

import string

import imp


flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', 'TensorDetect',
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', 'hypes/kitti_yolo.json',
                    'File storing model parameters.')

if 'TV_SAVE' in os.environ and os.environ['TV_SAVE']:
    tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug, '
                       'hence it will get overwritten by further runs.'))
else:
    tf.app.flags.DEFINE_boolean(
        'save', False, ('Whether to save the run. In case --nosave (default) '
                        'output will be saved to the folder TV_DIR_RUNS/debug '
                        'hence it will get overwritten by further runs.'))


def build(hypes, q, modules):
    '''
    Build full model for training, including forward / backward passes,
    optimizers, and summary statistics.
    '''

    data_input, encoder, objective, optimizer = modules

    learning_rate = tf.placeholder(tf.float32)

    images, labels, decoded_logits, losses = {}, {}, {}, {}
    for phase in ['train', 'val']:
        # Load images and Labels
        images[phase], labels[phase] = data_input.inputs(hypes, q[phase],
                                                         phase)

        # Run inference on the encoder network
        logits = encoder.inference(hypes, images[phase], phase)

        # Build decoder on top of the logits
        decoded_logits[phase] = objective.decoder(hypes, logits, phase)

        # Compute losses
        losses[phase] = objective.loss(hypes, decoded_logits[phase],
                                       labels[phase], phase)

    global_step = tf.Variable(0, trainable=False)

    # Build training operation
    train_op = optimizer.training(hypes, losses['train'][0],
                                  global_step, learning_rate)

    # Write Values to summary
    accuracy, smooth_op = objective.evaluation(
        hypes, images, labels, decoded_logits, losses, global_step)

    summary_op = tf.merge_all_summaries()

    return (losses, accuracy, summary_op, train_op,
            smooth_op, global_step, learning_rate)


def build_inference(hypes, modules, image):
    """Run one evaluation against the full epoch of data.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    modules : tuble
        the modules load in utils
    image : placeholder
    label : placeholder

    return:
        graph_ops
    """
    data_input, arch, objective, solver = modules

    logits = arch.inference(hypes, image, phase='val')

    decoded_logits = objective.decoder(hypes, logits, phase='val')

    (pred_boxes, pred_logits, pred_confidences,
     pred_confs_deltas, pred_boxes_deltas) = decoded_logits

    return pred_boxes, pred_confidences


def my_training(hypes):
    modules = utils.load_modules_from_hypes(hypes)
    data_input, arch, objective, solver = modules

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        q = {}
        enqueue_op = {}
        for phase in ['train', 'val']:
            q[phase] = data_input.create_queues(hypes, phase)

        # build the graph based on the loaded modules
        graph_ops = build(hypes, q, modules)
        (losses, accuracy, summary_op, train_op,
            smooth_op, global_step, learning_rate) = graph_ops

        total_loss = losses['train'][0]

        # prepaire the tv session
        sess_coll = core.start_tv_session(hypes)
        sess, saver, summary_op, summary_writer, coord, threads = sess_coll

        with tf.name_scope('Validation'):
            f = os.path.join(hypes['dirs']['base_path'],
                             hypes['model']['evaluator_file'])
            evaluator = imp.load_source("evaluator", f)
            image_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(image_pl, 0)
            inference_out = build_inference(hypes, modules,
                                            image=image)

        train._start_enqueuing_threads(hypes, q, sess, data_input)

        start = time.time()
        max_iter = hypes['solver'].get('max_steps', 10000000)
        display_iter = hypes['logging']['display_iter']
        eval_iter = hypes['logging']['eval_iter']

        for i in xrange(max_iter):
            adjusted_lr = solver.get_learning_rate(hypes, i)

            lr_feed = {learning_rate: adjusted_lr}

            if i % display_iter != 0:
                # train network
                batch_loss_train, _ = sess.run([total_loss, train_op],
                                               feed_dict=lr_feed)
            else:
                # test network every N iterations; log additional info
                if i > 0:
                    dt = (time.time() - start) /\
                         (hypes['batch_size'] * display_iter)
                (train_loss, test_accuracy, summary_str,
                    _, _) = sess.run([total_loss, accuracy['val'],
                                      summary_op, train_op, smooth_op,
                                      ], feed_dict=lr_feed)
                summary_writer.add_summary(summary_str, global_step=i)
                print_str = string.join([
                    'Step: %d',
                    'lr: %f',
                    'Train Loss: %.2f',
                    'Test Accuracy: %.1f%%',
                    'Time/image (ms): %.1f'
                ], ', ')

                logging.info(print_str %
                             (i, adjusted_lr, train_loss,
                              test_accuracy * 100, dt * 1000 if i > 0 else 0))
                if i % eval_iter == 0:
                    train._do_python_evaluation(hypes, i, sess_coll, evaluator,
                                                image_pl, inference_out)
                start = time.time()
            g_step = global_step.eval(session=sess)
            checkpoint_path = os.path.join(hypes['dirs']['output_dir'],
                                           'model.ckpt')
            if g_step % hypes['logging']['save_iter'] == 0 \
                    or g_step == max_iter - 1:
                        saver.save(sess, checkpoint_path,
                                   global_step=global_step)


def main(_):
    utils.set_gpus_to_use()

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'])

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    train.maybe_download_and_extract(hypes)
    logging.info("Start training")
    my_training(hypes)


if __name__ == '__main__':
    tf.app.run()
