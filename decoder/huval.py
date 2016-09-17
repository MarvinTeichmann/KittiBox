#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random

from utils import train_utils

import tensorflow as tf


def _build_yolo_fc_layer(hyp, cnn_output):
    '''
    build simple overfeat decoder
    '''
    scale_down = 0.01
    grid_size = hyp['grid_width'] * hyp['grid_height']
    channels = hyp['cnn_channels']
    lstm_input = tf.reshape(cnn_output * scale_down,
                            (hyp['batch_size'] * grid_size, channels))

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[hyp['cnn_channels'],
                                         hyp['lstm_size']])
        return tf.matmul(lstm_input, w)


def _variable_with_weight_decay(shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape,
                          initializer=initializer)

    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _bias_variable(shape, constant=0.0):
    initializer = tf.constant_initializer(constant)
    return tf.get_variable(name='biases', shape=shape,
                           initializer=initializer)


def _score_layer(hypes, bottom, name, num_classes):
    wd = 5e-4
    with tf.variable_scope(name) as scope:
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]
        # He initialization Sheme
        num_input = in_features
        stddev = (2 / num_input)**0.5
        # Apply convolution
        w_decay = wd
        weights = _variable_with_weight_decay(shape, stddev, w_decay)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = _bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        _activation_summary(bias)

        return bias


def _deconv(x, output_shape, channels):
    k_h = 2
    k_w = 2
    w = tf.get_variable('w_deconv',
                        initializer=tf.random_normal_initializer(stddev=0.01),
                        shape=[k_h, k_w, channels[1], channels[0]])
    y = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, k_h, k_w, 1],
                               padding='VALID')
    return y


def decoder(hyp, logits, train):
    """Apply decoder to the logits.

    Computation which decode CNN boxes.
    The output can be interpreted as bounding Boxes.


    Args:
      logits: Logits tensor, output von encoder

    Return:
      decoded_logits: values which can be interpreted as bounding boxes
    """

    scored_feat = logits['scored_feat']
    unpooled = logits['unpooled']

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    early_feat = logits['early_feat']

    size = 3
    stride = 2
    pool_size = 3

    with tf.variable_scope("deconv"):
        initializer = tf.random_normal_initializer(stddev=0.01)
        w = tf.get_variable('conv_pool_w', shape=[size, size,
                                                  hyp['cnn_channels'],
                                                  hyp['cnn_channels']],
                            initializer=initializer)
        cnn_s_pool = tf.nn.max_pool(unpooled,
                                    ksize=[1, pool_size, pool_size, 1],
                                    strides=[1, stride, stride, 1],
                                    padding='SAME')
        output_shape = [hyp['batch_size'], hyp['grid_height'],
                        hyp['grid_width'], hyp['cnn_channels']]
        cnn_deconv = _deconv(
            cnn_s_pool, output_shape=output_shape,
            channels=[hyp['cnn_channels'], hyp['cnn_channels']])

    with tf.variable_scope('huval_fc'):
        relu = tf.nn.relu(cnn_deconv)
        fc7 = _score_layer(hyp, relu, 'fc7', hyp['cnn_channels'])
        fc8 = _score_layer(hyp, relu, 'fc8', 6)

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('huval', initializer=initializer):

        pred_box = tf.reshape(fc8[:, :, :, 0:4],
                              (-1, hyp['rnn_len'], 4))
        nc = hyp['num_classes']
        pred_logits = tf.reshape(fc8[:, :, :, 4:4+nc],
                                 (-1, nc))

        pred_confidences = tf.nn.softmax(pred_logits)

        pred_confidences = tf.reshape(pred_confidences,
                                      [outer_size, hyp['rnn_len'],
                                       hyp['num_classes']])

    dlogits = {}
    dlogits['pred_boxes'] = pred_box
    dlogits['pred_logits'] = pred_logits
    dlogits['pred_confidences'] = pred_confidences

    return dlogits


def _computed_shaped_labels(hypes, labels):
    flags, confidences, boxes = labels

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']
    with tf.variable_scope('Label_Reshape'):
        outer_boxes = tf.reshape(boxes, [outer_size, hypes['rnn_len'], 4])
        outer_flags = tf.cast(
            tf.reshape(flags, [outer_size, hypes['rnn_len']]), 'int32')

        classes = tf.reshape(flags, (outer_size, 1))
        gt_box = tf.reshape(outer_boxes, (outer_size, 1, 4))
        box_mask = tf.reshape(
            tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * hypes['rnn_len']])

    return gt_box, box_mask, true_classes, classes


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      decoded_logits: output of decoder
      labels: Labels tensor; Output from data_input

    Returns:
      loss: Loss tensor of type float.
    """

    pred_box = decoded_logits['pred_boxes']
    pred_logits = decoded_logits['pred_logits']
    pred_confidences = decoded_logits['pred_confidences']

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']

    with tf.name_scope('Loss'):

        gt_box, box_mask, true_classes, classes = \
            _computed_shaped_labels(hypes, labels)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            pred_logits, true_classes)

        head = hypes['solver']['head_weights']

        confidences_loss = tf.reduce_mean(cross_entropy) * head[0]

        residual = tf.reshape(gt_box - pred_box * box_mask,
                              [outer_size, hypes['rnn_len'], 4])

        boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * head[1]

        loss = confidences_loss + boxes_loss

        tf.add_to_collection('losses', loss)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        losses = {}
        losses['total_loss'] = total_loss
        losses['confidences_loss'] = confidences_loss
        losses['boxes_loss'] = boxes_loss
        losses['weight_loss'] = total_loss - loss

    return losses


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):

    pred_confidences = decoded_logits['pred_confidences']
    pred_boxes = decoded_logits['pred_boxes']
    # Estimating Accuracy
    grid_size = hyp['grid_width'] * hyp['grid_height']
    flags, confidences, boxes = labels
    pred_confidences_r = tf.reshape(
        pred_confidences,
        [hyp['batch_size'], grid_size, hyp['rnn_len'], hyp['num_classes']])
    # Set up summary operations for tensorboard
    a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(
        pred_confidences_r[:, :, 0, :], 2))

    accuracy = tf.reduce_mean(tf.cast(a, 'float32'), name='/accuracy')

    eval_list = []
    eval_list.append(('Acc.', accuracy))
    eval_list.append(('Conf', losses['confidences_loss']))
    eval_list.append(('Box', losses['boxes_loss']))
    eval_list.append(('Weight', losses['weight_loss']))

    # Log Images
    # show ground truth to verify labels are correct
    test_true_confidences = confidences[0, :, :, :]
    test_true_boxes = boxes[0, :, :, :]

    # show predictions to visualize training progress
    pred_boxes_r = tf.reshape(
        pred_boxes, [hyp['batch_size'], grid_size, hyp['rnn_len'],
                     4])
    test_pred_confidences = pred_confidences_r[0, :, :, :]
    test_pred_boxes = pred_boxes_r[0, :, :, :]

    def log_image(np_img, np_confidences, np_boxes, np_global_step,
                  pred_or_true):

        merged = train_utils.add_rectangles(hyp, np_img, np_confidences,
                                            np_boxes,
                                            use_stitching=True,
                                            rnn_len=hyp['rnn_len'])[0]

        num_images = 10

        filename = '%s_%s.jpg' % \
            ((np_global_step // hyp['logging']['display_iter'])
                % num_images, pred_or_true)
        img_path = os.path.join(hyp['dirs']['output_dir'], filename)

        scp.misc.imsave(img_path, merged)
        return merged

    pred_log_img = tf.py_func(log_image,
                              [images, test_pred_confidences,
                               test_pred_boxes, global_step, 'pred'],
                              [tf.float32])
    true_log_img = tf.py_func(log_image,
                              [images, test_true_confidences,
                               test_true_boxes, global_step, 'true'],
                              [tf.float32])
    tf.image_summary('/pred_boxes', tf.pack(pred_log_img),
                     max_images=10)
    tf.image_summary('/true_boxes', tf.pack(true_log_img),
                     max_images=10)

    return eval_list


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
