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

try:
    from tensorflow.models.rnn import rnn_cell
except ImportError:
    rnn_cell = tf.nn.rnn_cell


def _deconv(x, output_shape, channels):
    k_h = 2
    k_w = 2
    w = tf.get_variable('w_deconv',
                        initializer=tf.random_normal_initializer(stddev=0.01),
                        shape=[k_h, k_w, channels[1], channels[0]])
    y = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, k_h, k_w, 1],
                               padding='VALID')
    return y


def _rezoom(hyp, pred_boxes, early_feat, early_feat_channels,
            w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points
    in a grid.

    If the predicted object center is at X, len(w_offsets) == 3,
    and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    indices = []
    for w_offset in w_offsets:
        for h_offset in h_offsets:
            indices.append(train_utils.bilinear_select(hyp,
                                                       pred_boxes,
                                                       early_feat,
                                                       early_feat_channels,
                                                       w_offset, h_offset))

    interp_indices = tf.concat(0, indices)
    rezoom_features = train_utils.interp(early_feat,
                                         interp_indices,
                                         early_feat_channels)
    rezoom_features_r = tf.reshape(rezoom_features,
                                   [len(w_offsets) * len(h_offsets),
                                    outer_size,
                                    hyp['rnn_len'],
                                    early_feat_channels])
    rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
    return tf.reshape(rezoom_features_t,
                      [outer_size,
                       hyp['rnn_len'],
                       len(w_offsets) * len(h_offsets) * early_feat_channels])


def _build_lstm_inner(hyp, lstm_input):
    '''
    build lstm decoder
    '''
    lstm_cell = rnn_cell.BasicLSTMCell(hyp['lstm_size'], forget_bias=0.0)
    if hyp['num_lstm_layers'] > 1:
        lstm = rnn_cell.MultiRNNCell([lstm_cell] * hyp['num_lstm_layers'])
    else:
        lstm = lstm_cell

    batch_size = hyp['batch_size'] * hyp['grid_height'] * hyp['grid_width']
    state = tf.zeros([batch_size, lstm.state_size])

    outputs = []
    with tf.variable_scope(
            'RNN', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        for time_step in range(hyp['rnn_len']):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = lstm(lstm_input, state)
            outputs.append(output)
    return outputs


def _build_overfeat_inner(hyp, lstm_input):
    '''
    build simple overfeat decoder
    '''
    if hyp['rnn_len'] > 1:
        raise ValueError('rnn_len > 1 only supported with use_lstm == True')
    outputs = []
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[1024, hyp['lstm_size']])
        outputs.append(tf.matmul(lstm_input, w))
    return outputs


def decoder(hyp, logits, phase):
    """Apply decoder to the logits.

    Computation which decode CNN boxes.
    The output can be interpreted as bounding Boxes.


    Args:
      logits: Logits tensor, output von encoder

    Return:
      decoded_logits: values which can be interpreted as bounding boxes
    """
    cnn, early_feat, _ = logits

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    reuse = {'train': None, 'test': True}[phase]

    early_feat_channels = hyp['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    if hyp['deconv']:
        size = 3
        stride = 2
        pool_size = 5

        with tf.variable_scope("deconv", reuse=reuse):
            initializer = tf.random_normal_initializer(stddev=0.01)
            w = tf.get_variable('conv_pool_w', shape=[size, size, 1024, 1024],
                                initializer=initializer)
            cnn_s = tf.nn.conv2d(cnn, w, strides=[1, stride, stride, 1],
                                 padding='SAME')
            cnn_s_pool = tf.nn.avg_pool(cnn_s[:, :, :, :256],
                                        ksize=[1, pool_size, pool_size, 1],
                                        strides=[1, 1, 1, 1], padding='SAME')

            cnn_s_with_pool = tf.concat(3, [cnn_s_pool, cnn_s[:, :, :, 256:]])
            output_shape = [hyp['batch_size'], hyp['grid_height'],
                            hyp['grid_width'], 256]
            cnn_deconv = _deconv(
                cnn_s_with_pool, output_shape=output_shape,
                channels=[1024, 256])
            cnn = tf.concat(3, (cnn_deconv, cnn[:, :, :, 256:]))

    elif hyp['avg_pool_size'] > 1:
        pool_size = hyp['avg_pool_size']
        cnn1 = cnn[:, :, :, :700]
        cnn2 = cnn[:, :, :, 700:]
        cnn2 = tf.nn.avg_pool(cnn2, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
        cnn = tf.concat(3, [cnn1, cnn2])

    num_ex = hyp['batch_size'] * hyp['grid_width'] * hyp['grid_height']

    cnn = tf.reshape(cnn, [num_ex, 1024])
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('decoder', reuse=reuse, initializer=initializer):
        scale_down = 0.01
        lstm_input = tf.reshape(
            cnn * scale_down, (hyp['batch_size'] * grid_size, 1024))
        if hyp['use_lstm']:
            lstm_outputs = _build_lstm_inner(hyp, lstm_input)
        else:
            lstm_outputs = _build_overfeat_inner(hyp, lstm_input)

        pred_boxes = []
        pred_logits = []
        for k in range(hyp['rnn_len']):
            output = lstm_outputs[k]
            if phase == 'train':
                output = tf.nn.dropout(output, 0.5)
            box_weights = tf.get_variable('box_ip%d' % k,
                                          shape=(hyp['lstm_size'], 4))
            conf_weights = tf.get_variable('conf_ip%d' % k,
                                           shape=(hyp['lstm_size'],
                                                  hyp['num_classes']))

            pred_boxes_step = tf.reshape(tf.matmul(output, box_weights) * 50,
                                         [outer_size, 1, 4])

            pred_boxes.append(pred_boxes_step)
            pred_logits.append(tf.reshape(tf.matmul(output, conf_weights),
                                          [outer_size, 1, hyp['num_classes']]))

        pred_boxes = tf.concat(1, pred_boxes)
        pred_logits = tf.concat(1, pred_logits)
        pred_logits_squash = tf.reshape(pred_logits,
                                        [outer_size * hyp['rnn_len'],
                                         hyp['num_classes']])
        pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, hyp['rnn_len'],
                                       hyp['num_classes']])

        if hyp['use_rezoom']:
            pred_confs_deltas = []
            pred_boxes_deltas = []
            w_offsets = hyp['rezoom_w_coords']
            h_offsets = hyp['rezoom_h_coords']
            num_offsets = len(w_offsets) * len(h_offsets)
            rezoom_features = _rezoom(
                hyp, pred_boxes, early_feat, early_feat_channels,
                w_offsets, h_offsets)
            if phase == 'train':
                rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
            for k in range(hyp['rnn_len']):
                delta_features = tf.concat(
                    1, [lstm_outputs[k], rezoom_features[:, k, :] / 1000.])
                dim = 128
                shape = [hyp['lstm_size'] + early_feat_channels * num_offsets,
                         dim]
                delta_weights1 = tf.get_variable('delta_ip1%d' % k,
                                                 shape=shape)
                # TODO: add dropout here ?
                ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
                if phase == 'train':
                    ip1 = tf.nn.dropout(ip1, 0.5)
                delta_confs_weights = tf.get_variable(
                    'delta_ip2%d' % k,
                    shape=[dim, hyp['num_classes']])
                if hyp['reregress']:
                    delta_boxes_weights = tf.get_variable(
                        'delta_ip_boxes%d' % k,
                        shape=[dim, 4])
                    rere_feature = tf.matmul(ip1, delta_boxes_weights) * 5
                    pred_boxes_deltas.append(tf.reshape(rere_feature,
                                                        [outer_size, 1, 4]))
                scale = hyp.get('rezoom_conf_scale', 50)
                feature2 = tf.matmul(ip1, delta_confs_weights) * scale
                pred_confs_deltas.append(tf.reshape(feature2,
                                                    [outer_size, 1,
                                                     hyp['num_classes']]))
            pred_confs_deltas = tf.concat(1, pred_confs_deltas)

            # moved from loss
            pred_confs_deltas = tf.reshape(pred_confs_deltas,
                                           [outer_size * hyp['rnn_len'],
                                            hyp['num_classes']])

            pred_logits_squash = tf.reshape(pred_confs_deltas,
                                            [outer_size * hyp['rnn_len'],
                                             hyp['num_classes']])
            pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
            pred_confidences = tf.reshape(pred_confidences_squash,
                                          [outer_size, hyp['rnn_len'],
                                           hyp['num_classes']])
            if hyp['reregress']:
                pred_boxes_deltas = tf.concat(1, pred_boxes_deltas)
        else:
            pred_confs_deltas = None
            pred_boxes_deltas = None

    logits = pred_boxes, pred_logits, pred_confidences,\
        pred_confs_deltas, pred_boxes_deltas

    return logits


def loss(hypes, decoded_logits, labels, phase):
    """Calculate the loss from the logits and the labels.

    Args:
      decoded_logits: output of decoder
      labels: Labels tensor; Output from data_input

    Returns:
      loss: Loss tensor of type float.
    """

    flags, confidences, boxes = labels

    (pred_boxes, pred_logits, pred_confidences,
     pred_confs_deltas, pred_boxes_deltas) = decoded_logits

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']

    with tf.variable_scope('decoder',
                           reuse={'train': None, 'test': True}[phase]):
        outer_boxes = tf.reshape(boxes, [outer_size, hypes['rnn_len'], 4])
        outer_flags = tf.cast(
            tf.reshape(flags, [outer_size, hypes['rnn_len']]), 'int32')
        if hypes['use_lstm']:
            assignments, classes, perm_truth, pred_mask = (
                tf.user_ops.hungarian(pred_boxes, outer_boxes, outer_flags,
                                      hypes['solver']['hungarian_iou']))
        else:
            classes = tf.reshape(flags, (outer_size, 1))
            perm_truth = tf.reshape(outer_boxes, (outer_size, 1, 4))
            pred_mask = tf.reshape(
                tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))
        true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                                  [outer_size * hypes['rnn_len']])
        pred_logit_r = tf.reshape(pred_logits,
                                  [outer_size * hypes['rnn_len'],
                                   hypes['num_classes']])

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        pred_logit_r, true_classes)

    cross_entropy_sum = (tf.reduce_sum(cross_entropy))

    head = hypes['solver']['head_weights']
    confidences_loss = cross_entropy_sum / outer_size * head[0]
    residual = tf.reshape(perm_truth - pred_boxes * pred_mask,
                          [outer_size, hypes['rnn_len'], 4])

    boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * head[1]
    if hypes['use_rezoom']:
        if hypes['rezoom_change_loss'] == 'center':
            error = (perm_truth[:, :, 0:2] - pred_boxes[:, :, 0:2]) \
                / tf.maximum(perm_truth[:, :, 2:4], 1.)
            square_error = tf.reduce_sum(tf.square(error), 2)
            inside = tf.reshape(tf.to_int64(
                tf.logical_and(tf.less(square_error, 0.2**2),
                               tf.greater(classes, 0))), [-1])
        elif hypes['rezoom_change_loss'] == 'iou':
            pred_boxes_flat = tf.reshape(pred_boxes, [-1, 4])
            perm_truth_flat = tf.reshape(perm_truth, [-1, 4])
            iou = train_utils.iou(train_utils.to_x1y1x2y2(pred_boxes_flat),
                                  train_utils.to_x1y1x2y2(perm_truth_flat))
            inside = tf.reshape(tf.to_int64(tf.greater(iou, 0.5)), [-1])
        else:
            assert not hypes['rezoom_change_loss']
            inside = tf.reshape(tf.to_int64((tf.greater(classes, 0))), [-1])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            pred_confs_deltas, inside)

        delta_confs_loss = tf.reduce_sum(cross_entropy) \
            / outer_size * hypes['solver']['head_weights'][0] * 0.1

        loss = confidences_loss + boxes_loss + delta_confs_loss
        if hypes['reregress']:
            delta_unshaped = perm_truth - (pred_boxes + pred_boxes_deltas)

            delta_residual = tf.reshape(delta_unshaped * pred_mask,
                                        [outer_size, hypes['rnn_len'], 4])
            sqrt_delta = tf.minimum(tf.square(delta_residual), 10. ** 2)
            delta_boxes_loss = (tf.reduce_sum(sqrt_delta) /
                                outer_size * head[1] * 0.03)
            boxes_loss = delta_boxes_loss

            tf.histogram_summary(
                phase + '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
            tf.histogram_summary(
                phase + '/delta_hist0_y', pred_boxes_deltas[:, 0, 1])
            tf.histogram_summary(
                phase + '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
            tf.histogram_summary(
                phase + '/delta_hist0_h', pred_boxes_deltas[:, 0, 3])
            loss += delta_boxes_loss
    else:
        loss = confidences_loss + boxes_loss

    return loss, confidences_loss, boxes_loss


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):

    loss, accuracy, confidences_loss, boxes_loss = {}, {}, {}, {}

    # Estimating Accuracy
    for phase in ['train', 'test']:

        flags, confidences, boxes = labels[phase]
        loss[phase], confidences_loss[phase], boxes_loss[phase] = losses[phase]

        (pred_boxes, pred_logits, pred_confidences,
         pred_confs_deltas, pred_boxes_deltas) = decoded_logits[phase]

        grid_size = hyp['grid_width'] * hyp['grid_height']

        pred_confidences_r = tf.reshape(
            pred_confidences,
            [hyp['batch_size'], grid_size, hyp['rnn_len'], hyp['num_classes']])
        pred_boxes_r = tf.reshape(
            pred_boxes, [hyp['batch_size'], grid_size, hyp['rnn_len'], 4])

        # Set up summary operations for tensorboard
        a = tf.equal(tf.argmax(confidences[:, :, 0, :], 2), tf.argmax(
            pred_confidences_r[:, :, 0, :], 2))
        accuracy[phase] = tf.reduce_mean(
            tf.cast(a, 'float32'), name=phase+'/accuracy')

    # Writing Metrics to Tensorboard

    moving_avg = tf.train.ExponentialMovingAverage(0.95)
    smooth_op = moving_avg.apply([accuracy['train'], accuracy['test'],
                                  confidences_loss[
                                      'train'], boxes_loss['train'],
                                  confidences_loss[
                                      'test'], boxes_loss['test'],
                                  ])

    for p in ['train', 'test']:
        tf.scalar_summary('%s/accuracy' % p, accuracy[p])
        tf.scalar_summary('%s/accuracy/smooth' %
                          p, moving_avg.average(accuracy[p]))
        tf.scalar_summary("%s/confidences_loss" % p, confidences_loss[p])
        tf.scalar_summary("%s/confidences_loss/smooth" % p,
                          moving_avg.average(confidences_loss[p]))
        tf.scalar_summary("%s/regression_loss" % p, boxes_loss[p])
        tf.scalar_summary("%s/regression_loss/smooth" % p,
                          moving_avg.average(boxes_loss[p]))

    test_image = images['test']
    # show ground truth to verify labels are correct
    test_true_confidences = confidences[0, :, :, :]
    test_true_boxes = boxes[0, :, :, :]

    # show predictions to visualize training progress
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
        img_path = os.path.join(hyp['save_dir'], filename)

        scp.misc.imsave(img_path, merged)
        return merged

    pred_log_img = tf.py_func(log_image,
                              [test_image, test_pred_confidences,
                               test_pred_boxes, global_step, 'pred'],
                              [tf.float32])
    true_log_img = tf.py_func(log_image,
                              [test_image, test_true_confidences,
                               test_true_boxes, global_step, 'true'],
                              [tf.float32])
    tf.image_summary(phase + '/pred_boxes', tf.pack(pred_log_img),
                     max_images=10)
    tf.image_summary(phase + '/true_boxes', tf.pack(true_log_img),
                     max_images=10)

    return accuracy, smooth_op
