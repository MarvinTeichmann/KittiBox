#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the fastbox decoder. For a detailed description see:
https://arxiv.org/abs/1612.07695 ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random

from utils import train_utils

import tensorflow as tf


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
    interp_indices = tf.concat(axis=0, values=indices)
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


def _build_inner_layer(hyp, encoded_features, train):
    '''
    Apply an 1x1 convolutions to compute inner features
    The layer consists of 1x1 convolutions implemented as
    matrix multiplication. This makes the layer very fast.
    The layer has "hyp['num_inner_channel']" channels
    '''
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']

    num_ex = hyp['batch_size'] * hyp['grid_width'] * hyp['grid_height']

    channels = int(encoded_features.shape[-1])
    hyp['cnn_channels'] = channels
    hidden_input = tf.reshape(encoded_features, [num_ex, channels])

    scale_down = hyp['scale_down']

    hidden_input = tf.reshape(
        hidden_input * scale_down, (hyp['batch_size'] * grid_size, channels))

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope('Overfeat', initializer=initializer):
        w = tf.get_variable('ip', shape=[hyp['cnn_channels'],
                                         hyp['num_inner_channel']])
        output = tf.matmul(hidden_input, w)

    if train:
        # Adding dropout during training
            output = tf.nn.dropout(output, 0.5)
    return output


def _build_output_layer(hyp, hidden_output):
    '''
    Build an 1x1 conv layer.
    The layer consists of 1x1 convolutions implemented as
    matrix multiplication. This makes the layer very fast.
    The layer has "hyp['num_inner_channel']" channels
    '''

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']

    box_weights = tf.get_variable('box_out',
                                  shape=(hyp['num_inner_channel'], 4))
    conf_weights = tf.get_variable('confs_out',
                                   shape=(hyp['num_inner_channel'],
                                          hyp['num_classes']))

    pred_boxes = tf.reshape(tf.matmul(hidden_output, box_weights) * 50,
                            [outer_size, 1, 4])

    # hyp['rnn_len']
    pred_logits = tf.reshape(tf.matmul(hidden_output, conf_weights),
                             [outer_size, 1, hyp['num_classes']])

    pred_logits_squash = tf.reshape(pred_logits,
                                    [outer_size,
                                     hyp['num_classes']])

    pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
    pred_confidences = tf.reshape(pred_confidences_squash,
                                  [outer_size, hyp['rnn_len'],
                                   hyp['num_classes']])
    return pred_boxes, pred_logits, pred_confidences


def _build_rezoom_layer(hyp, rezoom_input, train):

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']

    pred_boxes, pred_logits, pred_confidences, early_feat, \
        hidden_output = rezoom_input

    early_feat_channels = hyp['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]

    w_offsets = hyp['rezoom_w_coords']
    h_offsets = hyp['rezoom_h_coords']
    num_offsets = len(w_offsets) * len(h_offsets)
    rezoom_features = _rezoom(
        hyp, pred_boxes, early_feat, early_feat_channels,
        w_offsets, h_offsets)
    if train:
        rezoom_features = tf.nn.dropout(rezoom_features, 0.5)

    delta_features = tf.concat(
        axis=1,
        values=[hidden_output,
                rezoom_features[:, 0, :] / 1000.])
    dim = 128
    shape = [hyp['num_inner_channel'] +
             early_feat_channels * num_offsets,
             dim]
    delta_weights1 = tf.get_variable('delta1',
                                     shape=shape)
    # TODO: maybe adding dropout here?
    ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
    if train:
        ip1 = tf.nn.dropout(ip1, 0.5)
    delta_confs_weights = tf.get_variable(
        'delta2', shape=[dim, hyp['num_classes']])
    delta_boxes_weights = tf.get_variable('delta_boxes', shape=[dim, 4])

    rere_feature = tf.matmul(ip1, delta_boxes_weights) * 5
    pred_boxes_delta = (tf.reshape(rere_feature, [outer_size, 1, 4]))

    scale = hyp.get('rezoom_conf_scale', 50)
    feature2 = tf.matmul(ip1, delta_confs_weights) * scale
    pred_confs_delta = tf.reshape(feature2, [outer_size, 1,
                                  hyp['num_classes']])

    pred_confs_delta = tf.reshape(pred_confs_delta,
                                  [outer_size, hyp['num_classes']])

    pred_confidences_squash = tf.nn.softmax(pred_confs_delta)
    pred_confidences = tf.reshape(pred_confidences_squash,
                                  [outer_size, hyp['rnn_len'],
                                   hyp['num_classes']])

    return pred_boxes, pred_logits, pred_confidences, \
        pred_confs_delta, pred_boxes_delta


def decoder(hyp, logits, train):
    """Apply decoder to the logits.

    Computation which decode CNN boxes.
    The output can be interpreted as bounding Boxes.


    Args:
      logits: Logits tensor, output von encoder

    Return:
      decoded_logits: values which can be interpreted as bounding boxes
    """
    hyp['rnn_len'] = 1
    encoded_features = logits['deep_feat']

    early_feat = logits['early_feat']

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.variable_scope('decoder', initializer=initializer):
        # Build inner layer.
        # See https://arxiv.org/abs/1612.07695 fig. 2 for details
        hidden_output = _build_inner_layer(hyp, encoded_features, train)
        # Build output layer
        # See https://arxiv.org/abs/1612.07695 fig. 2 for details
        pred_boxes, pred_logits, pred_confidences = _build_output_layer(
            hyp, hidden_output)

        # Dictionary filled with return values
        dlogits = {}

        if hyp['use_rezoom']:
            rezoom_input = pred_boxes, pred_logits, pred_confidences, \
                early_feat, hidden_output
            # Build rezoom layer
            # See https://arxiv.org/abs/1612.07695 fig. 2 for details
            rezoom_output = _build_rezoom_layer(hyp, rezoom_input, train)

            pred_boxes, pred_logits, pred_confidences, \
                pred_confs_deltas, pred_boxes_deltas = rezoom_output

            dlogits['pred_confs_deltas'] = pred_confs_deltas
            dlogits['pred_boxes_deltas'] = pred_boxes_deltas

            dlogits['pred_boxes_new'] = pred_boxes + pred_boxes_deltas

    # Fill dict with return values
    dlogits['pred_boxes'] = pred_boxes
    dlogits['pred_logits'] = pred_logits
    dlogits['pred_confidences'] = pred_confidences

    return dlogits


def _add_rezoom_loss_histograms(hypes, pred_boxes_deltas):
    """
    Add some histograms to tensorboard.
    """
    tf.summary.histogram(
        '/delta_hist0_x', pred_boxes_deltas[:, 0, 0])
    tf.summary.histogram(
        '/delta_hist0_y', pred_boxes_deltas[:, 0, 1])
    tf.summary.histogram(
        '/delta_hist0_w', pred_boxes_deltas[:, 0, 2])
    tf.summary.histogram(
        '/delta_hist0_h', pred_boxes_deltas[:, 0, 3])


def _compute_rezoom_loss(hypes, rezoom_loss_input):
    """
    Computes loss for delta output. Only relevant
    if rezoom layers are used.
    """
    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']
    head = hypes['solver']['head_weights']

    perm_truth, pred_boxes, classes, pred_mask, \
        pred_confs_deltas, pred_boxes_deltas = rezoom_loss_input
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
        logits=pred_confs_deltas, labels=inside)

    delta_confs_loss = tf.reduce_sum(cross_entropy) \
        / outer_size * hypes['solver']['head_weights'][0] * 0.1

    delta_unshaped = perm_truth - (pred_boxes + pred_boxes_deltas)

    delta_residual = tf.reshape(delta_unshaped * pred_mask,
                                [outer_size, hypes['rnn_len'], 4])
    sqrt_delta = tf.minimum(tf.square(delta_residual), 10. ** 2)
    delta_boxes_loss = (tf.reduce_sum(sqrt_delta) /
                        outer_size * head[1] * 0.03)

    return delta_confs_loss, delta_boxes_loss


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      decoded_logits: output of decoder
      labels: Labels tensor; Output from data_input

    Returns:
      loss: Loss tensor of type float.
    """

    flags, confidences, boxes = labels

    pred_boxes = decoded_logits['pred_boxes']
    pred_logits = decoded_logits['pred_logits']
    pred_confidences = decoded_logits['pred_confidences']

    pred_confs_deltas = decoded_logits['pred_confs_deltas']
    pred_boxes_deltas = decoded_logits['pred_boxes_deltas']

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']

    head = hypes['solver']['head_weights']

    # Compute confidence loss
    classes = tf.reshape(flags, (outer_size, 1))
    true_classes = tf.reshape(tf.cast(tf.greater(classes, 0), 'int64'),
                              [outer_size * hypes['rnn_len']])

    pred_classes = tf.reshape(pred_logits,
                              [outer_size * hypes['rnn_len'],
                               hypes['num_classes']])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_classes, labels=true_classes)

    cross_entropy_sum = (tf.reduce_sum(cross_entropy))
    confidences_loss = cross_entropy_sum / outer_size * head[0]

    true_boxes = tf.reshape(boxes, (outer_size, hypes['rnn_len'], 4))

    # box loss for background prediction needs to be zerod out
    boxes_mask = tf.reshape(
        tf.cast(tf.greater(classes, 0), 'float32'), (outer_size, 1, 1))

    # danger zone
    residual = (true_boxes - pred_boxes) * boxes_mask

    boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * head[1]

    if hypes['use_rezoom']:
        # add rezoom loss
        rezoom_loss_input = true_boxes, pred_boxes, classes, boxes_mask, \
            pred_confs_deltas, pred_boxes_deltas

        delta_confs_loss, delta_boxes_loss = _compute_rezoom_loss(
            hypes, rezoom_loss_input)

        _add_rezoom_loss_histograms(hypes, pred_boxes_deltas)

        loss = confidences_loss + boxes_loss + delta_boxes_loss \
            + delta_confs_loss
    else:
        loss = confidences_loss + boxes_loss

    tf.add_to_collection('total_losses', loss)

    weight_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    total_loss = weight_loss + loss

    losses = {}
    losses['total_loss'] = total_loss
    losses['loss'] = loss
    losses['confidences_loss'] = confidences_loss
    losses['boxes_loss'] = boxes_loss
    losses['weight_loss'] = weight_loss
    if hypes['use_rezoom']:
        losses['delta_boxes_loss'] = delta_boxes_loss
        losses['delta_confs_loss'] = delta_confs_loss

    return losses


def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
    """
    Compute summary metrics for tensorboard
    """

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
    if hyp['use_rezoom']:
        eval_list.append(('Delta', losses['delta_boxes_loss'] +
                          losses['delta_confs_loss']))

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
            ((np_global_step // hyp['logging']['write_iter'])
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
    tf.summary.image('/pred_boxes', tf.stack(pred_log_img))
    tf.summary.image('/true_boxes', tf.stack(true_log_img))
    return eval_list
