from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_fcn import fcn8_vgg

import tensorflow as tf
import os


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], "vgg16.npy")
    vgg_fcn = fcn8_vgg.FCN8VGG(vgg16_npy_path=vgg16_npy_path)

    num_classes = 2  # does not influence training what so ever
    vgg_fcn.wd = hypes['wd']

    vgg_fcn.build(images, train=train, num_classes=num_classes,
                  random_init_fc8=True)

    vgg_dict = {'unpooled': vgg_fcn.conv5_3,
                'deep_feat': vgg_fcn.pool5,
                'deep_feat_channels': 512,
                'early_feat': vgg_fcn.conv4_3,
                'scored_feat': vgg_fcn.score_fr}

    return vgg_dict
