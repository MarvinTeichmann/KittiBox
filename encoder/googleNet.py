from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from utils import googlenet_load

encoder_net = []


def inference(hypes, images, phase):
    # Load googlenet and returns the cnn_codes

    if phase == 'train':
        encoder_net.append(googlenet_load.init(hypes))

    input_mean = 117.
    images -= input_mean
    cnn, early_feat, _ = googlenet_load.model(images, encoder_net[0], hypes)

    return cnn, early_feat, _
