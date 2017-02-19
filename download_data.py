"""Download data relevant to train the KittiSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
import os
import subprocess

import zipfile


from six.moves import urllib
from shutil import copy2

import argparse

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(1, 'incl')

# Please set kitti_data_url to the download link for the Kitti DATA.
#
# You can obtain by going to this website:
# http://www.cvlibs.net/download.php?file=data_road.zip
#
# Replace 'http://kitti.is.tue.mpg.de/kitti/?????????.???' by the
# correct URL.


vgg_url = 'https://dl.dropboxusercontent.com/u/50333326/vgg16.npy'


def get_pathes():
    """
    Get location of `data_dir` and `run_dir'.

    Defaut is ./DATA and ./RUNS.
    Alternativly they can be set by the environoment variabels
    'TV_DIR_DATA' and 'TV_DIR_RUNS'.
    """

    if 'TV_DIR_DATA' in os.environ:
        data_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        data_dir = "DATA"

    if 'TV_DIR_RUNS' in os.environ:
        run_dir = os.path.join(['hypes'], os.environ['TV_DIR_DATA'])
    else:
        run_dir = "RUNS"

    return data_dir, run_dir


def download(url, dest_directory):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    logging.info("Download URL: {}".format(url))
    logging.info("Download DIR: {}".format(dest_directory))

    def _progress(count, block_size, total_size):
                prog = float(count * block_size) / float(total_size) * 100.0
                sys.stdout.write('\r>> Downloading %s %.1f%%' %
                                 (filename, prog))
                sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath,
                                             reporthook=_progress)
    print()
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_url', default='', type=str)
    args = parser.parse_args()

    kitti_data_url = args.kitti_url

    data_dir, run_dir = get_pathes()

    vgg_weights = os.path.join(data_dir, 'vgg16.npy')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    kitti_dec_dir = os.path.join(data_dir, 'KittiBox')

    if not os.path.exists(kitti_dec_dir):
        os.makedirs(kitti_dec_dir)

    # Download VGG DATA
    if not os.path.exists(vgg_weights):
        download_command = "wget {} -P {}".format(vgg_url, data_dir)
        logging.info("Downloading VGG weights.")
        download(vgg_url, data_dir)
    else:
        logging.warning("File: {} exists.".format(vgg_weights))
        logging.warning("Please delete to redownload VGG weights.")

    kitti_dec_zip = os.path.join(kitti_dec_dir, 'data_object_image_2.zip')
    kitti_label_zip = os.path.join(kitti_dec_dir, "data_object_label_2.zip")

    # Download KITTI DATA
    if not os.path.exists(kitti_dec_zip):
        if kitti_data_url == '':
            logging.error("Data URL for Kitti Data not provided.")
            url = "http://www.cvlibs.net/download.php?file=data_object_image_2.zip"
            logging.error("Please visit: {}".format(url))
            logging.error("and request Kitti Download link.")
            logging.error("Rerun scipt using"
                          "'python download_data.py --kitti_url [url]'")
            exit(1)
        if not kitti_data_url[-29:] == 'kitti/data_object_image_2.zip':
            logging.error("Wrong url.")
            url = "http://www.cvlibs.net/download.php?file=data_object_image_2.zip"
            logging.error("Please visit: {}".format(url))
            logging.error("and request Kitti Download link.")
            logging.error("Rerun scipt using"
                          "'python download_data.py --kitti_url [url]'")
            exit(1)
        else:
            logging.info("Downloading Kitti Road Data.")
            download(kitti_data_url, kitti_dec_dir)

    if not os.path.exists(kitti_label_zip):
        logging.info("Downloading Kitti Label Data.")
        kitti_main = os.path.dirname(kitti_data_url)
        kitti_label_url = os.path.join(kitti_main, kitti_label_zip)
        kitti_label_url = os.path.join(kitti_main,
                                       os.path.basename(kitti_label_zip))
        download(kitti_label_url, kitti_dec_dir)

    # Extract and prepare KITTI DATA
    logging.info("Extracting kitti_road data.")
    zipfile.ZipFile(kitti_dec_zip, 'r').extractall(kitti_dec_dir)
    zipfile.ZipFile(kitti_label_zip, 'r').extractall(kitti_dec_dir)

    logging.info("Preparing kitti_road data.")

    copyfiles = ["train_2.idl", "train_3.idl", "train_4.idl",
                 "val_2.idl", "val_3.idl", "val_4.idl"]

    for file in copyfiles:
        filename = os.path.join('data', file)
        copy2(filename, kitti_dec_dir)

    logging.info("All data have been downloaded successful.")


if __name__ == '__main__':
    main()
