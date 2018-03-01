#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

'''
Demo script showing detections in sample images.

See README.md for installation instructions before running.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

import _init_paths
import datasets.classes
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

slim = tf.contrib.slim

CLASSES = ()


def preprocess(im):
    # Cast and remove mean
    im_orig = tf.cast(im, dtype=tf.float32)
    im_orig -= cfg.PIXEL_MEANS

    # Get shape and size
    im_shape = tf.cast(tf.shape(im_orig), dtype=tf.float32)
    im_size_min = tf.reduce_min(im_shape[:2])
    im_size_max = tf.reduce_max(im_shape[:2])

    # Resize image
    # Stan - Is this needed?
    im_scale = cfg.TEST.SCALES[0] / im_size_min
    # Prevent the biggest axis from being more than MAX_SIZE
    im_scale = tf.cond(
        tf.round(im_scale * im_size_max) > float(cfg.TEST.MAX_SIZE),
        lambda: cfg.TEST.MAX_SIZE / im_size_max,
        lambda: im_scale)
    im_resized = tf.image.resize_images(im_orig,
                                        tf.cast(
                                            tf.round(im_shape[:2] * im_scale),
                                            dtype=tf.int32),
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=True)
    im_blob = im_resized[tf.newaxis]
    im_info = tf.concat([tf.cast(tf.shape(im_resized)[:2], dtype=tf.float32),
                         [im_scale]], 0)[np.newaxis]
    return im_blob, im_info


def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser(
        description='Freeze TF Faster R-CNN model as .pb'
    )
    parser.add_argument(
        '-c',
        '--ckpt_fn',
        dest='ckpt_fn',
        help=('Path to checkpoint file(s) '
              '[/tmp/models/coco_2014_train+coco_2014_valminusminival/'
              'mobile_faster_rcnn_iter_1190000.ckpt]'
              ),
        default=('/tmp/models/coco_2014_train+coco_2014_valminusminival/'
                 'mobile_faster_rcnn_iter_1190000.ckpt'
                 ),
    )
    parser.add_argument(
        '-o',
        '--output_names',
        dest='output_names',
        help=('Comma-separated list of which nodes are needed in frozen model '
              '[MobilenetV1_2/rois/concat_1]'
              ),
        default='MobilenetV1_2/rois/concat_1',
    )
    parser.add_argument(
        '-f',
        '--out_fn',
        dest='out_fn',
        help=('Output .pb filename [replaces .ckpt w/ .pb in ckpt_fn]'
              ),
        default='',
    )
    parser.add_argument(
        '-p',
        '--print',
        dest='print',
        help=('Print all of the node names. '
              '[False, no printing]'
              ),
        action='store_true',
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    ckpt_fn = args.ckpt_fn
    output_node_names = args.output_names
    print_nodes = args.print

    if len(args.out_fn) == 0:
        print("Output filename unspecified; just changing extension.")
        output_graph_fn = ckpt_fn.replace(".ckpt", ".pb")
    else:
        output_graph_fn = args.out_fn

    if not os.path.isfile(ckpt_fn + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks '
                       'from our server and place them properly?.'
                       ).format(ckpt_fn + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if 'vgg16' in ckpt_fn:
        net = vgg16(batch_size=1)
    elif 'res101' in ckpt_fn:
        net = resnetv1(batch_size=1, num_layers=101)
    elif 'mobile' in ckpt_fn:
        net = mobilenetv1()
    else:
        raise NotImplementedError

    raw_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    image, im_info = preprocess(raw_image)

    # Set the right classes for visualization
    if 'coco' in ckpt_fn:
        CLASSES = datasets.classes.MS_COCO
        net.create_architecture('TEST', len(CLASSES),
                                tag='default',
                                anchor_scales=[4, 8, 16, 32],
                                anchor_ratios=[0.5, 1, 2],
                                image=image, im_info=im_info)
    elif 'pascal_voc' in ckpt_fn:
        CLASSES = datasets.classes.PASCAL_VOC
        net.create_architecture('TEST', len(CLASSES),
                                tag='default', anchor_scales=[8, 16, 32],
                                image=image, im_info=im_info)
    else:
        raise NotImplementedError

    rest_var = slim.get_variables_to_restore()

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        if print_nodes:
            for n in input_graph_def.node:
                print(n.name)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_fn)
        print('Loaded model.')

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            # The output node names are used to select the useful nodes
            output_node_names.split(',')
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_fn, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('{} ops in the final graph.'.format(len(output_graph_def.node)))

    print('Done freezing!')
