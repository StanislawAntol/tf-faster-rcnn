import sys
sys.path.insert(0, 'lib')

import numpy as np
import tensorflow as tf

from proposal_layer_tf import proposal_layer_tf


test_file = 'proposal_layer_test/test1'

npz_file = np.load(test_file + '_input.npz')
rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, anchors, num_anchors = \
    npz_file['rpn_cls_prob'], npz_file['rpn_bbox_pred'], npz_file['im_info'], \
    npz_file['cfg_key'].tolist(), npz_file['anchors'], npz_file['num_anchors']

npz_file = np.load(test_file + '_output.npz')
exp_blob, exp_scores = npz_file['blob'], npz_file['scores']


# Define intput variables
tf_rpn_cls_prob = tf.constant(rpn_cls_prob)
tf_rpn_bbox_pred = tf.constant(rpn_bbox_pred)
tf_im_info = tf.constant(im_info)
tf_anchors = tf.constant(anchors)
tf_num_anchors = tf.constant(num_anchors)

res_full = proposal_layer_tf(rpn_cls_prob=tf_rpn_cls_prob,
                             rpn_bbox_pred=tf_rpn_bbox_pred,
                             im_info=tf_im_info,
                             cfg_key=cfg_key,
                             anchors=tf_anchors,
                             num_anchors=num_anchors)

sess = tf.Session()
tf_blob_res, tf_scores_res = sess.run(res_full)

print '\nAll bounding boxes and scores correct?'
print np.allclose(tf_blob_res, exp_blob),
print np.allclose(tf_scores_res, exp_scores)

print '\nTop 50 bounding boxes and scores correct?'
print np.allclose(tf_blob_res[:50], exp_blob[:50]),
print np.allclose(tf_scores_res[:50], exp_scores[:50])
