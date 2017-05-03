import sys
sys.path.insert(0, 'lib')

import numpy as np
import tensorflow as tf

from snippets import generate_anchors_pre
from generate_anchors_tf import generate_anchors_pre_tf


# This tests compares if `generate_anchors_pre_tf` is doing exactly the same
# as the original numpy implementation `generate_anchors_pre`.
stride = 16
im_width = 800
im_height = 600
height = np.int32(np.ceil(im_height / np.float32(stride)))
width = np.int32(np.ceil(im_width / np.float32(stride)))


tf_op = generate_anchors_pre_tf(width, height, stride)
tf_res = tf.Session().run(tf_op)
np_res = generate_anchors_pre(width, height, stride)

print np.allclose(tf_res[0], np_res[0])
print np.allclose(tf_res[1], np_res[1])
