"""Tests for _FusedConv2D related functionality in tensorflow.ops.nn."""
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()


def test_conv_biasadd():
    x = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
    w = np.ones([1, 2, 1, 1]).astype(np.float32)
    b = np.array([1], dtype=np.float32)
    expected_result = np.array([[[[4.], [6.], [4.]],
                                 [[10.], [12.], [7.]],
                                 [[16.], [18.], [10.]]]])
    with tf.compat.v1.Session() as sess:
        fused_graph = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC'), b)
        result = sess.run(fused_graph)
        np.array_equal(expected_result, result)


def main():
    test_conv_biasadd()


if __name__ == '__main__':
    main()
