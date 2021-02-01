#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
a = tf.random.normal(shape=[10], dtype=tf.float32)

with tf.device("/MY_DEVICE:0"):
    b = tf.nn.relu(a)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=True))
print(sess.run(b))
