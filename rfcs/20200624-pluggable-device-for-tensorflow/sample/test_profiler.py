#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import os
tf.compat.v1.disable_eager_execution()

profile_options = tf.profiler.experimental.ProfilerOptions(
                                                   host_tracer_level = 3,
                                                   device_tracer_level = 1)

logpath = os.path.join('data', 'logs',  'profiler_demo')
 
a = tf.random.normal(shape=[1,10, 10, 8], dtype=tf.float32, seed=1)
w = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)

a1 = tf.random.normal(shape=[1, 10, 10, 8], dtype=tf.float32, seed=1)
w1 = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)


with tf.device("/MY_DEVICE:0"):
    tf.profiler.experimental.start(logpath)
    b = tf.nn.relu(a)
    c = tf.nn.conv2d(b, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
    tf.profiler.experimental.stop()
with tf.device("/CPU:0"):
    b1 = tf.nn.relu(a1)
    c1 = tf.nn.conv2d(b1, w1, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=True))
print(sess.run(tf.reduce_all(tf.less(c - c1, 1e-5))))

