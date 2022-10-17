# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
a = tf.random.normal(shape=[1,10, 10, 8], dtype=tf.float32, seed=1)
w = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)

a1 = tf.random.normal(shape=[1, 10, 10, 8], dtype=tf.float32, seed=1)
w1 = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)


with tf.device("/MY_DEVICE:0"):
    b = tf.nn.relu(a)
    c = tf.nn.conv2d(b, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

with tf.device("/CPU:0"):
    b1 = tf.nn.relu(a1)
    c1 = tf.nn.conv2d(b1, w1, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=True))
print(sess.run(tf.reduce_all(tf.less(c - c1, 1e-5))))
