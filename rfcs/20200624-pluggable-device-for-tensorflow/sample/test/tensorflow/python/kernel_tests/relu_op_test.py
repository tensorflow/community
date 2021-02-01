# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for Relu and ReluGrad."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import tf2
from tensorflow.python.compat import compat
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

class ReluTest(test.TestCase):

  def _npRelu(self, np_features):
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]])))

  def _testRelu(self, np_features):
    np_relu = self._npRelu(np_features)
    tf_relu = nn_ops.relu(np_features)
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  # ToDo(yangshe1): Add int8 relu test when kernel is enabled
  # def testReluInt8x4GoodShape(self):
  #   if not test.is_gpu_available(cuda_only=True):
  #     self.skipTest("No GPU available")
  #   inputs = np.array([[-50, 7, 23, 0], [-1, -5, 6, 11]])
  #   np_relu = self._npRelu(inputs)
  #   tf_relu = nn_ops.relu(constant_op.constant(inputs, dtypes.qint8))
  #   self.assertAllClose(np_relu, tf_relu)
  #   self.assertShapeEqual(np_relu, tf_relu)

  # @test_util.disable_xla("b/123338077")  # Passes with XLA
  # def testReluInt8x4BadShape(self):
  #   if not test.is_gpu_available(cuda_only=True):
  #     self.skipTest("No GPU available")
  #   inputs = constant_op.constant(
  #       np.array([[-50, 7, 23], [0, 1, -5], [6, -2, 11]]), dtypes.qint8)
  #   with self.assertRaisesRegex(
  #       errors.InvalidArgumentError,
  #       "Tensor size must be a multiple of 4 for Relu<qint8>. Got 9"):
  #     self.evaluate(nn_ops.relu(inputs))

  #   inputs = constant_op.constant(
  #       np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -8, 7, -6, 5, -4, 3, -2, 1]),
  #       dtypes.qint8)
  #   with self.assertRaisesRegex(
  #       errors.InvalidArgumentError,
  #       "Tensor size must be a multiple of 4 for Relu<qint8>. Got 17"):
  #     self.evaluate(nn_ops.relu(inputs))

  def testNoElement(self):
    self._testRelu(np.array([[], []], dtype=np.float32))

  # The gradient test for ReLU is a bit tricky as the derivative is not well
  # defined at around zero and we want to avoid that in terms of input values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu, [x]))
    self.assertLess(err, 1e-4)

  # The gradient for fp16 is inaccurate due to the low-precision.
  # We compare the fp16 analytical gradient against their fp32 counterpart.
  def testGradientFloat16(self):

    def grad(x):
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = nn_ops.l2_loss(nn_ops.relu(x))
      return tape.gradient(y, x)

    def f():
      with test_util.use_gpu():
        # Randomly construct a 1D shape from [1, 40)
        shape = random_ops.random_uniform([1],
                                          minval=1,
                                          maxval=40,
                                          dtype=dtypes.int32)
        x32 = random_ops.random_uniform(shape, minval=-1, maxval=1)
        x16 = math_ops.cast(x32, dtype=dtypes.float16)
        return grad(x32), grad(x16)

    # We're going to ensure that the fp16 and fp32 gradients
    # are "close" to each other for ~100 random values.
    #
    # In TensorFlow 1.x, invoking f() (without eager execution enabled)
    # would construct a graph. Instead of construct a graph with O(100) nodes,
    # we construct a single graph to be executed ~100 times in a Session.
    if not tf2.enabled():
      d32_tensor, d16_tensor = f()
      with self.cached_session() as sess:
        f = lambda: sess.run([d32_tensor, d16_tensor])

    # Repeat the experiment for 100 times. All tensor shapes and its tensor
    # values are randomly generated for each run.
    for _ in xrange(100):
      d32, d16 = f()
      self.assertAllClose(d32, d16, atol=3e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu, [x]))
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-4)

  def testGradGradFloat64(self):
    with self.cached_session():

      def f(x):
        assert x.dtype == dtypes.float64
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
    self.assertLess(err, 1e-10)

  def testGradientScalar(self):
    x = variables.Variable(100.)

    def loss():
      return nn_ops.relu(x)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.25)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), 50.0)

  def testGradientNoElement(self):
    with self.cached_session():

      def f(x):
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
        return tape.gradient(y, x)

      x = np.asarray([[], []], dtype=np.float32)
      z = list(gradient_checker_v2.compute_gradient(f, [x]))[0][0]
      self.assertAllEqual(z, np.reshape(x, (0, 0)))

class LeakyReluTest(test.TestCase):

  def _npLeakyRelu(self, np_features, alpha=0.1):
    return np.maximum(np_features, alpha * np_features)

  def testNpLeakyRelu(self):
    self.assertAllClose(
        np.array([[-0.09, 0.7, -0.05, 0.3, -0.01],
                  [0.1, -0.03, 0.5, -0.07, 0.9]]),
        self._npLeakyRelu(
            np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7,
                                                     0.9]]),
            alpha=0.1))

  def _testLeakyRelu(self, np_features, alpha):
    np_leaky_relu = self._npLeakyRelu(np_features, alpha)
    tf_leaky_relu = nn_ops.leaky_relu(np_features, alpha)
    self.assertAllClose(np_leaky_relu, tf_leaky_relu)
    self.assertShapeEqual(np_leaky_relu, tf_leaky_relu)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32]: # delete the float64
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testLeakyRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            alpha=0.2)

  def testNumbersGPU(self):
    # TODO(schen2): LeakyReluTest has accuracy issue caused by updating oneDNN. [TFDO-4283]
    self.skipTest("LeakyReluTest has accuracy issue caused by updating oneDNN.")
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    type_candidates = [np.float16, np.float32] #TODO , np.float64]
    for t in type_candidates:
      self._testLeakyRelu(
        np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
        alpha=0.1)

  # The gradient test for Leaky ReLU is a bit tricky as the derivative is not
  # well defined at around zero and we want to avoid that in terms of input
  # values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x]))
    print("leaky_relu (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradGradFloat32(self):
    with compat.forward_compatibility_horizon(2018, 11, 2):
      with self.cached_session():

        def f(x):
          assert x.dtype == dtypes.float32
          with backprop.GradientTape() as tape:
            tape.watch(x)
            y = nn_ops.leaky_relu(x)
          return tape.gradient(y, x)

        x = np.asarray(
            [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
            dtype=np.float32,
            order="F")
        err = gradient_checker_v2.max_error(
            *gradient_checker_v2.compute_gradient(f, [x]))
      print("leaky_relu (float32) gradient of gradient err = ", err)
      self.assertLess(err, 1e-4)

  def testGradientScalar(self):
    x = variables.Variable(-100.)

    def loss():
      return nn_ops.leaky_relu(x, 0.05)**2

    optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.2)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(optimizer.minimize(loss))
    self.assertAllClose(x.read_value(), -99.9)

class Relu6Test(test.TestCase):
  
  def _npRelu6(self, np_features):
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes)

  def testNpRelu6(self):
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7,
                                                    0.9]])))

  def _testRelu6(self, np_features):
    np_relu6 = self._npRelu6(np_features)
    tf_relu6 = nn_ops.relu6(np_features)
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, tf_relu6)

  def testNumbersCPU(self):
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  def testNumbersGPU(self):
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float, np.double]:
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t))

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    print("relu6 (float32) gradient err = ", err)
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    with self.cached_session():
      x = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F")
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
    print("relu6 (float64) gradient err = ", err)
    self.assertLess(err, 1e-10)


if __name__ == "__main__":
  test.main()
