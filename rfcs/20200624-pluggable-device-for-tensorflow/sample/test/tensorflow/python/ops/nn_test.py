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
"""Tests for miscellaneous functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test as test_lib


class DataFormatDimMapTest(test_lib.TestCase):

  def _test(self, x_val, y_val_expected):
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x)

    y_val = self.evaluate(y)
    self.assertAllEqual(y_val, y_val_expected)

  def test(self):
    self._test(0, 0)
    self._test(1, 2)
    self._test(2, 3)
    self._test(3, 1)
    self._test(-1, 1)
    self._test(-2, 3)
    self._test(-3, 2)
    self._test(-4, 0)
    self._test([1, 3], [2, 1])
    self._test([1, 3, -2], [2, 1, 3])
    self._test([1, -3, -2], [2, 2, 3])
    self._test([[1, -3], [1, -1]], [[2, 2], [2, 1]])

  def testNHWCtoNCHW(self):
    x_val = [1, -3, -2]
    y_val_expected = [2, 2, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="NCHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoHWNC(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [2, 0, 1, 3, 2, 0, 1, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNHWCtoWHCN(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 1, 0, 2, 3, 1, 0, 2]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoNCDHW(self):
    x_val = [1, -4, -3, -2]
    y_val_expected = [2, 2, 3, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="NCDHW")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testNDHWCtoDHWNC(self):
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [3, 0, 1, 2, 4, 3, 0, 1, 2, 4]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="DHWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testDNHWCtoWHDCN(self):
    x_val = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    y_val_expected = [4, 2, 1, 0, 3, 4, 2, 1, 0, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="NDHWC", dst_format="WHDCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)

  def testArbitraryASCII(self):
    x_val = [-4, -3, -2, -1, 0, 1, 2, 3]
    y_val_expected = [3, 2, 1, 0, 3, 2, 1, 0]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_dim_map(x, src_format="qwer", dst_format="rewq")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, y_val_expected)


class DataFormatVectorPermuteTest(test_lib.TestCase):

  def testNHWCToNCHW(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 3, 4, 9])

  def testNHWCToNCHW_Size2(self):
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4, 3, 7])

  @test_util.disable_xla("unsupported data format")
  def testNHWCToWHCN_Size2(self):
    x_val = [4, 9]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="WHCN")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 4])

  def testNCHWToNHWC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [7, 9, 3, 4])

  def testNCHWToNHWC_Size2(self):
    x_val = [9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 3])

  def testNHWCToHWNC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [4, 9, 7, 3])

  def testHWNCToNHWC(self):
    x_val = [7, 4, 9, 3]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [9, 7, 4, 3])

  def testNHWCToNCHW2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x)
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [5, 1], [9, 3], [4, 5]])

  def testNHWCToHWNC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NHWC", dst_format="HWNC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[9, 3], [4, 5], [7, 4], [5, 1]])

  def testHWNCToNHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="HWNC", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[4, 5], [7, 4], [9, 3], [5, 1]])

  def testNCHWToNHWC2D(self):
    x_val = [[7, 4], [9, 3], [4, 5], [5, 1]]
    x = constant_op.constant(x_val)
    y = nn_ops.data_format_vec_permute(x, src_format="NCHW", dst_format="NHWC")
    with test_util.use_gpu():
      y_val = self.evaluate(y)
      self.assertAllEqual(y_val, [[7, 4], [4, 5], [5, 1], [9, 3]])

class L2LossTest(test_lib.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testL2Loss(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant(
          [1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x", dtype=dtype)
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  @test_util.run_deprecated_v1
  def testGradient(self):
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      output = nn_ops.l2_loss(x)
      err = gradient_checker.compute_gradient_error(x, x_shape, output, [1])
    print("L2Loss gradient err = %g " % err)
    err_tolerance = 1e-10
    self.assertLess(err, err_tolerance)

if __name__ == "__main__":
  test_lib.main()
