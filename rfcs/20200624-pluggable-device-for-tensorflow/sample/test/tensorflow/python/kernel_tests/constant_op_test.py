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
"""Tests for ConstantOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test

class FillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = self.evaluate(tf_ans)
    self.assertAllClose(np_ans, out)
    # Fill does not set the shape.
    # self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, dims, val, np_ans):
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillDouble(self):
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt32(self):
    np_ans = np.array([[42] * 3] * 2).astype(np.int32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt64(self):
    np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex64(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex128(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex128)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  @test_util.run_deprecated_v1
  def testFillString(self):
    np_ans = np.array([[b"yolo"] * 3] * 2)
    with self.session(use_gpu=False):
      tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").eval()
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testFillNegative(self):
    with self.cached_session():
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(ValueError):
          array_ops.fill(shape, 7)

      # Using a placeholder so this won't be caught in static analysis.
      dims = array_ops.placeholder(dtypes_lib.int32)
      fill_t = array_ops.fill(dims, 3.0)
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(errors_impl.InvalidArgumentError):
          fill_t.eval({dims: shape})

  def testFillEmptyInput(self):
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex64)
    with self.cached_session():
      fill = array_ops.fill(np.random.random([0]), np_ans[0][0], name="fill")
      out = self.evaluate(fill)

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(ValueError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(ValueError):
      array_ops.fill([3, 2], [1.0, 2.0])

    # Partial dimension information.
    f = array_ops.fill(array_ops.placeholder(dtypes_lib.int32, shape=(4,)), 3.0)
    self.assertEqual([None, None, None, None], f.get_shape().as_list())

    f = array_ops.fill(
        [array_ops.placeholder(
            dtypes_lib.int32, shape=()), 17], 1.0)
    self.assertEqual([None, 17], f.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testGradient(self):
    with self.cached_session():
      in_v = constant_op.constant(5.0)
      out_shape = [3, 2]
      out_filled = array_ops.fill(out_shape, in_v)
      err = gradient_checker.compute_gradient_error(in_v, [], out_filled,
                                                    out_shape)
    self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()
