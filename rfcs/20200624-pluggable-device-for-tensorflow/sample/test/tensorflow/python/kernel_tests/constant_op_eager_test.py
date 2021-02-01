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

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops


class FillTest(test.TestCase):

  def _compare(self, dims, val, np_ans, use_gpu):
    ctx = context.context()
    device = "GPU:0" if (use_gpu and ctx.num_gpus()) else "CPU:0"
    with ops.device(device):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.numpy()
    self.assertAllClose(np_ans, out)

  def _compareAll(self, dims, val, np_ans):
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

#   def testFillFloat(self):
#     np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
#     self._compareAll([2, 3], np_ans[0][0], np_ans)
# 
#   def testFillDouble(self):
#     np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
#     self._compareAll([2, 3], np_ans[0][0], np_ans)
# 
#   def testFillInt32(self):
#     np_ans = np.array([[42] * 3] * 2).astype(np.int32)
#     self._compareAll([2, 3], np_ans[0][0], np_ans)
# 
#   def testFillInt64(self):
#     np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
#     self._compareAll([2, 3], np_ans[0][0], np_ans)
# 
#   def testFillComplex64(self):
#     np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
#     self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)
# 
#   def testFillComplex128(self):
#     np_ans = np.array([[0.15] * 3] * 2).astype(np.complex128)
#     self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)
# 
#   def testFillString(self):
#     np_ans = np.array([[b"yolo"] * 3] * 2)
#     tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").numpy()
#     self.assertAllEqual(np_ans, tf_ans)
# 
#   def testFillNegative(self):
#     for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
#       with self.assertRaises(errors_impl.InvalidArgumentError):
#         array_ops.fill(shape, 7)
# 
  def testShapeFunctionEdgeCases(self):
    # Non-vector dimensions.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([3, 2], [1.0, 2.0])


if __name__ == "__main__":
  test.main()
