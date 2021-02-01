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
"""Tests for array_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as test_lib


class StridedSliceChecker(object):
  """Check a given tensor against the numpy result."""

  REF_TENSOR = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
  REF_TENSOR_ALIGNED = np.arange(1, 97, dtype=np.float32).reshape(3, 4, 8)

  def __init__(self, test, x, tensor_type=dtypes.int32, check_type_infer=True):
    self.x_np = np.array(x).astype(tensor_type.as_numpy_dtype)
    if tensor_type.is_bool:
      self.x_np = np.array(x % 3).astype(np.bool)
    # Give the value a non-zero imaginary component for complex types.
    if tensor_type.is_complex:
      self.x_np -= 1j * self.x_np
    self.test = test
    self.x = constant_op.constant(self.x_np, dtype=tensor_type)
    self.check_type_infer = check_type_infer

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)

    def eval_if_tensor(x):
      try:
        return x.eval()
      except AttributeError:
        return x

    if isinstance(spec, bool) or \
      (isinstance(spec, ops.Tensor) and spec.dtype == dtypes.bool) or \
      (isinstance(spec, np.ndarray) and spec.dtype == bool) or \
      (isinstance(spec, (list, tuple)) and np.asarray(spec).dtype == bool):
      tensor = op.eval()
      np_spec = eval_if_tensor(spec)
      self.test.assertAllEqual(self.x_np[np_spec], tensor)
      return tensor

    if not isinstance(spec, (list, tuple)):
      spec = [spec]

    tensor = op.eval()

    # Make a numpy spec that pre-evals the tensors
    np_specs = []

    for s in spec:
      if isinstance(s, slice):
        start = eval_if_tensor(s.start)
        stop = eval_if_tensor(s.stop)
        step = eval_if_tensor(s.step)
        np_specs.append(slice(start, stop, step))
      else:
        np_specs.append(eval_if_tensor(s))

    self.test.assertAllEqual(self.x_np[tuple(np_specs)], tensor)
    if self.check_type_infer:
      self.test.assertAllEqual(tensor.shape, op.get_shape())
    return tensor


STRIDED_SLICE_TYPES = [
    dtypes.int32, dtypes.float32
]


class StridedSliceTest(test_util.TensorFlowTestCase):
  """Test the strided slice operation with variants of slices."""

  @test_util.run_deprecated_v1
  def test_basic_slice(self):
    for tensor_type in STRIDED_SLICE_TYPES:
      with self.subTest(tensor_type=tensor_type):
        with self.cached_session(use_gpu=True):
          checker = StridedSliceChecker(
              self, StridedSliceChecker.REF_TENSOR, tensor_type=tensor_type)
          _ = checker[:, :, :]
          # Various ways of representing identity slice
          _ = checker[:, :, :]
          _ = checker[::, ::, ::]
          _ = checker[::1, ::1, ::1]
          # Not zero slice
          _ = checker[::1, ::5, ::2]
          # Reverse in each dimension independently
          _ = checker[::-1, :, :]
          _ = checker[:, ::-1, :]
          _ = checker[:, :, ::-1]
          ## negative index tests i.e. n-2 in first component
          _ = checker[-2::-1, :, ::1]
          # negative index tests i.e. n-2 in first component, non-unit stride
          _ = checker[-2::-1, :, ::2]

          # Check rank-0 examples
          checker2 = StridedSliceChecker(self, 5, tensor_type=tensor_type)
          _ = checker2[None]
          _ = checker2[...]
          _ = checker2[tuple()]

  def testInt64GPU(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    with test_util.force_gpu():
      x = constant_op.constant([1., 2., 3.])
      begin = constant_op.constant([2], dtype=dtypes.int64)
      end = constant_op.constant([3], dtype=dtypes.int64)
      strides = constant_op.constant([1], dtype=dtypes.int64)
      s = array_ops.strided_slice(x, begin, end, strides)
      self.assertAllEqual([3.], self.evaluate(s))

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.assert_no_garbage_created
  def testTensorSliceEagerMemory(self):
    with context.eager_mode():
      inputs = constant_op.constant([[[1], [2], [3], [4]]],
                                    dtype=dtypes.float32)
      # Tests that slicing an EagerTensor doesn't leak memory
      inputs[0]  # pylint: disable=pointless-statement

  @test_util.assert_no_new_pyobjects_executing_eagerly
  @test_util.assert_no_garbage_created
  def testVariableSliceEagerMemory(self):
    with context.eager_mode():
      v = variables.Variable([1., 2.])
      v[0]  # pylint: disable=pointless-statement

  @test_util.run_deprecated_v1
  def testDegenerateSlices(self):
    with self.session(use_gpu=True):
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      # degenerate by offering a forward interval with a negative stride
      _ = checker[0:-1:-1, :, :]
      # degenerate with a reverse interval with a positive stride
      _ = checker[-1:0, :, :]
      # empty interval in every dimension
      _ = checker[-1:0, 2:2, 2:3:-1]
      # empty first dimension only (used to break for aligned tensors).
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      _ = checker[1:0]

  @test_util.run_deprecated_v1
  def testSliceWithUndefinedDimension(self):
    t = constant_op.constant([1, 2, 3])
    d = tensor_shape.Dimension(None)
    self.assertAllEqual(t[d:d:d], t)

  @test_util.run_deprecated_v1
  def testEllipsis(self):
    with self.session(use_gpu=True):
      raw = [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]]
      checker = StridedSliceChecker(self, raw)

      _ = checker[0:]
      # implicit ellipsis
      _ = checker[0:, ...]
      # ellipsis alone
      _ = checker[...]
      # ellipsis at end
      _ = checker[0:1, ...]
      # ellipsis at begin
      _ = checker[..., 0:1]
      # ellipsis at middle
      _ = checker[0:1, ..., 0:1]
      # multiple ellipses not allowed
      with self.assertRaisesRegex(ValueError, "Multiple ellipses"):
        _ = checker[..., :, ...].eval()

  @test_util.run_deprecated_v1
  def testShrink(self):
    with self.session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      _ = checker[:, :, :, :, 3]
      _ = checker[..., 3]
      _ = checker[:, 0]
      _ = checker[:, :, 0]

  @test_util.run_deprecated_v1
  def testBothNewAxisAndShrink(self):
    with self.session(use_gpu=True):
      ones = array_ops.placeholder(shape=[2, 2], dtype=dtypes.int16)
      self.assertAllEqual(
          ones[array_ops.newaxis, :,
               0].eval(feed_dict={ones: [[1, 1], [1, 1]]}), [[1, 1]])

  @test_util.run_deprecated_v1
  def testTensorIndexing(self):
    with self.session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw, check_type_infer=False)
      bar = constant_op.constant(2)
      bar2 = constant_op.constant(3)
      _ = checker[..., bar:bar2]
      _ = checker[..., bar]
      _ = checker[..., 3]
      _ = checker[..., 2**64 // 2**63]  # Test longs in Python 2

  def testTensorIndexingTypeError(self):
    with self.session(use_gpu=True):
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      expected = re.escape(array_ops._SLICE_TYPE_ERROR)
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker["foo"]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant("foo")]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[0.0]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant(0.0)]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[constant_op.constant([1, 2, 3])]
      with self.assertRaisesRegex(TypeError, expected):
        _ = checker[[2.1, -0.7, 1.5]]

  @test_util.run_deprecated_v1
  def testExpand(self):
    with self.session(use_gpu=True):
      raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
              [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
      checker = StridedSliceChecker(self, raw)
      # new axis (followed by implicit ellipsis)
      _ = checker[np.newaxis]
      # newaxis after ellipsis
      _ = checker[..., np.newaxis]
      # newaxis in between ellipsis and explicit range
      _ = checker[..., np.newaxis, :]
      _ = checker[:, ..., np.newaxis, :, :]
      # Reverse final dimension with new axis
      _ = checker[:, :, np.newaxis, :, 2::-1]
      # Ellipsis in middle of two newaxis
      _ = checker[np.newaxis, ..., np.newaxis]

  @test_util.run_deprecated_v1
  def testExpandVariable(self):
    with self.session(use_gpu=True):
      x = variables.Variable(7, dtype=dtypes.int32)
      self.evaluate(x.initializer)
      y = x[None].eval()
      self.assertEqual(y.shape, (1,))
      self.assertAllEqual(y, (7,))

  @test_util.run_deprecated_v1
  def testOptimizedCases(self):
    with self.session(use_gpu=True):
      checker = StridedSliceChecker(self,
                                    StridedSliceChecker.REF_TENSOR_ALIGNED)
      # Identity
      _ = checker[:]
      # Identity
      _ = checker[...]
      # Identity
      _ = checker[np.newaxis, ..., np.newaxis]
      # First axis slice
      _ = checker[1:]
      # First axis slice
      _ = checker[np.newaxis, 1:]

  @test_util.run_v1_only("currently failing on v2")
  def testMasks(self):
    with self.session(use_gpu=True):
      scalar = np.array(0)
      # Test tensor type mask
      checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
      _ = checker[checker.x > 2]
      _ = checker[checker.x <= 5]
      _ = checker[ops.convert_to_tensor(scalar)]

      # Test numpy array type mask
      raw = np.array([[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                       [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23,
                                                              24]]]]])
      checker1 = StridedSliceChecker(self, raw)
      _ = checker1[raw >= 4]
      _ = checker1[raw < 19]
      _ = checker1[scalar]

      # Test boolean and non boolean cases
      mask = np.array([True, False, True])
      raw1 = np.array([[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]])
      checker2 = StridedSliceChecker(self, raw1)
      _ = checker2[mask]
      _ = checker2[ops.convert_to_tensor(mask)]


class StridedSliceShapeChecker(object):

  def __init__(self, x):
    self.x = x

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    return op.get_shape()


class StridedSliceShapeTest(test_util.TensorFlowTestCase):
  """Test the shape inference of StridedSliceShapes."""

  @test_util.run_deprecated_v1
  def testUnknown(self):
    with self.session(use_gpu=True):
      uncertain_tensor = array_ops.placeholder(dtypes.float32)
      a = StridedSliceShapeChecker(uncertain_tensor)
      a_slice_shape = a[...]
      self.assertAllEqual(a_slice_shape.ndims, None)

  def tensorShapeEqual(self, x, y):
    self.assertTrue(x is not None and y is not None or x is None and y is None)
    self.assertEqual(x.as_list(), y.as_list())

  @test_util.run_deprecated_v1
  def testTensorShapeUncertain(self):
    with self.session(use_gpu=True):
      uncertain_tensor = array_ops.placeholder(
          dtypes.float32, shape=(5, None, 7))
      a = StridedSliceShapeChecker(uncertain_tensor)
      self.tensorShapeEqual(a[3:5], tensor_shape.TensorShape([2, None, 7]))
      self.tensorShapeEqual(a[3:5, :, 4], tensor_shape.TensorShape([2, None]))
      self.tensorShapeEqual(a[3:5, 3:4, 4], tensor_shape.TensorShape([2, None]))
      self.tensorShapeEqual(a[3:5, :, 5:10],
                            tensor_shape.TensorShape([2, None, 2]))
      self.tensorShapeEqual(a[3:5, :, 50:3],
                            tensor_shape.TensorShape([2, None, 0]))
      self.tensorShapeEqual(a[3:5, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[1:5:2, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[:5:3, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([2, None, 1, 0]))
      self.tensorShapeEqual(a[:2:3, :, array_ops.newaxis, 50:3,],
                            tensor_shape.TensorShape([1, None, 1, 0]))
      self.tensorShapeEqual(a[::-1, :, array_ops.newaxis, ::-2],
                            tensor_shape.TensorShape([5, None, 1, 4]))

  @test_util.run_deprecated_v1
  def testTensorValuedIndexShape(self):
    with self.session(use_gpu=True):
      defined_shape_tensor = array_ops.placeholder(
          dtypes.float32, shape=(5, 3, 7))
      index_value = array_ops.placeholder(dtypes.int32, shape=())
      a = StridedSliceShapeChecker(defined_shape_tensor)
      self.tensorShapeEqual(a[index_value], tensor_shape.TensorShape([3, 7]))
      self.tensorShapeEqual(a[index_value, ::-1],
                            tensor_shape.TensorShape([3, 7]))
      self.tensorShapeEqual(a[index_value, ::-2],
                            tensor_shape.TensorShape([2, 7]))
      other_scalar = array_ops.placeholder(dtypes.int32, shape=())
      self.tensorShapeEqual(a[index_value, other_scalar:2],
                            tensor_shape.TensorShape([None, 7]))


class GradSliceChecker(object):
  """Tests that we can compute a gradient for var^2."""

  def __init__(self, test, sess, var, varnp):
    self.test = test
    self.sess = sess
    self.val = var * var
    self.var = var
    self.varnp = varnp

  def __getitem__(self, spec):
    slice_var = self.var[spec]
    slice_val = self.val[spec]

    # compute analytic 2nd derivative
    analytic_grad2 = 2 * slice_val

    dy = variables.Variable(
        array_ops.ones_like(slice_var, dtype=dtypes.float32))
    assign = dy.assign(slice_var)
    slice_val_grad, = gradients_impl.gradients(slice_val, self.var, grad_ys=dy)
    slice_val_grad2, = gradients_impl.gradients(
        slice_val_grad, dy, grad_ys=self.var)
    self.sess.run(assign)
    slice_val_grad_evaled, slice_val_grad2_evaled = (
        self.sess.run([slice_val_grad, slice_val_grad2]))
    analytic_grad2_evaled = analytic_grad2.eval()
    self.test.assertAllEqual(slice_val_grad2_evaled, analytic_grad2_evaled)

    # compute analytic gradient for slice
    np_val_grad = (2 * self.varnp * self.varnp)
    np_sliceval_grad = np.zeros(self.var.get_shape())
    if isinstance(spec, ops.Tensor):
      spec = self.sess.run([spec])
    np_sliceval_grad[spec] = np_val_grad[spec]
    # verify gradient
    self.test.assertAllEqual(slice_val_grad_evaled, np_sliceval_grad)


class StridedSliceGradTest(test_util.TensorFlowTestCase):
  """Test that strided slice's custom gradient produces correct gradients."""

  @test_util.run_v1_only("b/120545219")
  def testGradient(self):
    with self.session(use_gpu=True) as sess:
      var = variables.Variable(
          array_ops.reshape(
              math_ops.range(1, 97, 1, dtype=dtypes.float32), shape=(6, 4, 4)))
      init = variables.global_variables_initializer()
      sess.run(init)

      raw = np.array(range(1, 97, 1)).reshape((6, 4, 4))
      grad = GradSliceChecker(self, sess, var, raw)
      _ = grad[2:6:2, 1:3, 1:3]
      _ = grad[3:0:-2, 1:3, 1:3]
      _ = grad[3:0:-2, array_ops.newaxis, 1:3, 2, array_ops.newaxis]
      _ = grad[3:0:-2, 1:3, 2]
      _ = grad[:, -1, :]
      _ = grad[:, -2, :]
      with self.assertRaisesRegex(ValueError, "out of bounds"):
        _ = grad[:, -200, :]
      with self.assertRaisesRegex(ValueError, "out of bounds"):
        _ = grad[:, 200, :]

      # Test numpy array type mask
      _ = grad[raw > 51]
      # Test tensor type mask
      _ = grad[ops.convert_to_tensor(raw) <= 76]

  @test_util.run_v1_only("b/120545219")
  def testGradientZero(self):
    with self.session(use_gpu=True) as sess:
      var = variables.Variable(8.)
      init = variables.global_variables_initializer()
      sess.run(init)
      grad = GradSliceChecker(self, sess, var, np.array(8))
      _ = grad[tuple()]

  @test_util.run_deprecated_v1
  def testInt64Indices(self):
    with self.session(use_gpu=True) as sess:
      a = math_ops.range(3, dtype=dtypes.float32)
      index = constant_op.constant(1, dtype=dtypes.int64)
      b = 2. * a[index]
      grad, = gradients_impl.gradients(b, a)
      self.assertAllEqual(self.evaluate(grad), [0., 2., 0.])


class StridedSliceGradTypeTest(test_util.TensorFlowTestCase):
  """Test varied index types and host located memory."""

  # @test_util.run_deprecated_v1
  # def testHostVsDevice(self):
  #   with self.session(use_gpu=True) as sess:
  #     var2 = variables.Variable(
  #         array_ops.reshape(
  #             math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
  #             shape=(4, 1, 1)))
  #     varshape = variables.Variable([6, 4, 4], dtype=dtypes.int32)
  #     self.evaluate(variables.global_variables_initializer())
  #     begin = constant_op.constant([0, 0, 0])
  #     end = constant_op.constant([4, 1, 1])
  #     strides = constant_op.constant([1, 1, 1])
  #     foo = array_ops.strided_slice_grad(varshape, begin, end, strides, var2)
  #     sess.run(foo)

  @test_util.run_deprecated_v1
  def testInt64Shape(self):
    with self.session(use_gpu=True) as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      self.evaluate(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int64)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                        original_dy)
      sess.run(dx)

  @test_util.run_deprecated_v1
  def testMixedIndexTypes(self):
    with self.session(use_gpu=True) as sess:
      original_dy = array_ops.reshape(
          math_ops.cast(math_ops.range(1, 5, 1), dtypes.float32),
          shape=(4, 1, 1))
      original_shape = constant_op.constant([6, 4, 4], dtype=dtypes.int64)
      self.evaluate(variables.global_variables_initializer())
      begin = constant_op.constant([0, 0, 0], dtype=dtypes.int32)
      end = constant_op.constant([4, 1, 1], dtype=dtypes.int64)
      strides = constant_op.constant([1, 1, 1], dtype=dtypes.int64)
      with self.assertRaisesRegex(
          TypeError, "Input 'begin' of 'StridedSliceGrad' Op has type int32"
          " that does not match type int64 of argument 'shape'"):
        dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                          original_dy)
        sess.run(dx)

class SnapshotOpTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testInvertPermutation(self):
    for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
      with self.cached_session(use_gpu=True):
        x = constant_op.constant([0, 1, 2, 3], dtype=dtype)
        y = gen_array_ops.snapshot(x)
        self.assertAllEqual(y.eval(), [0, 1, 2, 3])

if __name__ == "__main__":
  test_lib.main()
