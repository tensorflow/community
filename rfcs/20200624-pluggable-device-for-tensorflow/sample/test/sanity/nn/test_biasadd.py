from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class BiasAddTest(test_util.TensorFlowTestCase):
    """test BiasAdd op"""

    def _testHelper(self, use_gpu, dtype):
        # set both the global and operation-level seed to ensure this result is reproducible
        with self.session(use_gpu=use_gpu):
            tf.random.set_seed(5)
            x1 = tf.random.normal(shape=[1, 2, 2, 3], seed=1, dtype=dtype)
            x2 = tf.random.normal(shape=[3], seed=1, dtype=dtype)
            return self.evaluate(nn_ops.bias_add(x1, x2))

    def testBiasAddFp32(self):
        ans_cpu = self._testHelper(False, tf.dtypes.float32)
        ans_gpu = self._testHelper(True, tf.dtypes.float32)
        self.assertAllClose(ans_cpu, ans_gpu)

    def testBiasAddFp16(self):
        ans_cpu = self._testHelper(False, tf.dtypes.float16)
        ans_gpu = self._testHelper(True, tf.dtypes.float16)
        self.assertAllClose(ans_cpu, ans_gpu)

    def testBiasAddBf16(self):
        ans_cpu = self._testHelper(False, tf.dtypes.bfloat16)
        ans_gpu = self._testHelper(True, tf.dtypes.bfloat16)
        self.assertAllClose(ans_cpu, ans_gpu)


if __name__ == '__main__':
    test.main()
