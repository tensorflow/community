import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util


def GetRandomNormalInput(shape, dtype):
    scale = 0.1
    loc = 0.1
    vals = np.array(np.random.normal(loc, scale, np.prod(shape)), dtype=dtype)
    return vals.reshape(shape)


class BatchMatMulTest(test_util.TensorFlowTestCase):
    """test BatchMatMul op."""

    # Uses numpy to compute batch_matmul(x, y, adjoint_a, adjoint_b).
    def _npBatchMatmul(self, x, y, adjoint_a, adjoint_b):
        # output's shape depends on adj[0] and adj[1]
        if adjoint_a:
            x = np.conjugate(np.swapaxes(x, -1, -2))
        if adjoint_b:
            y = np.conjugate(np.swapaxes(y, -1, -2))
        return np.matmul(x, y)

    # Compares TensorFlow BatchMatmul with NumPy's matmul.
    def _compare(self, x_in, y_in, adjoint_a, adjoint_b, static_shape):
        x_t_shape = x_in.shape[:-2] + (x_in.shape[-1], x_in.shape[-2])
        y_t_shape = y_in.shape[:-2] + (y_in.shape[-1], y_in.shape[-2])
        x = x_in if not adjoint_a else x_in.reshape(x_t_shape)
        y = y_in if not adjoint_b else y_in.reshape(y_t_shape)
        with self.cached_session(use_gpu=True):
            dtype = tf.bfloat16
            x = tf.cast(x, dtype=dtype)
            y = tf.cast(y, dtype=dtype)
            z0 = math_ops.matmul(x, y, adjoint_a=adjoint_a, adjoint_b=adjoint_b)
            z0_val = self.evaluate(z0)
            z1 = self._npBatchMatmul(x, y, adjoint_a, adjoint_b)
            self.assertAllClose(z0_val, z1, rtol=1e-2, atol=1e-2)

    def _testNonEmpty(self, dtype, adjoint_a, adjoint_b):

        def CompareNonEmpty(self, a_shape, b_shape):
            self._compare(
                GetRandomNormalInput(a_shape, dtype),
                GetRandomNormalInput(b_shape, dtype),
                adjoint_a,
                adjoint_b,
                static_shape=True)

        CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
        CompareNonEmpty(self, [1, 2, 3], [1, 3, 1])
        CompareNonEmpty(self, [1, 1, 3], [1, 3, 5])
        CompareNonEmpty(self, [1, 2, 3], [1, 3, 5])
        CompareNonEmpty(self, [7, 1, 3], [7, 3, 5])
        CompareNonEmpty(self, [7, 2, 3], [7, 3, 1])
        CompareNonEmpty(self, [7, 2, 3], [7, 3, 5])
        CompareNonEmpty(self, [10, 64, 75], [10, 75, 30])
        CompareNonEmpty(self, [5, 7, 2, 3], [5, 7, 3, 5])

    def testBf16(self):
        for adjoint_a_ in False, True:
            for adjoint_b_ in False, True:
                self._testNonEmpty(np.float64, adjoint_a_, adjoint_b_)


if __name__ == "__main__":
    test.main()
