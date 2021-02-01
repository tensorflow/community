import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util

SHAPE1 = [64, 16]
SHAPE2 = [16, 64]

np.random.seed(1)
input_1 = np.reshape(np.random.normal(size=np.prod(SHAPE1)), newshape=SHAPE1)
input_2 = np.reshape(np.random.normal(size=np.prod(SHAPE2)), newshape=SHAPE2)


class MatmulTest(test_util.TensorFlowTestCase):
    """test Matmul op."""

    @test_util.run_deprecated_v1
    def testMatmulFp16(self):
        with self.session(use_gpu=False):
            ans_cpu = self.evaluate(tf.matmul(input_1, input_2))

        with self.session(use_gpu=True):
            dtype = tf.float16  # TODO: bf16 accuracy issue
            x1 = tf.cast(input_1, dtype)
            x2 = tf.cast(input_2, dtype)
            y_gpu = self.evaluate(tf.matmul(x1, x2))
            
        self.assertAllClose(tf.cast(ans_cpu, dtype), y_gpu, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test.main()
