"""Select Op test"""

from tensorflow import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops


class SelectTest(test_util.TensorFlowTestCase):
    """test Select Op."""

    def testSelect(self):
        x = constant_op.constant([True, False])
        a = constant_op.constant([[1, 2], [3, 4]])
        b = constant_op.constant([[5, 6], [7, 8]])
        expected = constant_op.constant([[1, 2], [7, 8]])
        self.assertAllEqual(expected, gen_math_ops.select(x, a, b))


if __name__ == '__main__':
    test.main()
