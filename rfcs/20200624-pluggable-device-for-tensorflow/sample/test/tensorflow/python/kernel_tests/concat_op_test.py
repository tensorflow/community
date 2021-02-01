from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import test


class ConcatOpTest(test.TestCase):
    def testConcatEmpty(self):
        with test_util.use_gpu():
            t1 = []
            t2 = []
            output = gen_array_ops.concat_v2([t1, t2], 0)
            self.assertFalse(self.evaluate(output))  # Checks that output is empty

    @test_util.run_deprecated_v1
    def testConcatInvalidAxis(self):
        with self.assertRaises(ValueError):
            with test_util.use_gpu():
                t1 = [1]
                t2 = [2]
                gen_array_ops.concat_v2([t1, t2], 1).eval()

    def testConcatNegativeAxis(self):
        with test_util.use_gpu():
            t1 = [[1, 2, 3], [4, 5, 6]]
            t2 = [[7, 8, 9], [10, 11, 12]]

            c = gen_array_ops.concat_v2([t1, t2], -2)
            self.assertEqual([4, 3], c.get_shape().as_list())
            output = self.evaluate(c)
            self.assertAllEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                                output)

            c = gen_array_ops.concat_v2([t1, t2], -1)
            self.assertEqual([2, 6], c.get_shape().as_list())
            output = self.evaluate(c)
            self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)

    def testConcatDtype(self):
        for dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16]:
            with test_util.use_gpu():
                t1 = constant_op.constant([[1, 2, 3], [4, 5, 6]], dtype=dtype)
                t2 = constant_op.constant([[7, 8, 9], [10, 11, 12]], dtype=dtype)

                c = gen_array_ops.concat_v2([t1, t2], 1)
                self.assertEqual([2, 6], c.get_shape().as_list())
                output = self.evaluate(c)
                self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)

    def testConcatAxisType(self):
        for dtype in [dtypes.int32, dtypes.int64]:
            with test_util.use_gpu():
                t1 = [[1, 2, 3], [4, 5, 6]]
                t2 = [[7, 8, 9], [10, 11, 12]]

                c = gen_array_ops.concat_v2([t1, t2],
                                            constant_op.constant(1, dtype=dtype))
                self.assertEqual([2, 6], c.get_shape().as_list())
                output = self.evaluate(c)
                self.assertAllEqual([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]], output)


if __name__ == '__main__':
    test.main()
