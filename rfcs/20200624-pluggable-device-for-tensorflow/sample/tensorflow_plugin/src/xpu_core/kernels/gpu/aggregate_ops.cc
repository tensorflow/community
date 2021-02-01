#include "tensorflow_plugin/src/xpu_core/kernels/gpu/aggregate_ops.h"

#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"

namespace intel_plugin {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (!context->ValidateInputsAreSameShape()) return;
    const Tensor& input0 = context->input(0);
    const int num = context->num_inputs();

    if (num == 1) {
      context->set_output(0, input0);
      return;
    }

    // Try to forward and accumulate the result in one of the input buffers.
    gtl::InlinedVector<int, 8> input_indices(num);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    Tensor* output = nullptr;
    /* TODO(schen2): This way is blocked by RefCountIsOne().
    int candidate_input_index = 0;
    int forwarded_input = -1;
    for (int input_idx = 0; input_idx < num; ++input_idx) {
      candidate_input_index = input_idx;
      gtl::ArraySlice<int> candidate_input_indices(&candidate_input_index, 1);
      VLOG(0) << "CBOSS DBG:" << forwarded_input;
      OP_REQUIRES_OK(&context, context->forward_input_or_allocate_output(
                                    candidate_input_indices, 0, input0.shape(),
                                    &output, &forwarded_input));
      if (forwarded_input != -1) break;
    }
    if (forwarded_input > 0) {
      // Move the forwarded buffer to the front so we don't double count
      // anything if there are more than 8 inputs.
      input_indices[0] = forwarded_input;
      input_indices[forwarded_input] = 0;
    }
    */
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input0.shape(), &output));
    auto To = output->flat<T>();

#define I(IDX) context->input(input_indices[IDX]).template flat<T>()

    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(context->eigen_gpu_device(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(context->eigen_gpu_device(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(context->eigen_gpu_device(), To, I(0), I(1), I(2), I(3), I(4),
                 I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(context->eigen_gpu_device(), To, I(r), I(r + 1), I(r + 2),
                I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
#undef I
  }
};

#define REGISTER_KERNEL(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AddN").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      AddNOp<GPUDevice, TYPE>)
TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_bfloat16(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
}  // namespace intel_plugin