#include "tensorflow_plugin/src/xpu_core/kernels/gpu/snapshot_op.h"

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

template <typename Device, typename Scalar>
class SnapshotOp : public OpKernel {
 public:
  explicit SnapshotOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    // Try to use buffer forwarding to avoid an explicit copy.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (!output->SharesBufferWith(input)) {
      functor::Snapshot<Device, Scalar> functor;
      functor(context->eigen_gpu_device(), input.flat<Scalar>(),
              output->flat<Scalar>());
    }
  }
};

typedef Eigen::GpuDevice GPUDevice;
#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<GPUDevice, TYPE>)

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
}  // namespace intel_plugin
