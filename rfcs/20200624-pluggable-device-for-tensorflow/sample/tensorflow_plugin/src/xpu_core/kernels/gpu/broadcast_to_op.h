#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_BROADCAST_TO_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_BROADCAST_TO_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/bcast.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

template <typename Device, typename T>
struct BroadcastTo {
  template <int NDIMS>
  void DoBCast32Bit(const Device& device, typename TTypes<T, NDIMS>::Tensor out,
                    typename TTypes<T, NDIMS>::ConstTensor in,
                    const typename Eigen::array<int, NDIMS>& bcast) const {
    To32Bit(out).device(device) = To32Bit(in).broadcast(bcast);
  }

  template <int NDIMS>
  void DoBCast(
      const Device& device, typename TTypes<T, NDIMS>::Tensor out,
      typename TTypes<T, NDIMS>::ConstTensor in,
      const typename Eigen::array<Eigen::DenseIndex, NDIMS>& bcast) const {
    out.device(device) = in.broadcast(bcast);
  }

  template <int NDIMS>
  void ReshapeAndBCast(const Device& device, Tensor& output_tensor,
                       const Tensor& input_tensor, const BCast& bcast) const {
    const bool can_use_32bit = std::is_same<Eigen::GpuDevice, Device>::value &&
                               output_tensor.NumElements() < kint32max &&
                               input_tensor.NumElements() < kint32max;
    if (can_use_32bit) {
      DoBCast32Bit<NDIMS>(
          device, output_tensor.template shaped<T, NDIMS>(bcast.result_shape()),
          input_tensor.template shaped<T, NDIMS>(bcast.x_reshape()),
          BCast::ToIndexArrayType<int, NDIMS>(bcast.x_bcast()));
    } else {
      DoBCast<NDIMS>(
          device, output_tensor.template shaped<T, NDIMS>(bcast.result_shape()),
          input_tensor.template shaped<T, NDIMS>(bcast.x_reshape()),
          BCast::ToIndexArrayType<Eigen::DenseIndex, NDIMS>(bcast.x_bcast()));
    }
  }

  // PRECONDITION: rank(input_shape) > 0 &&
  //               rank(input_shape) <= rank(output_shape)  &&
  //               output_shape.num_elements() > 0.
  void operator()(const Device& device, OpKernelContext* ctx,
                  Tensor& output_tensor, const TensorShape& output_shape,
                  const Tensor& input_tensor, const TensorShape& input_shape,
                  const BCast& bcast) const {
    const int ndims = bcast.y_reshape().size();
    switch (ndims) {
      case 1:
        ReshapeAndBCast<1>(device, output_tensor, input_tensor, bcast);
        break;
      case 2:
        ReshapeAndBCast<2>(device, output_tensor, input_tensor, bcast);
        break;
      case 3:
        ReshapeAndBCast<3>(device, output_tensor, input_tensor, bcast);
        break;
      case 4:
        ReshapeAndBCast<4>(device, output_tensor, input_tensor, bcast);
        break;
      case 5:
        ReshapeAndBCast<5>(device, output_tensor, input_tensor, bcast);
        break;
      default:
        ctx->SetStatus(errors::Unimplemented(
            "Broadcast between ", input_shape.DebugString(), " and ",
            output_shape.DebugString(), " is not supported yet."));
        break;
    }
  }
};
}  // namespace functor

class BroadcastToOp {};

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_BROADCAST_TO_OP_H_