#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_H_

#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct StridedSlice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).device(d) =
          To32Bit(input).stridedSlice(start_i, stop_i, strides_i);
    } else {
      output.device(d) =
          input.stridedSlice(start_indices, stop_indices, strides);
    }
  }
};

template <typename T, int NDIMS, typename Device>
struct InitOutput {
  static void run(const Device& d, typename TTypes<T, NDIMS>::Tensor output) {
    output.device(d) = output.constant(T(0));
  }
};

template <typename Device, typename T, int NDIMS>
struct StridedSliceGrad {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    InitOutput<T, NDIMS, Device>::run(d, output);
    if (input.size() == 0) {
      return;
    }

    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).stridedSlice(start_i, stop_i, strides_i).device(d) =
          input;
    } else {
      output.stridedSlice(start_indices, stop_indices, strides).device(d) =
          input;
    }
  }
};

template <typename Device, typename T, int NDIMS>
struct StridedSliceAssign {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& start_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& stop_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& strides) {
    const bool use_64bit = input.size() > Eigen::NumTraits<int>::highest();
    if (!use_64bit) {
      Eigen::DSizes<int, NDIMS> start_i, stop_i, strides_i;
      for (int i = 0; i < NDIMS; ++i) {
        start_i[i] = start_indices[i];
        stop_i[i] = stop_indices[i];
        strides_i[i] = strides[i];
      }
      To32Bit(output).stridedSlice(start_i, stop_i, strides_i).device(d) =
          To32Bit(input);
    } else {
      output.stridedSlice(start_indices, stop_indices, strides).device(d) =
          input;
    }
  }
};

template <typename Device, typename T>
struct StridedSliceAssignScalar {
  void operator()(const Device& d, typename TTypes<T, 1>::Tensor output,
                  typename TTypes<T, 1>::ConstTensor input) {
    output.device(d) = input;
  }
};

}  // namespace functor
}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_STRIDED_SLICE_OP_H_