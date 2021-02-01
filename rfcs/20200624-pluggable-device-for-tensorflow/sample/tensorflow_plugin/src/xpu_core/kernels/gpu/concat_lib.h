
#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CONCAT_LIB_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CONCAT_LIB_H_

#include <memory>
#include <vector>

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

// Functors to concatenate tensors. These always take a rank-2 tensor (i.e a
// matrix) and concatenate it along the axis 1 ("putting them next to each
// other" as opposed to "putting them on top of one another").
//
// Any concatenation of n-dimensional tensors across any axis can be reduced to
// a concatenation of two-dimensional tensors across the axis 1 by first
// partitioning the axes of the original tensors into those less than the axis
// to be concatenated across and the rest. Then reshape the tensors into a
// two-dimensional tensor by collapsing these two sets of axes and concatenate
// the resulting matrices across the axis 1, finally reshaping the result to
// have the proper shape.
//
// So, for example, when stacking N tensors, reshape each to have shape
// {1, Numelements} and reshape the result matrix to have shape
// {1, N * NumElements} before passing it to this functor.

// Assumes all inputs are nonempty
template <typename T>
void Concat(
    OpKernelContext* ctx,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output);

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CONCAT_LIB_H_
