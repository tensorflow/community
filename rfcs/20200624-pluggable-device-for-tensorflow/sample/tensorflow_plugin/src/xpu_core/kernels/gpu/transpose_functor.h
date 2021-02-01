#ifndef TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_TRANSPOSE_FUNCTOR_H_
#define TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_TRANSPOSE_FUNCTOR_H_

#include <numeric>
#include <string>
#include <vector>

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/array_slice.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"

namespace intel_plugin {

typedef Eigen::GpuDevice EigenGPU;

// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename T, bool conjugate>
Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out);

// Implementation details.
namespace internal {
// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(const TensorShape& input_shape,
                                        const std::vector<int32>& permutation) {
  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dim_size(perm_dim) == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, bool conjugate,
                         Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

}  // namespace internal

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_TRANSPOSE_FUNCTOR_H_
