
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/concat_lib.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/gpu_device_array.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

template <typename T>
class ConcatKernel;

template <typename T>
void ConcatImpl(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  GpuDeviceArrayOnHost<const T*> input_ptrs(c, inputs_flat.size());
  OP_REQUIRES_OK(c, input_ptrs.Init());
  for (size_t i = 0; i < inputs_flat.size(); ++i) {
    input_ptrs.Set(i, inputs_flat[i]->data());
  }
  OP_REQUIRES_OK(c, input_ptrs.Finalize());

  size_t split_size = inputs_flat[0]->dimension(1);
  size_t total_rows = output->dimension(0);
  size_t total_cols = output->dimension(1);

  auto* stream = c->GetDeviceStream();
  stream->submit([&](cl::sycl::handler& cgh) {
    GpuDeviceArrayStruct<const T*> input_ptr_data = input_ptrs.data();
    const T** input_ptrs_ptr =
        GetGpuDeviceArrayOnDevice<const T*>(&input_ptr_data);
    T* output_ptr = output->data();
    cgh.parallel_for<ConcatKernel<T>>(
        cl::sycl::range<2>{total_rows, total_cols},
        [=](cl::sycl::item<2> item) {
          auto row_id = item.get_id(0);
          auto col_id = item.get_id(1);
          auto split = col_id / split_size;
          const T* input_ptr = input_ptrs_ptr[split];
          auto col_offset = col_id % split_size;
          output_ptr[row_id * total_cols + col_id] =
              input_ptr[row_id * split_size + col_offset];
        });
  });
}

template <typename T>
void Concat(
    OpKernelContext* ctx,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  ConcatImpl<T>(ctx, inputs, output);
}

#define REGISTER_DPCPP(T)                                                     \
  template void Concat<T>(                                                    \
      OpKernelContext * ctx,                                                  \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

REGISTER_DPCPP(float)
REGISTER_DPCPP(Eigen::half)
REGISTER_DPCPP(Eigen::bfloat16)

#undef REGISTER_DPCPP
}  // namespace intel_plugin
