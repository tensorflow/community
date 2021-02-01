#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SEGMENT_REDUCTION_OP_FUNCTOR_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_SEGMENT_REDUCTION_OP_FUNCTOR_H_

#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/register_types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

// Functor for SegmentSumGPUOp.
// output_rows: the number of output segments (unique segment ids in
//                'segment_ids').
// segment_ids_shape: shape of 'segment_ids' tensor.
// segment_ids: unsorted map from input to output segment ids at which to
//                perform segment sum operation.
// data_size: size of input data tensor.
// data: input data tensor.
// output: output reshaped to {output_rows, output.size/output_rows}
template <typename Device, typename T, typename Index>
struct SegmentSumFunctor {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
};

template <typename Device, typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output);
};

// Note: All the below reduction method should avoid race condition by yourself.

template <typename T>
struct SumOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest += *value;
  }
};

template <typename T>
struct ProdOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest *= *value;
  }
};

template <typename T>
struct MaxOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest = *dest > *value ? *dest : *value;
  }
};

template <typename T>
struct MinOpGpu {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void operator()(T* dest,
                                                        const T* value) {
    *dest = *dest < *value ? *dest : *value;
  }
};

// initial value functors
template <typename T>
struct Zero {
  EIGEN_STRONG_INLINE T operator()() const { return T(0); }
};

template <typename T>
struct One {
  EIGEN_STRONG_INLINE T operator()() const { return T(1); }
};

template <typename T>
struct Lowest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::lowest();
  }
};

template <typename T>
struct Highest {
  EIGEN_STRONG_INLINE T operator()() const {
    return Eigen::NumTraits<T>::highest();
  }
};

namespace impl {

template <typename T, typename Index, typename KernelReductionFunctor>
class UnsortedKernel;

// as sycl do not have atomic for double type
// we change this algo to one column one thread
template <typename T, typename Index, typename KernelReductionFunctor>
Status UnsortedSegmentCustomKernel(const GPUDevice& device,
                                   const Index input_outer_dim_size,
                                   const Index inner_dim_size,
                                   const Index output_outer_dim_size,
                                   const Index* segment_ids, const T* input,
                                   T* output) {
  auto stream = device.stream();
  auto group_size =
      (*stream)
          .get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>();

  // The algorithm is different with CUDA Kernel because DPCPP doesn't support
  // atomic reduction operation on some data types, such as float.
  // The function of segment reduction(sum) can be illustrated as below,
  //
  //      +---------------------+
  // +----+ 3 | 1 |   |   |     |
  // |    +---------------------+
  // |
  // |    +---------------------+
  // |    | 4 | 7 |   |   |     +----+
  // |    +---------------------+    |
  // |                               |
  // |    +---------------------+    |
  // +----> 7 | 8 |   |   |     <----+
  //      +---------------------+
  //
  // The difference is
  //   +  CUDA will use all threads to compute all items based on the input. So
  //      the dest will be used at the same time, such as &3 -> &7 and &4 -> &7.
  //      That's why they require to use atomic reduction operation.
  //   +  DPCPP will use the algorithm based on the output row to parallel the
  //      computing. It means **only one thread to write the output, maybe
  //      multiple thread to read input values and not output reads in different
  //      threads**. So the max thread we can use is the distinct values of
  //      segment_ids (num_segment). Currently, it's the max group size.
  stream->submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<UnsortedKernel<T, Index, KernelReductionFunctor>>(
        cl::sycl::nd_range<1>(cl::sycl::range<1>(group_size),
                              cl::sycl::range<1>(group_size)),
        [=](cl::sycl::nd_item<1> item) {
          auto input_idx = item.get_global_id(0);
          KernelReductionFunctor redux_op;
          for (auto out_row_idx = input_idx;
               out_row_idx < output_outer_dim_size;
               out_row_idx += item.get_local_range(0)) {
            for (auto i = 0; i < input_outer_dim_size; ++i) {
              if (out_row_idx == segment_ids[i]) {
                // row reduction
                for (auto j = 0; j < inner_dim_size; ++j) {
                  redux_op(output + out_row_idx * inner_dim_size + j,
                           input + i * inner_dim_size + j);
                }
                item.mem_fence();
              }
            }
          }
        });
  });
  return Status::OK();
}

}  // end namespace impl

template <typename T, typename Index>
struct SegmentSumFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }
    // Set 'output' to initial value.
    auto init = T(0);
    output.device(d) = output.constant(init);
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }
    int num_segments = 1;
    for (int i = 1; i < segment_ids.dimension(0); i++) {
      if (segment_ids(i) != segment_ids(i - 1)) {
        num_segments += 1;
      }
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;

    auto status =
        impl::UnsortedSegmentCustomKernel<T, Index, functor::SumOpGpu<T>>(
            d, input_outer_dim_size, input_inner_dim_size, num_segments,
            segment_ids.data(), data, output.data());
  }
};

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const Index num_segments,
                  const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }

    // Set 'output' to initial value.
    const GPUDevice& d = ctx->eigen_gpu_device();
    auto init = InitialValueF()();
    output.device(d) = output.constant(init);
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = data_size / input_outer_dim_size;
    auto status = impl::UnsortedSegmentCustomKernel<T, Index, ReductionF>(
        d, input_outer_dim_size, input_inner_dim_size, num_segments,
        segment_ids.data(), data, output.data());
  }
};

// #define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index) \
//   template struct SegmentSumFunctor<GPUDevice, T, Index>

// #define DEFINE_SORTED_GPU_SPECS(T)         \
//   DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
//   DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

// TF_CALL_FLOAT_TYPES(DEFINE_SORTED_GPU_SPECS);
// TF_CALL_int32(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Lowest<T>, functor::MaxOpGpu<T>>;          \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Highest<T>, functor::MinOpGpu<T>>;         \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::ProdOpGpu<T>>;

// sum is the only op that supports all input types currently
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index) \
  template struct UnsortedSegmentFunctor<             \
      GPUDevice, T, Index, functor::Zero<T>, functor::SumOpGpu<T>>;

#define DEFINE_REAL_GPU_SPECS(T)                  \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int64);

#define DEFINE_SUM_GPU_SPECS(T)                  \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_FLOAT_TYPES(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
TF_CALL_FLOAT_TYPES(DEFINE_SUM_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);

#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

}  // namespace functor
}  // namespace intel_plugin
#endif
