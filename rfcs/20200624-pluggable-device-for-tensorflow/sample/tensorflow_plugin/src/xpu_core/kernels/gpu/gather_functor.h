#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_GATHER_OP_FUNCTOR_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_GATHER_OP_FUNCTOR_H_

#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, bool is_axis_zero>
struct GatherOpKernel {
  GatherOpKernel(const T* params, const Index* indices, T* out,
                 int64 gather_dim_size, int64 indices_size, int64 slice_size,
                 int64 out_size)
      : params(params),
        indices(indices),
        out(out),
        gather_dim_size_(gather_dim_size),
        indices_size_(indices_size),
        slice_size_(slice_size),
        out_size_(out_size) {}

  void operator()(cl::sycl::nd_item<1> item) {
    auto i = item.get_global_id()[0];
    if (i >= out_size_) return;
    Index batch_i = 0;
    Index indices_i = 0;
    Index slice_i = 0;
    if (is_axis_zero) {
      indices_i = i / slice_size_;
      slice_i = i - indices_i * slice_size_;
    } else {
      Index batch_indices_i = i / slice_size_;
      // The batch index into params to use for i.
      batch_i = batch_indices_i / indices_size_;
      // The index into indices to use for i.
      indices_i = batch_indices_i - batch_i * indices_size_;
      // Index into the current slice in params to use for i.
      slice_i = i - batch_indices_i * slice_size_;
    }

    // Index into the gather axis to use for i.
    // Index gather_i = ldg(indices + indices_i);
    Index gather_i = indices[indices_i];

    // Check gather_i is in [0, gather_dim_size_).
    if (!FastBoundsCheck(gather_i, gather_dim_size_)) {
      // Set indices out of range to zero
      // TODO(fpmc): Log an error for transfer back to host.
      out[i] = T(0);
    } else {
      // params is a [batch_size, gather_dim_size_, slice_size_] tensor. Read
      // params[batch_i, gather_i, slice_i] and write it to the i'th position in
      // out.
      Index params_i =
          (batch_i * gather_dim_size_ + gather_i) * slice_size_ + slice_i;
      out[i] = params[params_i];
    }
  }

 private:
  const T* params;
  const Index* indices;
  T* out;
  int64 gather_dim_size_;
  int64 indices_size_;
  int64 slice_size_;
  int64 out_size_;
};

template <typename T, typename Index, bool is_axis_zero>
class GatherKernel;

template <typename T, typename Index, bool is_axis_zero>
void LaunchGatherKernel(const gpuStream_t& stream, const int32 num_workgroup,
                        const int32 workgroup_size, const T* params,
                        const Index* indices, T* out, int64 gather_dim_size,
                        const int64 indices_size, const int64 slice_size,
                        const int64 out_size) {
  stream->submit([&](cl::sycl::handler& cgh) {
    GatherOpKernel<T, Index, is_axis_zero> task(params, indices, out,
                                                gather_dim_size, indices_size,
                                                slice_size, out_size);
    cgh.parallel_for<GatherKernel<T, Index, is_axis_zero>>(
        cl::sycl::nd_range<1>(
            cl::sycl::range<1>(num_workgroup * workgroup_size),
            cl::sycl::range<1>(workgroup_size)),
        task);
  });
}

namespace functor {

template <typename Device, typename T, typename Index>
struct GatherFunctor {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctor<GPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_axis_zero = params.dimension(0) == 1;
    const int64 gather_dim_size = params.dimension(1);
    const int64 indices_size = indices.size();
    const int64 slice_size = params.dimension(2);

    auto& stream = d.stream();
    auto total_threads =
        (*stream)
            .get_device()
            .template get_info<cl::sycl::info::device::max_work_group_size>();
    auto num_workgroups = (out_size + total_threads - 1) / total_threads;

    if (is_axis_zero) {
      LaunchGatherKernel<T, Index, true>(
          d.stream(), num_workgroups, total_threads, params.data(),
          indices.data(), out.data(), gather_dim_size, indices_size, slice_size,
          out_size);
    } else {
      LaunchGatherKernel<T, Index, false>(
          d.stream(), num_workgroups, total_threads, params.data(),
          indices.data(), out.data(), gather_dim_size, indices_size, slice_size,
          out_size);
    }
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indicies out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

}  // namespace functor
}  // namespace intel_plugin

#endif
