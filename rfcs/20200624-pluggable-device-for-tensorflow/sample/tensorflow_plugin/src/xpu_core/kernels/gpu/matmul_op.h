#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_MATMUL_OP_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_MATMUL_OP_H_

#include <string>
#include <vector>

#include "tensorflow/c/tf_status.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fill_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/bcast.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

using dnnl::memory;

namespace {
// Simple wrapper over BCast specialized for MatMul.
// Provides utilities for broadcasting across batch dimensions for binary
// MatMul-like operations.
class MatMulBCast {
 public:
  using Vec = BCast::Vec;

  MatMulBCast(Vec x, Vec y) {
    if (x.size() < 2 || y.size() < 2) return;
    x.resize(x.size() - 2);
    y.resize(y.size() - 2);

    batch_bcast_ = absl::make_unique<BCast>(std::move(x), std::move(y));
    if (!batch_bcast_->IsValid()) return;

    x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
    y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
    output_shape_ = TensorShape(batch_bcast_->output_shape());
    output_batch_size_ = output_shape_.num_elements();
    broadcasting_required_ =
        std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

    if (broadcasting_required_) {
      ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                          batch_bcast_->x_bcast(), &x_batch_indices_);
      ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                          batch_bcast_->y_bcast(), &y_batch_indices_);
    }
  }

  bool IsValid() const { return batch_bcast_ && batch_bcast_->IsValid(); }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  const int64 output_batch_size() const { return output_batch_size_; }
  const int64 x_batch_size() const { return x_batch_size_; }
  const int64 y_batch_size() const { return y_batch_size_; }
  const TensorShape& output_batch_shape() const { return output_shape_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& x_batch_indices() const { return x_batch_indices_; }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& y_batch_indices() const { return y_batch_indices_; }

 private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64 x_batch_size_;
  int64 y_batch_size_;
  TensorShape output_shape_;
  int64 output_batch_size_;
  std::vector<int64> x_batch_indices_;
  std::vector<int64> y_batch_indices_;
};

class DeviceMemoryWrapper {
 public:
  explicit DeviceMemoryWrapper(void* opaque) : opaque_(opaque) {}

  void* opaque() { return opaque_; }

 private:
  void* opaque_;  // Platform-dependent value representing allocated memory.
};

DeviceMemoryWrapper AsDeviceMemoryWrapper(const void* mem_ptr) {
  return DeviceMemoryWrapper(const_cast<void*>(mem_ptr));
}

}  // namespace

// dst(mb, m, n) = src(mb, m, k) * weight(mb, k, n)
template <typename T>
void MklBatchMatMul(OpKernelContext& ctx, void* src_handler,
                    void* weights_handler, void* dst_handler, bool adj_x,
                    bool adj_y, int64 m, int64 n, int64 k, int64 batch_size) {
  try {
    auto src_dims = memory::dims({batch_size, m, k});
    auto weight_dims = memory::dims({batch_size, k, n});
    auto dst_dims = memory::dims({batch_size, m, n});

    auto src_format = adj_x ? memory::format_tag::acb : memory::format_tag::abc;
    auto weight_format =
        adj_y ? memory::format_tag::acb : memory::format_tag::abc;
    auto dst_format = memory::format_tag::abc;

    auto src_md = memory::desc(src_dims, MklDnnType<T>(), src_format);
    auto weight_md = memory::desc(weight_dims, MklDnnType<T>(), weight_format);
    auto dst_md = memory::desc(dst_dims, MklDnnType<T>(), dst_format);

    auto dnnl_engine = CreateDnnlEngine(ctx);
    auto src_mem = CreateDnnlMemory(src_md, dnnl_engine, src_handler);
    auto weight_mem = CreateDnnlMemory(weight_md, dnnl_engine, weights_handler);
    auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine, dst_handler);

    std::shared_ptr<dnnl::matmul::desc> matmul_desc;
    matmul_desc.reset(new dnnl::matmul::desc(src_md, weight_md, dst_md));

    std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd;
    matmul_pd.reset(
        new dnnl::matmul::primitive_desc(*matmul_desc, dnnl_engine));

    auto matmul_primitive = dnnl::matmul(*matmul_pd);

    auto dnnl_stream = CreateDnnlStream(ctx, dnnl_engine);
    matmul_primitive.execute(dnnl_stream, {
                                              {DNNL_ARG_SRC, src_mem},
                                              {DNNL_ARG_WEIGHTS, weight_mem},
                                              {DNNL_ARG_DST, dst_mem},
                                          });
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(
        &ctx, errors::Aborted("Operation received an exception:", error_msg));
  }
}

template <typename T>
void DoBlasGemmBatchedInternal(
    OpKernelContext& ctx, bool adj_x, bool adj_y, int64 m, int64 n, int64 k,
    const std::vector<DeviceMemoryWrapper*>& a_ptrs_to_wrappers,
    const std::vector<DeviceMemoryWrapper*>& b_ptrs_to_wrappers,
    const std::vector<DeviceMemoryWrapper*>& c_ptrs_to_wrappers,
    int64 batch_count) {
  Tensor a;
  int64 a_size = batch_count * m * k * sizeof(T);
  OP_REQUIRES(
      &ctx,
      ctx.allocate_temp(DataTypeToEnum<uint8>::v(), TensorShape({a_size}), &a)
          .ok(),
      errors::Internal("Allocate Temp device memory error."));
  Tensor b;
  int64 b_size = batch_count * k * n * sizeof(T);
  OP_REQUIRES(
      &ctx,
      ctx.allocate_temp(DataTypeToEnum<uint8>::v(), TensorShape({b_size}), &b)
          .ok(),
      errors::Internal("Allocate Temp device memory error."));
  Tensor c;
  int64 c_size = batch_count * m * n * sizeof(T);
  OP_REQUIRES(
      &ctx,
      ctx.allocate_temp(DataTypeToEnum<uint8>::v(), TensorShape({c_size}), &c)
          .ok(),
      errors::Internal("Allocate Temp device memory error."));

  for (int i = 0; i < batch_count; i++) {
    dpcppMemcpyDtoDAsync(static_cast<T*>(a.data()) + i * m * k,
                         a_ptrs_to_wrappers[i]->opaque(), m * k * sizeof(T),
                         ctx.GetDeviceStream());
    dpcppMemcpyDtoDAsync(static_cast<T*>(b.data()) + i * k * n,
                         b_ptrs_to_wrappers[i]->opaque(), k * n * sizeof(T),
                         ctx.GetDeviceStream());
  }

  MklBatchMatMul<T>(ctx, a.data(), b.data(), c.data(), adj_x, adj_y, m, n, k,
                    batch_count);

  for (int i = 0; i < batch_count; i++) {
    dpcppMemcpyDtoDAsync(c_ptrs_to_wrappers[i]->opaque(),
                         static_cast<T*>(c.data()) + i * m * n,
                         m * n * sizeof(T), ctx.GetDeviceStream());
  }
}

template <typename T>
void LaunchBatchMatMul(OpKernelContext& context, const Tensor& src,
                       const Tensor& weights, bool adj_x, bool adj_y,
                       const MatMulBCast& bcast, Tensor* out) {
  const int64 batch_size = bcast.output_batch_size();

  std::vector<DeviceMemoryWrapper> a_device_memory;
  std::vector<DeviceMemoryWrapper> b_device_memory;
  std::vector<DeviceMemoryWrapper> c_device_memory;
  std::vector<DeviceMemoryWrapper*> a_ptrs;
  std::vector<DeviceMemoryWrapper*> b_ptrs;
  std::vector<DeviceMemoryWrapper*> c_ptrs;
  a_device_memory.reserve(bcast.x_batch_size());
  b_device_memory.reserve(bcast.y_batch_size());
  c_device_memory.reserve(batch_size);
  a_ptrs.reserve(batch_size);
  b_ptrs.reserve(batch_size);
  c_ptrs.reserve(batch_size);
  auto* a_base_ptr = src.template flat<T>().data();
  auto* b_base_ptr = weights.template flat<T>().data();
  auto* c_base_ptr = out->template flat<T>().data();

  const int64 m = src.dim_size(adj_x ? 2 : 1);
  const int64 k = src.dim_size(adj_x ? 1 : 2);
  const int64 n = weights.dim_size(adj_y ? 1 : 2);
  if (!bcast.IsBroadcastingRequired()) {
    for (int64 i = 0; i < batch_size; ++i) {
      a_device_memory.push_back(AsDeviceMemoryWrapper(a_base_ptr + i * m * k));
      b_device_memory.push_back(AsDeviceMemoryWrapper(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemoryWrapper(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }
  } else {
    const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
    const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
    for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
      a_device_memory.push_back(AsDeviceMemoryWrapper(a_base_ptr + i * m * k));
    }
    for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
      b_device_memory.push_back(AsDeviceMemoryWrapper(b_base_ptr + i * k * n));
    }
    for (int64 i = 0; i < batch_size; ++i) {
      c_device_memory.push_back(AsDeviceMemoryWrapper(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
      b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
      c_ptrs.push_back(&c_device_memory.back());
    }
  }

  DoBlasGemmBatchedInternal<T>(context, adj_x, adj_y, m, n, k, a_ptrs, b_ptrs,
                               c_ptrs, batch_size);
}
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_MATMUL_OP_H_
