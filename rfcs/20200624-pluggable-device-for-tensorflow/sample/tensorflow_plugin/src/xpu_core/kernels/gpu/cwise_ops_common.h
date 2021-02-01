#ifndef TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_COMMON_H_

#include <iostream>
#include <string>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/eigen_stream_device.h"
#include "tensorflow_plugin/src/xpu_core/device/gpu/gpu_device_plugin.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/cwise_ops_gradients.h"
#include "tensorflow_plugin/src/xpu_core/kernels/gpu/fill_functor.h"
#include "tensorflow_plugin/src/xpu_core/util/bcast.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_types.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace intel_plugin {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// Partial specialization of UnaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    if (in.size() > 0) {
      To32Bit(out).device(d) = To32Bit(in).unaryExpr(typename Functor::func());
    }
  }
};

// Partial specialization of BinaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor, int NDIMS, bool has_errors>
struct BinaryFunctor<GPUDevice, Functor, NDIMS, has_errors> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    To32Bit(out).device(d) =
        To32Bit(in0).binaryExpr(in1, typename Functor::func());
  }

  void Left(const GPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_left<Tout, Tin, Binary> Unary;
    // TODO(zhuwei): explicit broadcast as Accuracy issur in Eign broadcast, we
    // should fix it in the future and use below code
    // To32Bit(out).device(d) = To32Bit(in).unaryExpr(Unary(scalar.data()));
    constexpr int NumDims = Functor::tin_type::NumDimensions;
    static_assert(NumDims == 1, "Unexpected size");
    Eigen::Sizes<1> scalar_dim;
    out.device(d) = scalar.reshape(scalar_dim)
                        .broadcast(in.dimensions())
                        .binaryExpr(in, Binary());
  }

  void Right(const GPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<Tout, Tin, Binary> Unary;
    // TODO(zhuwei): explicit broadcast as Accuracy issur in Eign broadcast, we
    // should fix it in the future and use below code
    // To32Bit(out).device(d) = To32Bit(in).unaryExpr(Unary(scalar.data()));
    constexpr int NumDims = Functor::tin_type::NumDimensions;
    static_assert(NumDims == 1, "Unexpected size");
    Eigen::Sizes<1> scalar_dim;
    out.device(d) = in.binaryExpr(
        scalar.reshape(scalar_dim).broadcast(in.dimensions()), Binary());
  }

  void BCast(const GPUDevice& d,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if ((NDIMS == 2) && Functor::use_bcast_optimization &&
        use_bcast_optimization<T>::value) {
      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).binaryExpr(To32Bit(in1).broadcast(bcast1), func);
        return;
      }
      if (!bcast0_all_one && bcast1_all_one) {
        To32Bit(out).device(d) =
            To32Bit(in0).broadcast(bcast0).binaryExpr(To32Bit(in1), func);
        return;
      }
    }
    To32Bit(out).device(d) = To32Bit(in0).broadcast(bcast0).binaryExpr(
        To32Bit(in1).broadcast(bcast1), func);
  }
};

}  // namespace functor

typedef struct UnaryOp {
} UnaryOp;

template <typename Functor>
void* UnaryOp_Create(TF_OpKernelConstruction* ctx);
void UnaryOp_Delete(void* kernel);
template <typename Functor>
void UnaryOp_Compute(void* kernel, TF_OpKernelContext* ctx);

typedef struct BinaryOp {
  string op;
  bool incompatible_shape_error;
  bool has_attr;
} BinaryOp;

// helper class and functions for binaryOp
struct BinaryOpState {
  // Sets up bcast with the shape of in0 and in1, ensures that the bcast
  // is valid, and if so, set out, either by allocating a new buffer using
  // ctx->output(...) or by creating an alias for an owned input buffer for
  // in-place computation.
  // Caller must check ctx->status() upon return for non-ok status.
  // If ctx->status().ok() is true, then out is guaranteed to be allocated.
  explicit BinaryOpState(OpKernelContext* ctx, BinaryOp* kernel);

  const Tensor& in0;
  const Tensor& in1;
  BCast bcast;

  // BCast bcast;
  Tensor* out = nullptr;
  int64 out_num_elements;

  int64 in0_num_elements;
  int64 in1_num_elements;

  int ndims;
  bool result;
};

void SetUnimplementedError(OpKernelContext* ctx);
void SetComputeError(OpKernelContext* ctx, const string& op);

void* BinaryOp_Create(TF_OpKernelConstruction* ctx);
void BinaryOp_Delete(void* kernel);
template <typename Functor>
void BinaryOp_Compute(void* kernel, TF_OpKernelContext* ctx);

// TODO(ZhuWei): skip Signature checking, as cannot get input types and output
// types from OpKernelConstruction in function MatchSignature. It can be done
// after intergrating graph c api
template <typename Functor>
void* UnaryOp_Create(TF_OpKernelConstruction* ctx) {
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  // OpKernelConstruction context(ctx);
  // auto in = DataTypeToEnum<Tin>::v();
  // auto out = DataTypeToEnum<Tout>::v();
  // OP_REQUIRES_OK_PTR(&context, context.MatchSignature({in}, {out}));
  UnaryOp* kernel = new UnaryOp;
  return kernel;
}

template <typename Functor>
void UnaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  OpKernelContext context(ctx);
  const Tensor& inp = context.input(0);
  Tensor* out = nullptr;
  if (std::is_same<Tin, Tout>::value) {
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {0}, 0, inp.shape(), &out));
  } else {
    OP_REQUIRES_OK(&context, context.allocate_output(0, inp.shape(), &out));
  }
  functor::UnaryFunctor<GPUDevice, Functor>()(
      context.eigen_gpu_device(), out->flat<Tout>(), inp.flat<Tin>());
}

template <typename Functor>
void BinaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type
  OpKernelContext context(ctx);
  BinaryOp* op_kernel = static_cast<BinaryOp*>(kernel);
  const Tensor& input_0 = context.input(0);
  const Tensor& input_1 = context.input(1);
  const GPUDevice& eigen_device = context.eigen_gpu_device();
  bool error = false;
  bool* const error_ptr = Functor::has_errors ? &error : nullptr;

  // NOTE: Handle three simple cases before building the BinaryOpState, which
  // is relatively expensive for small operations.
  if (input_0.shape() == input_1.shape()) {
    // tensor op tensor with no broadcasting.
    Tensor* out;
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {0, 1}, 0, input_0.shape(), &out));
    functor::BinaryFunctor<GPUDevice, Functor, 1>()(
        eigen_device, out->template flat<Tout>(), input_0.template flat<Tin>(),
        input_1.template flat<Tin>(), error_ptr);
    if (Functor::has_errors && error) {
      SetComputeError(&context, op_kernel->op);
    }
    return;
  } else if (input_0.shape().dims() == 0) {
    // scalar op tensor.
    Tensor* out;
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {1}, 0, input_1.shape(), &out));

    functor::BinaryFunctor<GPUDevice, Functor, 1>().Left(
        eigen_device, out->template flat<Tout>(),
        input_0.template scalar<Tin>(), input_1.template flat<Tin>(),
        error_ptr);
    if (Functor::has_errors && error) {
      SetComputeError(&context, op_kernel->op);
    }
    return;
  } else if (input_1.shape().dims() == 0) {
    // tensor op scalar.
    Tensor* out;
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {0}, 0, input_0.shape(), &out));
    functor::BinaryFunctor<GPUDevice, Functor, 1>().Right(
        eigen_device, out->template flat<Tout>(), input_0.template flat<Tin>(),
        input_1.template scalar<Tin>(), error_ptr);
    if (Functor::has_errors && error) {
      SetComputeError(&context, op_kernel->op);
    }
    return;
  }

  // 'state': helper not dependent on T to reduce code size
  BinaryOpState state(&context, op_kernel);
  if (context.status().code() == TF_Code::TF_RESOURCE_EXHAUSTED) {
    // Stop when BinaryOpState's constructor failed due to OOM.
    return;
  }
  auto& bcast = state.bcast;
  Tensor* out = state.out;
  if (!bcast.IsValid()) {
    if (context.status().ok()) {
      if (state.result) {
        functor::SetOneFunctor<GPUDevice, bool>()(eigen_device,
                                                  out->flat<bool>());
      } else {
        functor::SetZeroFunctor<GPUDevice, bool>()(eigen_device,
                                                   out->flat<bool>());
      }
    }
    return;
  }

  auto& in0 = state.in0;
  auto& in1 = state.in1;
  if (state.out_num_elements == 0) {
    return;
  }

  const int ndims = state.ndims;
  if (ndims <= 1) {
    auto out_flat = out->flat<Tout>();
    if (state.in1_num_elements == 1) {
      // tensor op scalar
      functor::BinaryFunctor<GPUDevice, Functor, 1>().Right(
          eigen_device, out_flat, in0.template flat<Tin>(),
          in1.template scalar<Tin>(), error_ptr);
    } else if (state.in0_num_elements == 1) {
      // scalar op tensor
      functor::BinaryFunctor<GPUDevice, Functor, 1>().Left(
          eigen_device, out_flat, in0.template scalar<Tin>(),
          in1.template flat<Tin>(), error_ptr);
    } else {
      functor::BinaryFunctor<GPUDevice, Functor, 1>()(
          eigen_device, out_flat, in0.template flat<Tin>(),
          in1.template flat<Tin>(), error_ptr);
    }
  } else if (ndims == 2) {
    functor::BinaryFunctor<GPUDevice, Functor, 2>().BCast(
        eigen_device, out->shaped<Tout, 2>(bcast.result_shape()),
        in0.template shaped<Tin, 2>(bcast.x_reshape()),
        BCast::ToIndexArray<2>(bcast.x_bcast()),
        in1.template shaped<Tin, 2>(bcast.y_reshape()),
        BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);
  } else if (ndims == 3) {
    functor::BinaryFunctor<GPUDevice, Functor, 3>().BCast(
        eigen_device, out->shaped<Tout, 3>(bcast.result_shape()),
        in0.template shaped<Tin, 3>(bcast.x_reshape()),
        BCast::ToIndexArray<3>(bcast.x_bcast()),
        in1.template shaped<Tin, 3>(bcast.y_reshape()),
        BCast::ToIndexArray<3>(bcast.y_bcast()), error_ptr);
  } else if (ndims == 4) {
    functor::BinaryFunctor<GPUDevice, Functor, 4>().BCast(
        eigen_device, out->shaped<Tout, 4>(bcast.result_shape()),
        in0.template shaped<Tin, 4>(bcast.x_reshape()),
        BCast::ToIndexArray<4>(bcast.x_bcast()),
        in1.template shaped<Tin, 4>(bcast.y_reshape()),
        BCast::ToIndexArray<4>(bcast.y_bcast()), error_ptr);
  } else if (ndims == 5) {
    functor::BinaryFunctor<GPUDevice, Functor, 5>().BCast(
        eigen_device, out->shaped<Tout, 5>(bcast.result_shape()),
        in0.template shaped<Tin, 5>(bcast.x_reshape()),
        BCast::ToIndexArray<5>(bcast.x_bcast()),
        in1.template shaped<Tin, 5>(bcast.y_reshape()),
        BCast::ToIndexArray<5>(bcast.y_bcast()), error_ptr);
  } else {
    SetUnimplementedError(&context);
  }
  if (Functor::has_errors && error) {
    SetComputeError(&context, op_kernel->op);
  }
}

typedef struct SimpleBinaryOp {
} SimpleBinaryOp;

void* SimpleBinaryOp_Create(TF_OpKernelConstruction* ctx);
void SimpleBinaryOp_Delete(void* kernel);

template <typename Functor>
void SimpleBinaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type
  OpKernelContext context(ctx);
  SimpleBinaryOp* op_kernel = static_cast<SimpleBinaryOp*>(kernel);
  const Tensor& in0 = context.input(0);
  const Tensor& in1 = context.input(1);
  auto in0_flat = in0.flat<Tin>();
  auto in1_flat = in1.flat<Tin>();
  const GPUDevice& eigen_device = context.eigen_gpu_device();

  Tensor* out = nullptr;
  if (std::is_same<Tin, Tout>::value) {
    OP_REQUIRES_OK(&context, context.forward_input_or_allocate_output(
                                 {0, 1}, 0, in0.shape(), &out));
  } else {
    OP_REQUIRES_OK(&context, context.allocate_output(0, in0.shape(), &out));
  }
  auto out_flat = out->flat<Tout>();
  functor::SimpleBinaryFunctor<GPUDevice, Functor>()(eigen_device, out_flat,
                                                     in0_flat, in1_flat);
}

typedef struct DnnBinaryOp {
} DnnBinaryOp;

void* DnnBinaryOp_Create(TF_OpKernelConstruction* ctx);

void DnnBinaryOp_Delete(void* kernel);

// Expand dimension size to `max_dim`, the new dimensions will be added to
// left: 4 -- > 1x4.
// It follows Numpy broadcast rule:
// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
void DnnBinaryOp_ExpandDim(dnnl::memory::dims& dims, int max_dim);

bool DnnBinaryOp_UnsupportShape(const TensorShape& shape0,
                                const TensorShape& shape1);

template <typename Functor, typename T>
bool DnnBinaryOp_ShouldFallback(const TensorShape& shape0,
                                const TensorShape& shape1) {
  // oneDNN doesn't support Sub and SuqaredDiff yet.
  if (std::is_same<Functor, functor::sub<T>>::value) {
    return true;
  }
  if (DnnBinaryOp_UnsupportShape(shape0, shape1)) return true;
  return false;
}

template <typename Functor, dnnl::algorithm alg_kind, typename T>
void DnnBinaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto onednn_engine = CreateDnnlEngine(context);
    const int kNumInputs = 2;
    std::vector<TensorShape> tf_input_shapes(kNumInputs);
    std::vector<const T*> inputs_data(kNumInputs);

    for (int i = 0; i < kNumInputs; i++) {
      const Tensor& input = context.input(i);
      inputs_data[i] = input.flat<T>().data();
      tf_input_shapes[i] = input.shape();
    }

    if (!DnnBinaryOp_ShouldFallback<Functor, T>(tf_input_shapes[0],
                                                tf_input_shapes[1])) {
      std::vector<dnnl::memory::dims> srcs_dims(kNumInputs);
      for (int i = 0; i < kNumInputs; i++) {
        srcs_dims[i] = TFShapeToMklDnnDims(tf_input_shapes[i]);
      }

      // oneDNN only supports inputs[1] bcast to inputs[0]. So if inputs[1]
      // has more elements than inputs[0], swap the 2 inputs.
      // Use an index to indicate the swapped result.
      const int kFirst = (tf_input_shapes[1].num_elements() >
                          tf_input_shapes[0].num_elements()) ||
                         (tf_input_shapes[1].num_elements() ==
                              tf_input_shapes[0].num_elements() &&
                          (srcs_dims[1].size() > srcs_dims[0].size() ||
                           tf_input_shapes[0].dims() == 0));
      const int kSecond = 1 - kFirst;

      // oneDNN only supports inputs with same rank size, so expand dimension
      // if they are not consistent.
      // E.g. 8x4 * 4 --> 8x4 * 1x4.
      if (srcs_dims[0].size() != srcs_dims[1].size()) {
        const int kSmall = srcs_dims[0].size() > srcs_dims[1].size();
        DnnBinaryOp_ExpandDim(srcs_dims[kSmall], srcs_dims[1 - kSmall].size());
      }

      std::vector<dnnl::memory::desc> srcs_md(kNumInputs);
      for (int i = 0; i < kNumInputs; i++) {
        auto src_strides = CalculateTFStrides(srcs_dims[i]);
        srcs_md[i] = CreateBlockedMemDesc<T>(srcs_dims[i], src_strides);
      }

      std::shared_ptr<dnnl::binary::desc> desc(new dnnl::binary::desc(
          alg_kind, srcs_md[kFirst], srcs_md[kSecond], srcs_md[kFirst]));
      std::shared_ptr<dnnl::binary::primitive_desc> binary_pd(
          new dnnl::binary::primitive_desc(*desc, onednn_engine));
      std::shared_ptr<dnnl::primitive> binary_primitive(
          new dnnl::binary(*binary_pd));

      Tensor* dst_tensor = nullptr;
      OP_REQUIRES_OK(&context, context.allocate_output(
                                   0, tf_input_shapes[kFirst], &dst_tensor));
      DCHECK(dst_tensor != nullptr) << "Output tensor pointer is NULL";

      auto src0_mem = CreateDnnlMemory(
          binary_pd->src_desc(0), onednn_engine,
          static_cast<void*>(const_cast<T*>(inputs_data[kFirst])));
      auto src1_mem = CreateDnnlMemory(
          binary_pd->src_desc(1), onednn_engine,
          static_cast<void*>(const_cast<T*>(inputs_data[kSecond])));
      auto dst_mem =
          CreateDnnlMemory(binary_pd->dst_desc(), onednn_engine,
                           static_cast<void*>(dst_tensor->flat<T>().data()));

      std::unordered_map<int, dnnl::memory> net_args;
      net_args.insert({DNNL_ARG_SRC_0, src0_mem});
      net_args.insert({DNNL_ARG_SRC_1, src1_mem});
      net_args.insert({DNNL_ARG_DST, dst_mem});

      auto onednn_stream = CreateDnnlStream(context, onednn_engine);
      binary_primitive->execute(onednn_stream, net_args);
    } else {
      BinaryOp_Compute<Functor>(kernel, ctx);
    }
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
}
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CWISE_OPS_COMMON_H_
