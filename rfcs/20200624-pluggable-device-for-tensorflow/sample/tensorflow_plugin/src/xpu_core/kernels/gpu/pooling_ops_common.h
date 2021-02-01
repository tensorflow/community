#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_POOLING_OPS_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_POOLING_OPS_COMMON_H_

#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/padding.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"
#include "tensorflow_plugin/src/xpu_core/util/types.h"

namespace intel_plugin {

using dnnl::memory;

struct PoolParameters {
  int depth;

  int tensor_in_planes;  // Pool3D
  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_planes;  // Pool3D
  int window_rows;
  int window_cols;
  int depth_window;

  int planes_stride;  // Pool3D
  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_planes;  // Pool3D
  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_P1;  // Pool3D
  int64 pad_P2;  // Pool3D
  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;
  PoolParameters()
      : depth(0),
        tensor_in_planes(0),
        tensor_in_cols(0),
        tensor_in_rows(0),
        tensor_in_batch(0),
        window_planes(0),
        window_rows(0),
        window_cols(0),
        depth_window(0),
        planes_stride(0),
        row_stride(0),
        col_stride(0),
        depth_stride(0),
        out_planes(0),
        out_height(0),
        out_width(0),
        out_depth(0),
        pad_P1(0),
        pad_P2(0),
        pad_left(0),
        pad_right(0),
        pad_top(0),
        pad_bottom(0),
        pad_depth(0),
        data_format(TensorFormat::FORMAT_NCHW) {}

  // Updates context->status if there is an invalid input.
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            const std::vector<int32>& padding_list, TensorFormat data_format,
            const TensorShape& tensor_in_shape);

 private:
  // Common initialization for TensorFlow and MKL formats
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            const std::vector<int32>& padding_list, TensorFormat data_format);
};

//////////////////////////////////////////////////////////////////////////
//           PoolingOpBase
//////////////////////////////////////////////////////////////////////////
struct PoolingOpBase {
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_tf_;
  dnnl::memory::format_tag data_format_mkldnn_;
  std::vector<int32> padding_list_;
};

template <typename T>
void PoolingOpBase_Create(void* kernel, TF_OpKernelConstruction* ctx) {
  auto op_kernel = static_cast<PoolingOpBase*>(kernel);
  OpKernelConstruction context(ctx);
  string data_format;

  OP_REQUIRES_OK(&context, context.GetAttr("data_format", &data_format));

  OP_REQUIRES(&context,
              FormatFromString(data_format, &op_kernel->data_format_tf_),
              errors::InvalidArgument("Invalid data format"));
  OP_REQUIRES_OK(&context, context.GetAttr("ksize", &op_kernel->ksize_));
  OP_REQUIRES(&context,
              op_kernel->ksize_.size() == 4 || op_kernel->ksize_.size() == 5,
              errors::InvalidArgument("Sliding window ksize field must "
                                      "specify 4 or 5 dimensions"));
  OP_REQUIRES_OK(&context, context.GetAttr("strides", &op_kernel->stride_));
  OP_REQUIRES(&context,
              op_kernel->stride_.size() == 4 || op_kernel->stride_.size() == 5,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 or 5 dimensions"));
  string padding;
  OP_REQUIRES_OK(&context, context.GetAttr("padding", &padding));
  if (padding == "VALID")
    op_kernel->padding_ = Padding::VALID;
  else if (padding == "SAME")
    op_kernel->padding_ = Padding::SAME;
  else
    op_kernel->padding_ = Padding::EXPLICIT;
  if (context.HasAttr("explicit_paddings")) {
    OP_REQUIRES_OK(&context, context.GetAttr("explicit_paddings",
                                             &op_kernel->padding_list_));
  }

  OP_REQUIRES(&context, op_kernel->ksize_[0] == 1 && op_kernel->stride_[0] == 1,
              errors::Unimplemented("Pooling is not yet supported on the "
                                    "batch dimension."));
  bool is_pool2d = (op_kernel->ksize_.size() == 4);
  MklTensorFormat tensor_format_mkldnn =
      is_pool2d ? TFDataFormatToMklDnnDataFormat(op_kernel->data_format_tf_)
                : TFDataFormatToMklDnn3DDataFormat(op_kernel->data_format_tf_);

  op_kernel->data_format_mkldnn_ =
      MklTensorFormatToMklDnnDataFormat(tensor_format_mkldnn);
}

template <typename T>
void PoolingOpBase_Compute(void* kernel, OpKernelContext* ctx) {}

// Calculate output shape of pooling op in MKL-DNN and TensorFlow order.
// MKL-DNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
// But TensorFlow output will be in NHWC/NCHW(Pool2D) or
// NDHWC/NCDHW(Pool3D) format depending on data format. Function expects
// output height and width to have already been int32 bounds-checked.
void PoolingOpBase_GetOutputDims(void* kernel,
                                 const PoolParameters& mkl_pool_params,
                                 TensorFormat tf_format,
                                 memory::dims* output_dims_mkl_order,
                                 TensorShape* output_tf_shape);

void PoolingOpBase_InitPoolParameters(void* kernel, OpKernelContext* context,
                                      PoolParameters* pool_params,
                                      const TensorShape& input_tensor_shape,
                                      const std::vector<int32>& padding_list);

void PoolingOpBase_PoolParamsToDims(const PoolParameters* pool_params,
                                    memory::dims* filter_dims,
                                    memory::dims* strides,
                                    memory::dims* padding_left,
                                    memory::dims* padding_right,
                                    bool is_pool2d);

void PoolingOpBase_AllocateEmptyOutputTensor(
    void* kernel, OpKernelContext* context, const int kOutputIndex,
    PoolParameters* pool_params, const memory::dims output_dims_mkl_order,
    Tensor** output_tensor);

//////////////////////////////////////////////////////////////////////////
//           PoolingForwardOpBase
//////////////////////////////////////////////////////////////////////////
struct PoolingForwardOpBase : public PoolingOpBase {
  const int kInputTensorIndexInput = 0;
  const int kOutputTensorIndexOutput = 0;
};

template <typename T>
void PoolingForwardOpBase_Create(void* kernel, TF_OpKernelConstruction* ctx) {
  PoolingOpBase_Create<T>(kernel, ctx);
}

template <typename T>
void PoolingForwardOpBase_Compute(void* kernel, OpKernelContext* ctx) {}

void PoolingForwardOpBase_AllocateOutputTensor(OpKernelContext* context,
                                               TensorShape* output_tf_shape,
                                               Tensor** output_tensor);

void PoolingForwardOpBase_SanityCheckInput(OpKernelContext* context,
                                           const Tensor& input_tensor);

//////////////////////////////////////////////////////////////////////////
//           PoolingOp
//////////////////////////////////////////////////////////////////////////
struct PoolingOp : public PoolingForwardOpBase {};

template <typename T>
void* PoolingOp_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel = new PoolingOp;
  PoolingForwardOpBase_Create<T>(kernel, ctx);
  return kernel;
}

void PoolingOp_Delete(void* kernel);

template <typename T, dnnl::algorithm algo>
void PoolingOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  OpKernelContext context(ctx);
  try {
    auto op_kernel = static_cast<PoolingOp*>(kernel);
    auto onednn_engine = CreateDnnlEngine(context);

    const Tensor& input_tensor =
        context.input(op_kernel->kInputTensorIndexInput);
    PoolingForwardOpBase_SanityCheckInput(&context, input_tensor);

    // Initialize variables for the pooling op.
    PoolParameters pool_params;
    // Check whether pooling is 2D or 3D.
    bool is_pool2d = (op_kernel->ksize_.size() == 4);
    // Get the input tensor and initialize the pooling parameters.
    TensorShape input_tensor_shape = input_tensor.shape();
    PoolingOpBase_InitPoolParameters(kernel, &context, &pool_params,
                                     input_tensor_shape,
                                     op_kernel->padding_list_);

    Tensor* output_tensor = nullptr;
    dnnl::memory::dims dst_dims;
    TensorShape tf_output_shape;
    PoolingOpBase_GetOutputDims(kernel, pool_params, op_kernel->data_format_tf_,
                                &dst_dims, &tf_output_shape);

    // If input is an empty tensor, allocate an empty output tensor.
    if (input_tensor.NumElements() == 0) {
      PoolingOpBase_AllocateEmptyOutputTensor(
          kernel, &context, op_kernel->kOutputTensorIndexOutput, &pool_params,
          dst_dims, &output_tensor);
      return;
    }
    PoolingForwardOpBase_AllocateOutputTensor(&context, &tf_output_shape,
                                              &output_tensor);
    DCHECK(output_tensor);

    dnnl::memory::dims filter_dims, strides, padding_left, padding_right;
    // Get src/filter/stride/padding information.
    PoolingOpBase_PoolParamsToDims(&pool_params, &filter_dims, &strides,
                                   &padding_left, &padding_right, is_pool2d);

    // Get the input memory descriptor.
    dnnl::memory::dims src_dims =
        is_pool2d ? TFShapeToMklDnnDimsInNCHW(input_tensor.shape(),
                                              op_kernel->data_format_tf_)
                  : TFShapeToMklDnnDimsInNCDHW(input_tensor.shape(),
                                               op_kernel->data_format_tf_);

    dnnl::prop_kind pooling_prop_kind = dnnl::prop_kind::forward_inference;
    dnnl::memory::desc src_md(src_dims, MklDnnType<T>(),
                              op_kernel->data_format_mkldnn_);
    dnnl::memory::desc dst_md(dst_dims, MklDnnType<T>(),
                              op_kernel->data_format_mkldnn_);

    dnnl::pooling_forward::desc fwd_desc(pooling_prop_kind, algo, src_md,
                                         dst_md, strides, filter_dims,
                                         padding_left, padding_right);
    dnnl::pooling_forward::primitive_desc fwd_pd(fwd_desc, onednn_engine);
    dnnl::pooling_forward fwd(fwd_pd);
    dnnl::primitive fwd_primitives(fwd);

    const T* src_data = input_tensor.flat<T>().data();
    T* dst_data = output_tensor->flat<T>().data();

    auto onednn_stream = CreateDnnlStream(context, onednn_engine);
    auto src_mem =
        CreateDnnlMemory(fwd_pd.src_desc(), onednn_engine,
                         static_cast<void*>(const_cast<T*>(src_data)));
    auto dst_mem = CreateDnnlMemory(fwd_pd.dst_desc(), onednn_engine,
                                    static_cast<void*>(dst_data));
    std::unordered_map<int, dnnl::memory> net_args(
        {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    fwd_primitives.execute(onednn_stream, net_args);

  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(&context, errors::Aborted("Operation received an exception:",
                                             error_msg));
  }
}
}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_POOLING_OPS_COMMON_H_
