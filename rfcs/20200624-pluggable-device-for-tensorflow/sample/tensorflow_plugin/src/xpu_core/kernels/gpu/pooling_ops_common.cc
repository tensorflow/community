#include "tensorflow_plugin/src/xpu_core/kernels/gpu/pooling_ops_common.h"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/common_shape_fns.h"

namespace intel_plugin {
using dnnl::prop_kind;
// Initialization for TensorFlow format
void PoolParameters::Init(OpKernelContext* context,
                          const std::vector<int32>& ksize,
                          const std::vector<int32>& stride, Padding padding,
                          const std::vector<int32>& padding_list,
                          TensorFormat data_format,
                          const TensorShape& tensor_in_shape) {
  // For max pooling, tensor_in should have 4 or 5 dimensions.
  OP_REQUIRES(context,
              tensor_in_shape.dims() == 4 || tensor_in_shape.dims() == 5,
              errors::InvalidArgument("tensor_in must be 4 or 5-dimensional"));

  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  if (tensor_in_shape.dims() == 4) {
    // Pool2D
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  } else {
    // Pool3D
    tensor_in_planes = GetTensorDim(tensor_in_shape, data_format, '0');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, '1');
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, '2');
  }
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  Init(context, ksize, stride, padding, padding_list, data_format);
}

// Common Initialization for TensorFlow and MKL formats.
void PoolParameters::Init(OpKernelContext* context,
                          const std::vector<int32>& ksize,
                          const std::vector<int32>& stride, Padding padding,
                          const std::vector<int32>& padding_list,
                          TensorFormat data_format) {
  // Get the data format.
  this->data_format = data_format;

  bool is_pool2d = (ksize.size() == 4);
  if (is_pool2d) {
    // Pool2D
    // Get the output sizes.
    window_rows = GetTensorDim(ksize, data_format, 'H');
    window_cols = GetTensorDim(ksize, data_format, 'W');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    row_stride = GetTensorDim(stride, data_format, 'H');
    col_stride = GetTensorDim(stride, data_format, 'W');
    depth_stride = GetTensorDim(stride, data_format, 'C');

    // We only support 2D pooling across width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
                errors::Unimplemented(
                    "MaxPooling supports exactly one of pooling across depth "
                    "or pooling across width/height."));
  } else {
    // Pool3D
    // Get the output sizes.
    window_planes = GetTensorDim(ksize, data_format, '0');
    window_rows = GetTensorDim(ksize, data_format, '1');
    window_cols = GetTensorDim(ksize, data_format, '2');
    depth_window = GetTensorDim(ksize, data_format, 'C');

    // Get the strides.
    planes_stride = GetTensorDim(stride, data_format, '0');
    row_stride = GetTensorDim(stride, data_format, '1');
    col_stride = GetTensorDim(stride, data_format, '2');
    depth_stride = GetTensorDim(stride, data_format, 'C');

    // We only support 3D pooling across depth/width/height and depthwise
    // pooling, not a combination.
    OP_REQUIRES(context,
                (depth_window == 1 ||
                 (window_rows == 1 && window_cols == 1 && window_planes == 1)),
                errors::Unimplemented(
                    "AvgPooling3D supports exactly one of pooling across depth "
                    "or pooling across depth/width/height."));
  }

  if (depth_window == 1) {  // We are pooling in the D (Pool3D only), H and W.
    if (!is_pool2d) {
      OP_REQUIRES_OK(
          context, GetWindowedOutputSizeVerbose(tensor_in_planes, window_planes,
                                                planes_stride, padding,
                                                &out_planes, &pad_P1, &pad_P2));
    }
    if (padding == Padding::EXPLICIT) {
      if (data_format == FORMAT_NHWC) {
        pad_top = static_cast<int64>(padding_list[2]);
        pad_left = static_cast<int64>(padding_list[4]);
        pad_bottom = static_cast<int64>(padding_list[3]);
        pad_right = static_cast<int64>(padding_list[5]);
      } else if (data_format == FORMAT_NCHW) {
        pad_top = static_cast<int64>(padding_list[4]);
        pad_left = static_cast<int64>(padding_list[6]);
        pad_bottom = static_cast<int64>(padding_list[5]);
        pad_right = static_cast<int64>(padding_list[7]);
      }
    }
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));

    // TF can work with int64, but dnnl only supports int32.
    // Fail if the depth, height or width are greater than MAX_INT.
    // We check depth only for 3D pooling case.
    if (!is_pool2d) {
      OP_REQUIRES(context,
                  FastBoundsCheck(out_planes, std::numeric_limits<int>::max()),
                  errors::InvalidArgument("output depth/planes is too large"));
    }

    OP_REQUIRES(context,
                FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output height is too large"));

    OP_REQUIRES(context,
                FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output width is too large"));

    out_depth = depth;  // Output will have the same depth as the input.
  } else {              // We are pooling in the depth dimension.
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the depth
    // stride (no overlapping).
    OP_REQUIRES(context, depth % depth_window == 0,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to evenly divide the"
                                      " input depth"));
    OP_REQUIRES(context, depth_stride == depth_window,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to equal the depth"
                                      " stride"));
    out_depth = depth / depth_window;
  }
}

//////////////////////////////////////////////////////////////////////////
//           PoolingOpBase
//////////////////////////////////////////////////////////////////////////

// Calculate output shape of pooling op in MKL-DNN and TensorFlow order.
// MKL-DNN uses NCHW(Pool2D) or NCDHW(Pool3D) for output order.
// But TensorFlow output will be in NHWC/NCHW(Pool2D) or
// NDHWC/NCDHW(Pool3D) format depending on data format. Function expects
// output height and width to have already been int32 bounds-checked.
void PoolingOpBase_GetOutputDims(void* kernel,
                                 const PoolParameters& mkl_pool_params,
                                 TensorFormat tf_format,
                                 memory::dims* output_dims_mkl_order,
                                 TensorShape* output_tf_shape) {
  auto op_kernel = static_cast<PoolingOpBase*>(kernel);
  if (op_kernel->ksize_.size() == 4) {
    // Pooling2D: MKL-DNN always needs output in NCHW format.
    *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                              mkl_pool_params.out_depth,
                              static_cast<int>(mkl_pool_params.out_height),
                              static_cast<int>(mkl_pool_params.out_width)};

    if (tf_format == TensorFormat::FORMAT_NCHW) {
      output_tf_shape->AddDim(mkl_pool_params.tensor_in_batch);
      output_tf_shape->AddDim(mkl_pool_params.out_depth);
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_height));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_width));
    } else {
      output_tf_shape->AddDim(mkl_pool_params.tensor_in_batch);
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_height));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_width));
      output_tf_shape->AddDim(mkl_pool_params.out_depth);
    }
  } else {
    // Pooling3D: MKL-DNN always needs output in NCDHW format.
    *output_dims_mkl_order = {mkl_pool_params.tensor_in_batch,
                              mkl_pool_params.out_depth,
                              static_cast<int>(mkl_pool_params.out_planes),
                              static_cast<int>(mkl_pool_params.out_height),
                              static_cast<int>(mkl_pool_params.out_width)};
    if (tf_format == TensorFormat::FORMAT_NCHW) {
      output_tf_shape->AddDim(mkl_pool_params.tensor_in_batch);
      output_tf_shape->AddDim(mkl_pool_params.out_depth);
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_planes));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_height));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_width));
    } else {
      output_tf_shape->AddDim(mkl_pool_params.tensor_in_batch);
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_planes));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_height));
      output_tf_shape->AddDim(static_cast<int>(mkl_pool_params.out_width));
      output_tf_shape->AddDim(mkl_pool_params.out_depth);
    }
  }
}

void PoolingOpBase_InitPoolParameters(void* kernel, OpKernelContext* context,
                                      PoolParameters* pool_params,
                                      const TensorShape& input_tensor_shape,
                                      const std::vector<int32>& padding_list) {
  auto op_kernel = static_cast<PoolingOpBase*>(kernel);
  pool_params->Init(context, op_kernel->ksize_, op_kernel->stride_,
                    op_kernel->padding_, padding_list,
                    op_kernel->data_format_tf_, input_tensor_shape);
}

void PoolingOpBase_PoolParamsToDims(const PoolParameters* pool_params,
                                    memory::dims* filter_dims,
                                    memory::dims* strides,
                                    memory::dims* padding_left,
                                    memory::dims* padding_right,
                                    bool is_pool2d) {
  if (is_pool2d) {
    // Pool2D
    *filter_dims =
        memory::dims({pool_params->window_rows, pool_params->window_cols});
    *strides = memory::dims({pool_params->row_stride, pool_params->col_stride});
    *padding_left = memory::dims({static_cast<int>(pool_params->pad_top),
                                  static_cast<int>(pool_params->pad_left)});
    *padding_right = memory::dims({static_cast<int>(pool_params->pad_bottom),
                                   static_cast<int>(pool_params->pad_right)});
  } else {
    // Pool3D
    *filter_dims =
        memory::dims({pool_params->window_planes, pool_params->window_rows,
                      pool_params->window_cols});
    *strides = memory::dims({pool_params->planes_stride,
                             pool_params->row_stride, pool_params->col_stride});

    *padding_left = memory::dims({static_cast<int>(pool_params->pad_P1),
                                  static_cast<int>(pool_params->pad_top),
                                  static_cast<int>(pool_params->pad_left)});
    *padding_right = memory::dims({static_cast<int>(pool_params->pad_P2),
                                   static_cast<int>(pool_params->pad_bottom),
                                   static_cast<int>(pool_params->pad_right)});
  }
}

void PoolingOpBase_AllocateEmptyOutputTensor(
    void* kernel, OpKernelContext* context, const int kOutputIndex,
    PoolParameters* pool_params, const memory::dims output_dims_mkl_order,
    Tensor** output_tensor) {
  auto op_kernel = static_cast<PoolingOpBase*>(kernel);
  TensorShape output_tf_shape;
  if (pool_params->data_format == TensorFormat::FORMAT_NCHW) {
    output_tf_shape = MklDnnDimsToTFShape(output_dims_mkl_order);
  } else {
    memory::dims output_dims_order;
    // determine Pooling2D (NHWC) or Pooling3D (NDHWC)
    if (op_kernel->ksize_.size() == 4) {
      output_dims_order = {pool_params->tensor_in_batch,
                           static_cast<int>(pool_params->out_height),
                           static_cast<int>(pool_params->out_width),
                           pool_params->out_depth};
    } else {
      output_dims_order = {pool_params->tensor_in_batch,
                           static_cast<int>(pool_params->out_planes),
                           static_cast<int>(pool_params->out_height),
                           static_cast<int>(pool_params->out_width),
                           pool_params->out_depth};
    }
    output_tf_shape = MklDnnDimsToTFShape(output_dims_order);
  }
  OP_REQUIRES_OK(context, context->allocate_output(
                              kOutputIndex, output_tf_shape, output_tensor));
  DCHECK(output_tensor);
}

//////////////////////////////////////////////////////////////////////////
//           PoolingForwardOpBase
//////////////////////////////////////////////////////////////////////////
void PoolingForwardOpBase_AllocateOutputTensor(OpKernelContext* context,
                                               TensorShape* output_tf_shape,
                                               Tensor** output_tensor) {
  DCHECK(output_tensor);
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, *output_tf_shape, output_tensor));
  DCHECK(*output_tensor);
}

void PoolingForwardOpBase_SanityCheckInput(OpKernelContext* context,
                                           const Tensor& input_tensor) {
  OP_REQUIRES(context, input_tensor.dims() == 4 || input_tensor.dims() == 5,
              errors::InvalidArgument("Input must be 4 or 5-dimensional"));
}

//////////////////////////////////////////////////////////////////////////
//           PoolingOp
//////////////////////////////////////////////////////////////////////////
void PoolingOp_Delete(void* kernel) {
  if (kernel) delete static_cast<PoolingOp*>(kernel);
}
}  // namespace intel_plugin
