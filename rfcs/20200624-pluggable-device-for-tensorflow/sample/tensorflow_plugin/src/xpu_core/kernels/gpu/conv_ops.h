#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CONV_OPS_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_KERNELS_GPU_CONV_OPS_H_

#include <limits>
#include <memory>
#include <vector>

#include "dnnl.hpp"
#include "tensorflow_plugin/src/xpu_core/util/bounds_check.h"
#include "tensorflow_plugin/src/xpu_core/util/common_shape_fns.h"
#include "tensorflow_plugin/src/xpu_core/util/dnnl_util.h"
#include "tensorflow_plugin/src/xpu_core/util/errors.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/padding.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"

using dnnl::algorithm;

using ConvFwdDesc = dnnl::convolution_forward::desc;
using ConvFwdPd = dnnl::convolution_forward::primitive_desc;

namespace intel_plugin {

#define DNNL_SIZE_DTYPE long int

const int kInputIndex_Src = 0, kInputIndex_Filter = 1, kInputIndex_Bias = 2;
const int kOutputIndex_Dst = 0;
const bool kPadEnabled = false;
const float kReluUpBound = 0.0, kRelu6UpBound = 6.0, kEluUpBound = 1.0;

struct PostOpParam {
  string name;
  algorithm alg;
  std::vector<float> param;
};

typedef struct {
  TensorFormat data_format_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> dilations_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  bool is_depthwise_;

  /* Fused Conv */
  int num_args_;
  int InputIndex_Pad;
  bool fuse_add_;
  bool fuse_biasadd_;
  bool fuse_pad_;
  bool fuse_activation_;
  std::vector<string> fused_ops_;

  std::vector<PostOpParam> post_op_params_;
  float relu_up_bound_;
  algorithm activation_alg_;
} ConvOpBase;

class ConvUtil {
 protected:
  OpKernelContext* context_;  // We don't own this.
  ConvOpBase* kernel_;        // We don't own this.

 public:
  ConvUtil(OpKernelContext* context, ConvOpBase* kernel)
      : context_(context), kernel_(kernel) {}

  virtual ~ConvUtil() { context_ = nullptr; }

  // Calculate Convolution strides
  virtual inline void GetStrideDimension(dnnl::memory::dims* strides) {
    // For now we take the stride from the second and third dimensions only
    // (we do not support striding on the batch or depth dimension).
    OP_REQUIRES(context_, strides != nullptr,
                errors::InvalidArgument("strides shoud not be nullptr."));

    if (kernel_->strides_.size() == 4) {
      int stride_rows =
          GetTensorDim(kernel_->strides_, kernel_->data_format_, 'H');
      int stride_cols =
          GetTensorDim(kernel_->strides_, kernel_->data_format_, 'W');
      *strides = {stride_rows, stride_cols};
    } else if (kernel_->strides_.size() == 5) {
      int stride_planes =
          GetTensorDim(kernel_->strides_, kernel_->data_format_, '0');
      int stride_rows =
          GetTensorDim(kernel_->strides_, kernel_->data_format_, '1');
      int stride_cols =
          GetTensorDim(kernel_->strides_, kernel_->data_format_, '2');
      *strides = {stride_planes, stride_rows, stride_cols};
    }
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationDimension(dnnl::memory::dims* dilations) {
    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).
    OP_REQUIRES(context_, dilations != nullptr,
                errors::InvalidArgument("dilations shoud not be nullptr."));

    if (kernel_->dilations_.size() == 4) {
      int dilations_rows =
          GetTensorDim(kernel_->dilations_, kernel_->data_format_, 'H');
      int dilations_cols =
          GetTensorDim(kernel_->dilations_, kernel_->data_format_, 'W');
      *dilations = {dilations_rows, dilations_cols};
    } else if (kernel_->dilations_.size() == 5) {
      int dilations_planes =
          GetTensorDim(kernel_->dilations_, kernel_->data_format_, '0');
      int dilations_rows =
          GetTensorDim(kernel_->dilations_, kernel_->data_format_, '1');
      int dilations_cols =
          GetTensorDim(kernel_->dilations_, kernel_->data_format_, '2');
      *dilations = {dilations_planes, dilations_rows, dilations_cols};
    }
  }

  // Calculate Convolution input size in OneDNN order. OneDNN
  // requires input in NCHW/NCDHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetInputDimension(const TensorShape& input_shape,
                                        dnnl::memory::dims* input_dims) {
#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    OP_REQUIRES(context_, input_dims != nullptr,
                errors::InvalidArgument("input_dims shoud not be nullptr."));

    // Input channel
    int64 input_depth_raw =
        GetTensorDim(input_shape, kernel_->data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input batch
    int64 input_batch_raw =
        GetTensorDim(input_shape, kernel_->data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (kernel_->strides_.size() == 4) {  // NCHW format for Conv2D
      // Input rows/height
      int64 input_rows_raw =
          GetTensorDim(input_shape, kernel_->data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw =
          GetTensorDim(input_shape, kernel_->data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // OneDNN always requires input in NCHW format Conv2D.
      std::vector<DNNL_SIZE_DTYPE> input_dims_tmp(4, -1);
      input_dims_tmp[DimensionIndex::Dim_N] = input_batch;
      input_dims_tmp[DimensionIndex::Dim_C] = input_depth;
      input_dims_tmp[DimensionIndex::Dim_H] = input_rows;
      input_dims_tmp[DimensionIndex::Dim_W] = input_cols;

      *input_dims = input_dims_tmp;
    } else if (kernel_->strides_.size() == 5) {  // NCDHW format for Conv3D
      // Input planes/third-dimension
      int64 input_planes_raw =
          GetTensorDim(input_shape, kernel_->data_format_, '0');
      CHECK_BOUNDS(input_planes_raw, "Input depth too large");
      int input_planes = static_cast<int>(input_planes_raw);

      // Input rows/height
      int64 input_rows_raw =
          GetTensorDim(input_shape, kernel_->data_format_, '1');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw =
          GetTensorDim(input_shape, kernel_->data_format_, '2');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // OneDNN always requires input in NCDHW format for Conv3D.
      std::vector<DNNL_SIZE_DTYPE> input_dims_tmp(5, -1);
      input_dims_tmp[DimensionIndex3D::Dim3d_N] = input_batch;
      input_dims_tmp[DimensionIndex3D::Dim3d_C] = input_depth;
      input_dims_tmp[DimensionIndex3D::Dim3d_D] = input_planes;
      input_dims_tmp[DimensionIndex3D::Dim3d_H] = input_rows;
      input_dims_tmp[DimensionIndex3D::Dim3d_W] = input_cols;

      *input_dims = input_dims_tmp;
    }
#undef CHECK_BOUNDS
  }

  // Calculate Convolution filter size in OneDNN order.
  // OneDNN requires filter in OIHW (Conv2D) or OIDHW (Conv3D) format.
  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetFilterDimension(const TensorShape& input_shape,
                                         const TensorShape& filter_shape,
                                         dnnl::memory::dims* filter_dims) {
    OP_REQUIRES(context_, filter_dims != nullptr,
                errors::InvalidArgument("filter_dims shoud not be nullptr."));
    OP_REQUIRES(context_, filter_shape.dims() == kernel_->strides_.size(),
                errors::InvalidArgument((kernel_->strides_.size() == 4)
                                            ? "filter must be 4-dimensional: "
                                            : "filter must be 5-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < ((kernel_->strides_.size() == 4) ? 3 : 5); i++) {
      OP_REQUIRES(context_,
                  FastBoundsCheck(filter_shape.dim_size(i),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, kernel_->data_format_, 'C');

    if (kernel_->strides_.size() == 4) {  // Conv2D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(2),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(2)));

      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_O));
      // OneDNN always needs filter in OIHW format for regular convolutions
      // and GOIHW for grouped/depthwise convolutions,
      // OIHW = (out_depth, in_depth, rows, cols)
      // GOIHW = (group, out_depth, in_depth, rows, cols)
      // Specifically for depthwise G=filter_indepth, O=filter_outdepth, I=1
      if (kernel_->is_depthwise_) {
        std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(5, -1);
        filter_dims_tmp[GROUP_FILTER_DIM_G] = filter_in_depth;
        filter_dims_tmp[GROUP_FILTER_DIM_O] = filter_out_depth;
        filter_dims_tmp[GROUP_FILTER_DIM_I] = 1;
        filter_dims_tmp[GROUP_FILTER_DIM_H] = filter_rows;
        filter_dims_tmp[GROUP_FILTER_DIM_W] = filter_cols;
        *filter_dims = filter_dims_tmp;
      } else {
        std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(4, -1);
        filter_dims_tmp[DimensionIndex::Dim_O] = filter_out_depth;
        filter_dims_tmp[DimensionIndex::Dim_I] = filter_in_depth;
        filter_dims_tmp[DimensionIndex::Dim_H] = filter_rows;
        filter_dims_tmp[DimensionIndex::Dim_W] = filter_cols;
        *filter_dims = filter_dims_tmp;
      }
    } else {  // Conv3D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(3),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(3)));

      // TF filter is always in (planes, rows, cols, in_depth, out_depth) order.
      int filter_planes =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_P));
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_I));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_O));

      // OneDNN always needs filter in OIDHW format.
      // OIDHW = (out_depth, in_depth, planes, rows, cols)
      std::vector<DNNL_SIZE_DTYPE> filter_dims_tmp(5, -1);
      filter_dims_tmp[DimensionIndex3D::Dim3d_O] = filter_out_depth;
      filter_dims_tmp[DimensionIndex3D::Dim3d_I] = filter_in_depth;
      filter_dims_tmp[DimensionIndex3D::Dim3d_D] = filter_planes;
      filter_dims_tmp[DimensionIndex3D::Dim3d_H] = filter_rows;
      filter_dims_tmp[DimensionIndex3D::Dim3d_W] = filter_cols;
      *filter_dims = filter_dims_tmp;
    }
  }

  // Calculate Bias size for 2D or 3D Convolution. Function does not
  // return anything, but may set an error in context status.
  virtual inline void GetBiasDimension(const TensorShape& bias_shape,
                                       dnnl::memory::dims* bias_dims) {
    OP_REQUIRES(context_, bias_shape.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional: ",
                                        bias_shape.DebugString()));

    *bias_dims = {static_cast<int>(bias_shape.dim_size(0))};
  }

  // Function to calculate output and padding size for 2D/3D convolution.
  //
  // Calculate output shape of Convolution in OneDNN and TensorFlow order.
  // OneDNN uses NCHW(Conv2D) or NCDHW(Conv3D) for output order.
  // But TensorFlow output will be in NHWC||NCHW(Conv2D) or
  // NDHWC||NCDHW(Conv3D) format depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  virtual inline void GetOutputAndPadDimension(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const dnnl::memory::dims& strides, const dnnl::memory::dims& dilations,
      dnnl::memory::dims* output_dims_tf_order,
      dnnl::memory::dims* output_dims_onednn, dnnl::memory::dims* pad_left_dims,
      dnnl::memory::dims* pad_right_dims, bool pad_enabled = false) {
    OP_REQUIRES(
        context_, output_dims_tf_order != nullptr,
        errors::InvalidArgument("output_dims_tf_order shoud not be nullptr."));
    OP_REQUIRES(
        context_, output_dims_onednn != nullptr,
        errors::InvalidArgument("output_dims_onednn shoud not be nullptr."));
    OP_REQUIRES(context_, pad_left_dims != nullptr,
                errors::InvalidArgument("pad_left_dims shoud not be nullptr."));
    OP_REQUIRES(
        context_, pad_right_dims != nullptr,
        errors::InvalidArgument("pad_right_dims shoud not be nullptr."));

    bool is_conv2d = (kernel_->strides_.size() == 4);
    int input_planes, input_rows, input_cols;
    if (is_conv2d) {
      input_rows = GetTensorDim(input_shape, kernel_->data_format_, 'H');
      input_cols = GetTensorDim(input_shape, kernel_->data_format_, 'W');
    } else {
      input_planes = GetTensorDim(input_shape, kernel_->data_format_, '0');
      input_rows = GetTensorDim(input_shape, kernel_->data_format_, '1');
      input_cols = GetTensorDim(input_shape, kernel_->data_format_, '2');
    }

    // Filter dimension
    // Conv2D:
    //    First dimension: rows/height.
    //    Second dimension: cols/width.
    // Conv3D:
    //    First dimension: planes/depth.
    //    Second dimension: rows/height.
    //    Third dimension: cols/width.

    int filter_planes, filter_rows, filter_cols;
    if (is_conv2d) {
      filter_rows = filter_shape.dim_size(TF_2DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_2DFILTER_DIM_W);
    } else {
      filter_planes = filter_shape.dim_size(TF_3DFILTER_DIM_P);
      filter_rows = filter_shape.dim_size(TF_3DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_3DFILTER_DIM_W);
    }

    int stride_planes, stride_rows, stride_cols;
    int dilation_planes, dilation_rows, dilation_cols;
    if (is_conv2d) {
      // Conv2D stride is a vector of 2 elements: {s_r, s_c}
      stride_rows = strides[0];
      stride_cols = strides[1];
      dilation_rows = dilations[0];
      dilation_cols = dilations[1];
    } else {
      // Conv3D stride is a vector of 3 elements: {s_d, s_r, s_c}
      stride_planes = strides[0];
      stride_rows = strides[1];
      stride_cols = strides[2];
      dilation_planes = dilations[0];
      dilation_rows = dilations[1];
      dilation_cols = dilations[2];
    }

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, kernel_->data_format_, 'N');
    int out_depth;

    // Output depth is same as last dimension for filters for regular
    // convolutions. For depthwise it is in_depth * channel_multiplier.
    // The channel_multiplier is the last dimension of TF filter for
    // depthwise convolutions.
    if (kernel_->is_depthwise_) {
      out_depth = (filter_shape.dim_size(TF_2DFILTER_DIM_I) *
                   filter_shape.dim_size(TF_2DFILTER_DIM_O));
    } else {
      out_depth = filter_shape.dim_size(
          is_conv2d ? static_cast<int>(TF_2DFILTER_DIM_O)
                    : static_cast<int>(TF_3DFILTER_DIM_O));
    }

    int64 out_rows = 0, out_cols = 0, out_planes = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int64 pad_D1 = 0, pad_D2 = 0;
    if (is_conv2d) {
      Padding padding_type;
      if (pad_enabled) {
        padding_type = Padding::EXPLICIT;
        pad_top = static_cast<int64>((*pad_left_dims)[0]);
        pad_left = static_cast<int64>((*pad_left_dims)[1]);
        pad_bottom = static_cast<int64>((*pad_right_dims)[0]);
        pad_right = static_cast<int64>((*pad_right_dims)[1]);
      } else {
        padding_type = kernel_->padding_;
        if (padding_type == Padding::EXPLICIT) {
          GetExplicitPaddingForDim(kernel_->explicit_paddings_,
                                   kernel_->data_format_, 'H', &pad_top,
                                   &pad_bottom);
          GetExplicitPaddingForDim(kernel_->explicit_paddings_,
                                   kernel_->data_format_, 'W', &pad_left,
                                   &pad_right);
        }
      }

      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    } else {
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerboseV2(
                                   input_planes, filter_planes, dilation_planes,
                                   stride_planes, kernel_->padding_,
                                   &out_planes, &pad_D1, &pad_D2));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         kernel_->padding_, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         kernel_->padding_, &out_cols, &pad_left, &pad_right));
    }

    if (is_conv2d) {
      // Conv + pad fusion is enabled only for 2D.
      // If pad_enabled, i.e., pad and conv op are fused, then
      // all pads are already passed from pad op through
      // *pad_l and *pad_r and they don't need to be set here.
      if (!pad_enabled) {
        *pad_left_dims = {static_cast<int>(pad_top),
                          static_cast<int>(pad_left)};
        *pad_right_dims = {static_cast<int>(pad_bottom),
                           static_cast<int>(pad_right)};
      }
    } else {
      // Set padding for Conv3D here
      *pad_left_dims = {static_cast<int>(pad_D1), static_cast<int>(pad_top),
                        static_cast<int>(pad_left)};
      *pad_right_dims = {static_cast<int>(pad_D2), static_cast<int>(pad_bottom),
                         static_cast<int>(pad_right)};
    }
    // Tensorflow output is in data_format order.
    //     Conv2D: NHWC or NCHW
    //     Conv3D: NDHWC or NCDHW
    // OneDNN uses asymmetric padding.
    TensorShape out_shape =
        is_conv2d
            ? ShapeFromFormat(kernel_->data_format_, out_batch, out_rows,
                              out_cols, out_depth)
            : ShapeFromFormat(kernel_->data_format_, out_batch,
                              {{out_planes, out_rows, out_cols}}, out_depth);
    *output_dims_tf_order = TFShapeToMklDnnDims(out_shape);

    if (is_conv2d) {
      // For Conv2D, OneDNN always needs output in NCHW format.
      std::vector<DNNL_SIZE_DTYPE> output_dims_onednn_tmp(4, -1);
      output_dims_onednn_tmp[DimensionIndex::Dim_N] = out_batch;
      output_dims_onednn_tmp[DimensionIndex::Dim_C] = out_depth;
      output_dims_onednn_tmp[DimensionIndex::Dim_H] =
          static_cast<int>(out_rows);
      output_dims_onednn_tmp[DimensionIndex::Dim_W] =
          static_cast<int>(out_cols);
      *output_dims_onednn = output_dims_onednn_tmp;
    } else {
      std::vector<DNNL_SIZE_DTYPE> output_dims_onednn_tmp(5, -1);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_N] = out_batch;
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_C] = out_depth;
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_D] =
          static_cast<int>(out_planes);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_H] =
          static_cast<int>(out_rows);
      output_dims_onednn_tmp[DimensionIndex3D::Dim3d_W] =
          static_cast<int>(out_cols);
      *output_dims_onednn = output_dims_onednn_tmp;
    }
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // Conv2D/Conv3D in OneDNN order:
  //     Conv2D: NCHW for input and output; OIHW for filter.
  //     Conv3D: NCDHW for input and output; OIDHW for filter.
  // Function also calculates output shape in Tensorflow order.
  // Additionally, it also calculates strides and paddings.
  //
  // Function does not return anything, but sets error in context status.
  inline void InitFwdDimensions(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      dnnl::memory::dims* input_dims, dnnl::memory::dims* filter_dims,
      dnnl::memory::dims* strides, dnnl::memory::dims* dilations,
      dnnl::memory::dims* output_dims_tf_order,
      dnnl::memory::dims* output_dims_onednn, dnnl::memory::dims* pad_left_dims,
      dnnl::memory::dims* pad_right_dims, bool pad_enabled = false) {
    GetInputDimension(input_shape, input_dims);
    GetFilterDimension(input_shape, filter_shape, filter_dims);
    GetStrideDimension(strides);
    GetDilationDimension(dilations);
    GetOutputAndPadDimension(input_shape, filter_shape, *strides, *dilations,
                             output_dims_tf_order, output_dims_onednn,
                             pad_left_dims, pad_right_dims, pad_enabled);
  }

  void InitPadWithFusion(const Tensor& pad_tensor,
                         dnnl::memory::dims* pad_left_dims,
                         dnnl::memory::dims* pad_right_dims,
                         bool quantized_pad_enabled) {
    int64_t* paddings = nullptr;
    if (quantized_pad_enabled) {
      paddings = kernel_->explicit_paddings_.data();
    } else {
      OP_REQUIRES(context_, pad_tensor.dims() == 2,
                  errors::InvalidArgument("paddings must be 2-dimensional: ",
                                          pad_tensor.shape().DebugString()));
      // Flatten tensor to get individual paddings.
      paddings = static_cast<int64_t*>(
          const_cast<int64_t*>(pad_tensor.flat<int64_t>().data()));
    }
    // If the data format is NHWC, indices 0, 1, 6 and 7 of paddings(_tf)
    // will be zero.
    // Example:
    // paddings_tf = [ [0, 0] [1, 2] [3, 4] [0, 0] ],
    // flat method = row-major, then:
    // paddings = {0, 0, 1, 2, 3, 4, 0, 0}.
    // Hence, the values are: top = 1, bottom = 2, left = 3, right = 4.
    //
    // Similarly, if the data format is NCHW, indices 0, 1, 2 and 3 of
    // paddings(_tf) will be zero.
    // i.e. for the above example, paddings = {0, 0, 0, 0, 1, 2, 3, 4}.
    int64 pad_top = 0, pad_left = 0;
    int64 pad_bottom = 0, pad_right = 0;
    string data_format = ToString(kernel_->data_format_);
    if (data_format == "NHWC") {
      pad_top = paddings[2];
      pad_bottom = paddings[3];
      pad_left = paddings[4];
      pad_right = paddings[5];
    } else if (data_format == "NCHW") {
      pad_top = paddings[4];
      pad_bottom = paddings[5];
      pad_left = paddings[6];
      pad_right = paddings[7];
    }
    // Create padding arrays for OneDNN convolutions.
    // OneDNN uses asymmetric padding.
    // std::vector<DNNL_SIZE_DTYPE> output_dims_onednn_tmp(5, -1);
    *pad_left_dims = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
    *pad_right_dims = {static_cast<int>(pad_bottom),
                       static_cast<int>(pad_right)};

    return;
  }

  virtual void InitPostOps() {
    // Add fusions as post ops
    // NOTE: Fusion of BiasAdd is handled directly inside MklConvOp by
    // checking `fuse_biasadd_` flag.
    kernel_->post_op_params_.clear();
    if (kernel_->fuse_add_) {
      float add_scale = 1.0;
      kernel_->post_op_params_.push_back(
          {"sum", algorithm::undef, {add_scale}});
    }
    if (kernel_->fuse_activation_) {
      float activation_scale = 1.0;
      float activation_alpha = kernel_->relu_up_bound_;
      float activation_beta = 0.0;
      kernel_->post_op_params_.push_back(
          {"activation",
           kernel_->activation_alg_,
           {activation_scale, activation_alpha, activation_beta}});
    }
  }
};

Status CheckValidPadding(Padding padding_type,
                         const std::vector<int64>& explicit_paddings,
                         int num_dims, TensorFormat data_format) {
  if (padding_type == Padding::EXPLICIT) {
    if (explicit_paddings.size() != 2 * num_dims) {
      return errors::InvalidArgument(
          "explicit_paddings attribute must contain ", 2 * num_dims,
          " values, but got: ", explicit_paddings.size());
    }
    for (int64 padding_value : explicit_paddings) {
      if (padding_value < 0) {
        return errors::InvalidArgument(
            "All elements of explicit_paddings must be nonnegative");
      }
    }
    const int32 batch_index = GetTensorBatchDimIndex(num_dims, data_format);
    const int32 depth_index = GetTensorFeatureDimIndex(num_dims, data_format);
    if (explicit_paddings[2 * batch_index] != 0 ||
        explicit_paddings[2 * batch_index + 1] != 0 ||
        explicit_paddings[2 * depth_index] != 0 ||
        explicit_paddings[2 * depth_index + 1] != 0) {
      return errors::InvalidArgument(
          "Nonzero explicit padding in the batch or depth dimensions is not "
          "supported");
    }
  } else if (!explicit_paddings.empty()) {
    return errors::InvalidArgument(
        "explicit_paddings attribute must be empty if the padding attribute is "
        "not EXPLICIT");
  }
  return Status::OK();
}

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_KERNELS_GPU_CONV_OPS_H_
