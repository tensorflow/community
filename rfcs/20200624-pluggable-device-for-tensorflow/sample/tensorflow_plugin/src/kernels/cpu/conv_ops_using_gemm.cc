//#define EIGEN_USE_THREADS

#include <map>
#include <string.h>
#include <vector>

#include "gemm_functors.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

namespace demo_plugin {

struct StatusDeleter {
  void operator()(TF_Status *s) {
    if (s != nullptr) {
      TF_DeleteStatus(s);
    }
  }
};

struct TensorDeleter {
  void operator()(TF_Tensor *t) {
    if (t != nullptr) {
      TF_DeleteTensor(t);
    }
  }
};

using StatusSafePtr = std::unique_ptr<TF_Status, StatusDeleter>;
using TensorSafePtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

enum Padding {
  VALID = 1,    // No padding.
  SAME = 2,     // Input and output layers have the same size.
  EXPLICIT = 3, // Padding is explicitly specified.
};

enum TensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NCHW_VECT_C = 2,
  FORMAT_NHWC_VECT_W = 3,
  FORMAT_HWNC = 4,
  FORMAT_HWCN = 5,
};

template <typename T> struct TypeToEnum {};

template <> struct TypeToEnum<float> {
  static TF_DataType v() { return TF_DataType::TF_FLOAT; }
};

template <> struct TypeToEnum<double> {
  static TF_DataType v() { return TF_DataType::TF_DOUBLE; }
};

template <> struct TypeToEnum<Eigen::half> {
  static TF_DataType v() { return TF_DataType::TF_HALF; }
};

template <> struct TypeToEnum<Eigen::bfloat16> {
  static TF_DataType v() { return TF_DataType::TF_BFLOAT16; }
};

static bool GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                                  int64_t dilation_rate, int64_t stride,
                                  Padding padding_type, int64_t *output_size,
                                  int64_t *padding_before) {
  if (stride <= 0) {
    std::cerr << "Stride must be > 0, but got " << stride << std::endl;
    return false;
  }
  if (dilation_rate < 1) {
    std::cerr << "Dilation rate must be >= 1, but got " << dilation_rate
              << std::endl;
    return false;
  }

  int64_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
  case Padding::VALID:
    *output_size = (input_size - effective_filter_size + stride) / stride;
    *padding_before = 0;
    break;
  case Padding::SAME:
    *output_size = (input_size + stride - 1) / stride;
    const int64_t padding_needed =
        std::max(int64_t{0}, (*output_size - 1) * stride +
                                 effective_filter_size - input_size);
    // For odd values of total padding, add more padding at the 'right'
    // side of the given dimension.
    *padding_before = padding_needed / 2;
    break;
  }
  if (*output_size < 0) {
    std::cerr << "Computed output size would be negative: " << *output_size
              << " [input_size: " << input_size
              << ", effective_filter_size: " << effective_filter_size
              << ", stride: " << stride << "]" << std::endl;
    return false;
  }
  return true;
}

static int64_t GetTensorDim(TF_Tensor *tensor, std::string &format, char dim) {
  int idx = -1;
  if (format == "NCHW") {
    switch (dim) {
    case 'N': {
      idx = 0;
      break;
    }
    case 'C': {
      idx = 1;
      break;
    }
    case 'H': {
      idx = 2;
      break;
    }
    case 'W': {
      idx = 3;
      break;
    }
    default: {
      idx = -1;
    }
    }
  } else if (format == "NHWC") {
    switch (dim) {
    case 'N': {
      idx = 0;
      break;
    }
    case 'C': {
      idx = 3;
      break;
    }
    case 'H': {
      idx = 1;
      break;
    }
    case 'W': {
      idx = 2;
      break;
    }
    default: {
      idx = -1;
    }
    }
  } else {
    std::cerr << "Unsupport data_format now" << std::endl;
    return -1;
  }
  return TF_Dim(tensor, idx);
}

#define CHECK_CONSTRUCT_STATUS(ctx, status)                                    \
  do {                                                                         \
    if (TF_GetCode(status) != TF_OK) {                                         \
      TF_OpKernelConstruction_Failure(ctx, status);                            \
    }                                                                          \
  } while (0);

#define CHECK_CTX_STATUS(ctx, status)                                          \
  do {                                                                         \
    if (TF_GetCode(status) != TF_OK) {                                         \
      TF_OpKernelContext_Failure(ctx, status);                                 \
    }                                                                          \
  } while (0);

namespace {
const size_t kMaxChunkSize = (16 * 1024 * 1024);

// Implements convolution as a two stage process, first packing the patches of
// the input image into columns (im2col) and then running GEMM to produce the
// final result.
template <class T1, class T2, class T3, class TGemmFunctor>
class Im2ColConvFunctor {
public:
  void operator()(const T1 *input_data, int input_batches, int input_height,
                  int input_width, int input_depth, const T2 *filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride_rows, int stride_cols, Padding padding,
                  T3 *output_data, int output_height, int output_width) {
    if ((input_batches <= 0) || (input_width <= 0) || (input_height <= 0) ||
        (input_depth <= 0)) {
      std::cerr << "Conv2D was called with bad input dimensions: "
                << input_batches << ", " << input_height << ", " << input_width
                << ", " << input_depth;
      return;
    }
    if ((filter_width <= 0) || (filter_height <= 0) || (filter_count <= 0)) {
      std::cerr << "Conv2D was called with bad filter dimensions: "
                << filter_width << ", " << filter_height << ", "
                << filter_count;
      return;
    }
    if ((output_width <= 0) || (output_height <= 0)) {
      std::cerr << "Conv2D was called with bad output width or height: "
                << output_width << ", " << output_height;
      return;
    }
    // We can just use a GEMM if the im2col is the identity operator, e.g., if
    // // the kernel is 1x1 or the input data and filter have same height/width.
    if (filter_height == 1 && filter_width == 1 && stride_rows == 1 &&
        stride_cols == 1) {
      // The kernel is 1x1.
      const int m = input_batches * input_height * input_width;
      const int n = filter_count;
      const int k = input_depth;
      const int lda = k;
      const int ldb = filter_count;
      const int ldc = filter_count;
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, input_data, lda, filter_data, ldb, output_data,
                   ldc);
      return;
    } else if (filter_height == input_height && filter_width == input_width &&
               padding == VALID) {
      // The input data and filter have the same height/width.
      const int m = input_batches;
      const int n = filter_count;
      const int k = input_height * input_width * input_depth;
      const int lda = k;
      const int ldb = filter_count;
      const int ldc = filter_count;
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, input_data, lda, filter_data, ldb, output_data,
                   ldc);
      return;
    }

    // These calculations define how the patches will be positioned within the
    // input image. The actual definitions are quite complex, and rely on the
    // previously-calculated output size.
    int filter_left_offset;
    int filter_top_offset;
    if (padding == VALID) {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width + 1) /
          2;
      filter_top_offset = ((output_height - 1) * stride_rows + filter_height -
                           input_height + 1) /
                          2;
    } else {
      filter_left_offset =
          ((output_width - 1) * stride_cols + filter_width - input_width) / 2;
      filter_top_offset =
          ((output_height - 1) * stride_rows + filter_height - input_height) /
          2;
    }

    // The im2col buffer has # of patches rows, and # of filters cols.
    // It's laid out like this, in row major order in memory:
    //        < filter value count >
    //   ^   +---------------------+
    // patch |                     |
    // count |                     |
    //   v   +---------------------+
    // Each patch row contains a filter_width x filter_height patch of the
    // input, with the depth channel as the most contiguous in memory, followed
    // by the width, then the height. This is the standard memory order in the
    // image world if it helps to visualize it.
    const int filter_value_count = filter_width * filter_height * input_depth;
    if ((filter_value_count * sizeof(T1)) > kMaxChunkSize) {
      std::cerr << "Im2Col patch too large for buffer" << std::endl;
      return;
    }
    const int64_t patches_per_chunk =
        kMaxChunkSize / (filter_value_count * sizeof(T1));
    const int64_t chunk_value_count =
        (kMaxChunkSize + (sizeof(T1) - 1)) / sizeof(T1);
    // This means that multiple ops can't be run simultaneously on different
    // threads, because we have a single shared resource. The platforms this is
    // aimed at have intra-op parallelism as their focus though, so it shouldn't
    // be an issue.
    // T1* im2col_buffer = new T1[chunk_value_count];
    std::unique_ptr<T1> im2col_buffer(new T1[chunk_value_count]);

    const int64_t patch_count = (input_batches * output_height * output_width);
    const int64_t chunk_count =
        (patch_count + (patches_per_chunk - 1)) / patches_per_chunk;
    for (int64_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      const int64_t patch_index_start = chunk_index * patches_per_chunk;
      const int64_t patch_index_end =
          std::min(patch_index_start + patches_per_chunk, patch_count);
      for (int64_t patch_index = patch_index_start;
           patch_index < patch_index_end; ++patch_index) {
        const int64_t batch = patch_index / (output_height * output_width);
        const int64_t out_y = (patch_index / output_width) % output_height;
        const int64_t out_x = patch_index % output_width;
        const T1 *input_batch_start =
            input_data + (batch * input_height * input_width * input_depth);
        const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
        const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
        const int patch_index_within_chunk = patch_index % patches_per_chunk;
        T1 *im2col_patch_start =
            im2col_buffer.get() +
            (patch_index_within_chunk * filter_value_count);
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int in_y = in_y_origin + filter_y;
          T1 *im2col_row_start =
              im2col_patch_start + (filter_y * filter_width * input_depth);
          // If we're off the top or the bottom of the input, fill the
          // whole row with zeroes.
          if ((in_y < 0) || (in_y >= input_height)) {
            T1 *im2col_row_end =
                im2col_row_start + (filter_width * input_depth);
            std::fill(im2col_row_start, im2col_row_end, T1(0));
          } else {
            // What we're doing here is trying to copy and fill the im2col
            // buffer as efficiently as possible, using functions to set or
            // duplicate values en masse. We know we don't have to worry about
            // vertical edges because we dealt with that case above, so we
            // just need to handle filters that overlap the left or right
            // edges. Here's what that looks like:
            //
            // < left_zero_count > < center_copy_count > < right_zero_count >
            // +------------------+---------------------+--------------------+
            // |     (filter)     |       (image)       |      (filter)      |
            // +------------------+---------------------+--------------------+
            // in_x_origin        0                 input_width       in_x_end
            //
            // In reality it's unlikely that a filter patch will be wider
            // than an input, but this shows all the edge cases.
            // We use std::fill() to set the left and right sections to zeroes
            // and std::copy() to copy over the input data for the center.
            const int in_x_end = in_x_origin + filter_width;
            const int left_zero_count = std::max(0, 0 - in_x_origin);
            const int right_zero_count = std::max(0, in_x_end - input_width);
            const int center_copy_count =
                filter_width - (left_zero_count + right_zero_count);
            if (left_zero_count > 0) {
              T1 *im2col_left_start = im2col_row_start;
              T1 *im2col_left_end =
                  im2col_left_start + (left_zero_count * input_depth);
              std::fill(im2col_left_start, im2col_left_end, T1(0));
            }
            if (center_copy_count > 0) {
              const T1 *input_row_start =
                  input_batch_start + (in_y * input_width * input_depth) +
                  (std::max(0, in_x_origin) * input_depth);
              const T1 *input_row_end =
                  input_row_start + (center_copy_count * input_depth);
              T1 *im2col_center_start =
                  im2col_row_start + (left_zero_count * input_depth);
              std::copy(input_row_start, input_row_end, im2col_center_start);
            }
            if (right_zero_count > 0) {
              T1 *im2col_right_start =
                  im2col_row_start +
                  ((left_zero_count + center_copy_count) * input_depth);
              T1 *im2col_right_end =
                  im2col_right_start + (right_zero_count * input_depth);
              std::fill(im2col_right_start, im2col_right_end, T1(0));
            }
          }
        }
      }
      // Now we've assembled a set of image patches into a matrix, apply a
      // GEMM matrix multiply of the patches as rows, times the filter
      // weights in columns, to get partial results in the output matrix.
      const int how_many_patches = patch_index_end - patch_index_start;
      const int m = how_many_patches;
      const int n = filter_count;
      const int k = filter_value_count;
      const int lda = filter_value_count;
      const int ldb = filter_count;
      const int ldc = filter_count;
      T3 *chunk_output_data = output_data + (patch_index_start * filter_count);
      TGemmFunctor gemm_functor;
      gemm_functor(m, n, k, im2col_buffer.get(), lda, filter_data, ldb,
                   chunk_output_data, ldc);
    }
  }
};

} // namespace

template <class T> struct Conv2DUsingGemmOp {
  Conv2DUsingGemmOp() : data_format_("") {}
  std::vector<int32_t> strides_;
  Padding padding_;
  std::string data_format_;
};

template <class T>
void *Conv2DUsingGemmOp_Create(TF_OpKernelConstruction *ctx) {
  auto kernel = new Conv2DUsingGemmOp<T>();

  StatusSafePtr status(TF_NewStatus());
  int32_t list_size = 0;
  int32_t total_size = 0;

  // Get strides
  TF_OpKernelConstruction_GetAttrSize(ctx, "strides", &list_size, &total_size,
                                      status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  kernel->strides_.resize(list_size);
  TF_OpKernelConstruction_GetAttrInt32List(
      ctx, "strides", kernel->strides_.data(), list_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());

  // Get data_format
  TF_OpKernelConstruction_GetAttrSize(ctx, "data_format", &list_size,
                                      &total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::vector<char> format_vec(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", format_vec.data(),
                                        total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  kernel->data_format_ = std::move(std::string(format_vec.data(), total_size));

  // Get padding
  TF_OpKernelConstruction_GetAttrSize(ctx, "padding", &list_size, &total_size,
                                      status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::vector<char> padding_vec(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding_vec.data(),
                                        total_size, status.get());
  CHECK_CONSTRUCT_STATUS(ctx, status.get());
  std::string padding_str(padding_vec.data(), total_size);
  if (padding_str == "VALID") {
    kernel->padding_ = Padding::VALID;
  } else if (padding_str == "SAME") {
    kernel->padding_ = Padding::SAME;
  } else {
    std::cerr << "Unsupported padding type: " << padding_str;
    return nullptr;
  }
  return kernel;
}

template <typename T> void Conv2DUsingGemmOp_Delete(void *kernel) {
  if (kernel != nullptr) {
    delete static_cast<Conv2DUsingGemmOp<T> *>(kernel);
  }
}

template <class T, class TConvFunctor>
void Conv2DUsingGemmOp_Compute(void *kernel, TF_OpKernelContext *ctx) {
  StatusSafePtr status(TF_NewStatus());
  // Input tensor is of the following dimensions:
  // [ batch, in_rows, in_cols, in_depth ]
  TF_Tensor *input = nullptr;
  TF_GetInput(ctx, 0, &input, status.get());
  CHECK_CTX_STATUS(ctx, status.get());
  TensorSafePtr input_safe_ptr(input);

  // Input filter is of the following dimensions:
  // [ filter_rows, filter_cols, in_depth, out_depth]
  TF_Tensor *filter = nullptr;
  TF_GetInput(ctx, 1, &filter, status.get());
  CHECK_CTX_STATUS(ctx, status.get());
  TensorSafePtr filter_safe_ptr(filter);

  if (TF_NumDims(input) != 4) {
    std::cerr << "input must be 4 dimensional" << std::endl;
    return;
  }
  if (TF_NumDims(filter) != 4) {
    std::cerr << "filter must be 4 dimensional" << std::endl;
    return;
  }

  for (int i = 0; i < 3; i++) {
    if (TF_Dim(filter, i) >= std::numeric_limits<int>::max()) {
      std::cerr << "filter too large" << std::endl;
      return;
    }
  }

  // The last dimension for input is in_depth. It must be the same as the
  // filter's in_depth.
  const int64_t in_depth = GetTensorDim(
      input, static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_, 'C');
  if (in_depth != TF_Dim(filter, 2)) {
    std::cerr << "input and filter must have the same depth" << std::endl;
    return;
  }

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(TF_Dim(filter, 3));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64_t input_rows_raw = GetTensorDim(
      input, static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_, 'H');
  if (input_rows_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "Input rows too large";
    return;
  }
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(TF_Dim(filter, 0));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64_t input_cols_raw = GetTensorDim(
      input, static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_, 'W');
  if (input_cols_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "Input cols too large" << std::endl;
    return;
  }
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(TF_Dim(filter, 1));

  // The first dimension for input is batch.
  const int64_t batch_raw = GetTensorDim(
      input, static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_, 'N');
  if (batch_raw >= std::numeric_limits<int>::max()) {
    std::cerr << "batch is too large" << std::endl;
    return;
  }
  const int batch = static_cast<int>(batch_raw);

  // For now we take the stride from the second and third dimensions only (we
  // do not support striding on the batch or depth dimension).
  int stride_rows = 0;
  int stride_cols = 0;
  if (static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_ == "NCHW") {
    stride_rows = static_cast<Conv2DUsingGemmOp<T> *>(kernel)->strides_[2];
    stride_cols = static_cast<Conv2DUsingGemmOp<T> *>(kernel)->strides_[3];
  } else if (static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_ ==
             "NHWC") {
    stride_rows = static_cast<Conv2DUsingGemmOp<T> *>(kernel)->strides_[1];
    stride_cols = static_cast<Conv2DUsingGemmOp<T> *>(kernel)->strides_[2];
  } else {
    std::cerr << "Unsupported data format" << std::endl;
    return;
  }

  int64_t out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  if (!GetWindowedOutputSize(
          input_rows, filter_rows, 1, stride_rows,
          static_cast<Conv2DUsingGemmOp<T> *>(kernel)->padding_, &out_rows,
          &pad_rows)) {
    std::cerr << "Invalid filter size" << std::endl;
    return;
  }

  if (!GetWindowedOutputSize(
          input_cols, filter_cols, 1, stride_cols,
          static_cast<Conv2DUsingGemmOp<T> *>(kernel)->padding_, &out_cols,
          &pad_cols)) {
    std::cerr << "Invalid filter size" << std::endl;
    return;
  }
  auto output_size = batch * out_rows * out_cols * out_depth;
  std::vector<int64_t> out_shape;
  out_shape.push_back(batch);
  if (static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_ == "NCHW") {
    out_shape.push_back(out_depth);
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
  } else if (static_cast<Conv2DUsingGemmOp<T> *>(kernel)->data_format_ ==
             "NHWC") {
    out_shape.push_back(out_rows);
    out_shape.push_back(out_cols);
    out_shape.push_back(out_depth);
  } else {
    std::cerr << "Unsupported data_foramt" << std::endl;
    return;
  }

  // Output tensor is of the following dimensions:
  // [ in_batch, out_rows, out_cols, out_depth ]``
  TensorSafePtr output_safe_ptr(TF_AllocateOutput(
      ctx, 0, TF_ExpectedOutputDataType(ctx, 0), out_shape.data(),
      out_shape.size(), sizeof(T) * output_size, status.get()));

  // If there is nothing to compute, return.
  if (output_size == 0) {
    return;
  }
  TConvFunctor conv_functor;
  conv_functor(static_cast<T *>(TF_TensorData(input_safe_ptr.get())), batch,
               input_rows, input_cols, in_depth,
               static_cast<T *>(TF_TensorData(filter_safe_ptr.get())),
               filter_rows, filter_cols, out_depth, stride_rows, stride_cols,
               static_cast<Conv2DUsingGemmOp<T> *>(kernel)->padding_,
               static_cast<T *>(TF_TensorData(output_safe_ptr.get())), out_rows,
               out_cols);
};
template <typename T> void RegisterConvOpKernel(const char *device_type) {
  StatusSafePtr status(TF_NewStatus());
  auto *builder = TF_NewKernelBuilder(
      "Conv2D", device_type, &Conv2DUsingGemmOp_Create<T>,
      &Conv2DUsingGemmOp_Compute<
          T, Im2ColConvFunctor<T, T, T, FastGemmFunctor<T, T, T>>>,
      &Conv2DUsingGemmOp_Delete<T>);
  TF_KernelBuilder_TypeConstraint(builder, "T", TypeToEnum<T>::v(),
                                  status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel with attribute T";
  TF_RegisterKernelBuilder("Conv2DOp", builder, status.get());
  if (TF_OK != TF_GetCode(status.get()))
    std::cout << " Error while registering relu kernel";
}

} // namespace demo_plugin

void RegisterDeviceConv2D(const char *device_type) {
  demo_plugin::RegisterConvOpKernel<float>(device_type);
}
