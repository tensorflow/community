#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_DNNL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_DNNL_UTIL_H_

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "tensorflow_plugin/src/xpu_core/util/logging.h"
#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"
#include "tensorflow_plugin/src/xpu_core/util/op_requires.h"
#include "tensorflow_plugin/src/xpu_core/util/status.h"
#include "tensorflow_plugin/src/xpu_core/util/strcat.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_format.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

#ifndef ENABLE_DNNL_V2
#define ENABLE_DNNL_V2
#endif

using MemoryArgsMap = std::unordered_map<int, dnnl::memory>;

namespace intel_plugin {

// The dimensions order that DNNL internally uses for 2D activations
// [Batch, Channel, Height, Width] and
// for 2D filters [Out_Channel, In_Channel, Height, Width].
typedef enum {
  Dim_N = 0,
  Dim_C = 1,
  Dim_H = 2,
  Dim_W = 3,
  Dim_O = 0,
  Dim_I = 1
} DimensionIndex;

// The dimensions order that DNNL internally uses for 3D activations
// [Batch, Channel, Depth, Height, Width] and
// for 3D filters [Out_Channel, In_Channel, Depth, Height, Width].
typedef enum {
  Dim3d_N = 0,
  Dim3d_C = 1,
  Dim3d_D = 2,
  Dim3d_H = 3,
  Dim3d_W = 4,
  Dim3d_O = 0,
  Dim3d_I = 1
} DimensionIndex3D;

// In MKL-DNN v1.x, the format (ex. NCHW) used to initialize a memory descriptor
// (md) structure will no longer be recorded in its `format` field. Instead, it
// will be set to a canonical `blocked` format for every fully described md.
//
// Currently, we query this `format` field while mapping MKL-DNN's data format
// to TF's data format. Due to the above restriction, we will now get this data
// format information from TF's `data_format` attribute (i.e. via
// `TensorFormat`) for MKL-DNN v1.x.
//
// Some MKL-DNN operators such as ReLU do not have a `data_format` attribute
// since they are usually in `blocked` format. Therefore, in order to
// distinguish between blocked and non-blocked formats, we have defined a new
// enum called `MklTensorFormat` that is semantically similar to `TensorFormat`
// but with the following additional fields namely:
//  1) FORMAT_BLOCKED: as described above, this is needed for element-wise
//     operators such as ReLU.
//  2) FORMAT_INVALID: for error-checking (ex. unsupported format)
//  3) FORMAT_X, FORMAT_NC, FORMAT_TNC: to distinguish between MKL tensors based
//     on their dimensions in operators such as Softmax, i.e.:
//        FORMAT_X   - 1D tensor
//        FORMAT_NC  - 2D tensor
//        FORMAT_TNC - 3D tensor
enum class MklTensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NDHWC = 2,
  FORMAT_NCDHW = 3,
  FORMAT_X = 4,
  FORMAT_NC = 5,
  FORMAT_TNC = 6,
  FORMAT_BLOCKED = 7,
  FORMAT_INVALID = 8,
};

// Enum for the order of dimensions of a TF 2D filter with shape [filter_height,
// filter_width, in_channels, out_channels]
typedef enum {
  TF_2DFILTER_DIM_H = 0,
  TF_2DFILTER_DIM_W = 1,
  TF_2DFILTER_DIM_I = 2,
  TF_2DFILTER_DIM_O = 3
} TFFilterDims2d;

// Enum for the order of dimensions of a TF 3D filter with shape [filter_depth,
// filter_height, filter_width, in_channels, out_channels]
typedef enum {
  TF_3DFILTER_DIM_P = 0,
  TF_3DFILTER_DIM_H = 1,
  TF_3DFILTER_DIM_W = 2,
  TF_3DFILTER_DIM_I = 3,
  TF_3DFILTER_DIM_O = 4
} TFFilterDims3d;

// The dimensions order that MKL-DNN requires for the filter in a grouped
// convolution (2D only)
typedef enum {
  GROUP_FILTER_DIM_G = 0,
  GROUP_FILTER_DIM_O = 1,
  GROUP_FILTER_DIM_I = 2,
  GROUP_FILTER_DIM_H = 3,
  GROUP_FILTER_DIM_W = 4
} FilterGroupDims;

// Forward declarations
dnnl::memory::format_tag MklTensorFormatToMklDnnDataFormat(
    MklTensorFormat format);

TensorFormat MklDnn3DDataFormatToTFDataFormat(MklTensorFormat format);
TensorFormat MklDnnDataFormatToTFDataFormat(MklTensorFormat format);

dnnl::memory::dims CalculateTFStrides(const dnnl::memory::dims& dims_tf_order);
Status CreateBlockedMemDescHelper(const dnnl::memory::dims& dim,
                                  const dnnl::memory::dims& strides,
                                  dnnl::memory::data_type dtype,
                                  dnnl_memory_desc_t* blocked_md);

/// Return MKL-DNN data type (memory::data_type) for input type T
///
/// @input None
/// @return dnnl::memory::data_type corresponding to type T
template <typename T>
static dnnl::memory::data_type MklDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
dnnl::memory::data_type MklDnnType<float>() {
  return dnnl::memory::data_type::f32;
}

template <>
dnnl::memory::data_type MklDnnType<Eigen::half>() {
  return dnnl::memory::data_type::f16;
}

template <>
dnnl::memory::data_type MklDnnType<quint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
dnnl::memory::data_type MklDnnType<uint8>() {
  return dnnl::memory::data_type::u8;
}

template <>
dnnl::memory::data_type MklDnnType<qint8>() {
  return dnnl::memory::data_type::s8;
}

template <>
dnnl::memory::data_type MklDnnType<qint32>() {
  return dnnl::memory::data_type::s32;
}

template <>
dnnl::memory::data_type MklDnnType<Eigen::bfloat16>() {
  return dnnl::memory::data_type::bf16;
}

inline dnnl::engine CreateDnnlEngine(OpKernelContext& ctx) {
  auto* dpcpp_stream = ctx.GetDeviceStream();
#ifdef ENABLE_DNNL_V2
  return dnnl::sycl_interop::make_engine(dpcpp_stream->get_device(),
                                         dpcpp_stream->get_context());
#else
  return dnnl::engine(dnnl::engine::kind::gpu, dpcpp_stream->get_device(),
                      dpcpp_stream->get_context());
#endif
}

inline dnnl::stream CreateDnnlStream(OpKernelContext& ctx,
                                     const dnnl::engine& engine) {
  auto* dpcpp_stream = ctx.GetDeviceStream();
#ifdef ENABLE_DNNL_V2
  return dnnl::sycl_interop::make_stream(engine, *dpcpp_stream);
#else
  return dnnl::stream(engine, *dpcpp_stream);
#endif
}

inline dnnl::memory CreateDnnlMemory(dnnl::memory::desc md,
                                     const dnnl::engine& engine,
                                     void* data_handle) {
#ifdef ENABLE_DNNL_V2
  dnnl::sycl_interop::memory_kind kind = dnnl::sycl_interop::memory_kind::usm;
  return dnnl::sycl_interop::make_memory(md, engine, kind, data_handle);
#else
  return dnnl::memory(md, engine, data_handle);
#endif
}

// TODO this one may have memory leak.
inline dnnl::memory* GetMkldnnMemory(const dnnl::memory::desc& md,
                                     const dnnl::engine& engine,
                                     void* data_handle) {
  dnnl::memory* mem;
  try {
    mem = new dnnl::memory(md, engine, data_handle);
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + string(e.message) + ", in file " +
                       string(__FILE__) + ":" + std::to_string(__LINE__);
  }
  return mem;
}

class MklDnnShape {
 private:
  typedef struct {
    // Flag to indicate if the tensor is an MKL tensor or not
    bool is_mkl_tensor_ = false;
    // Number of dimensions in Tensorflow format
    size_t dimension_ = 0;
    dnnl_dims_t sizes_;  // Required by MKL for conversions
    MklTensorFormat tf_data_format_ = MklTensorFormat::FORMAT_BLOCKED;
    dnnl::memory::data_type T_ = dnnl::memory::data_type::undef;
    // MKL layout
    dnnl_memory_desc_t mkl_md_;
    /// TF dimension corresponding to this MKL dimension
    dnnl_dims_t map_;
  } MklShapeData;
  MklShapeData data_;

  typedef std::remove_extent<dnnl_dims_t>::type dnnl_dim_t;

 public:
  MklDnnShape() {
    for (size_t i = 0; i < sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
         ++i) {
      data_.sizes_[i] = -1;
    }
    for (size_t i = 0; i < sizeof(data_.map_) / sizeof(data_.map_[0]); ++i) {
      data_.map_[i] = -1;
    }
  }

  ~MklDnnShape() {}
  TF_DISALLOW_COPY_AND_ASSIGN(MklDnnShape);  // Cannot copy

  /// Equality function for MklDnnShape objects
  /// @return true if both are equal; false otherwise.
  inline bool operator==(const MklDnnShape& input_shape) const {
    if (this->IsMklTensor() != input_shape.IsMklTensor()) {
      return false;
    }

    // If input tensors are in MKL layout, then we check for dimensions and
    // sizes.
    if (this->IsMklTensor()) {
      const dnnl_memory_desc_t& cur_md = (this->GetMklLayout()).data;
      const dnnl_memory_desc_t& input_shape_md =
          input_shape.GetMklLayout().data;
      return this->GetTfShape() == input_shape.GetTfShape() &&
             dnnl_memory_desc_equal(&cur_md, &input_shape_md);
    }

    // Both inputs are not MKL tensors.
    return true;
  }

  /// Equality operator for MklDnnShape and TFShape.
  /// Returns: true if TF shapes for both are the same, false otherwise
  inline bool operator==(const TensorShape& input_shape) const {
    if (!this->IsMklTensor()) {
      return false;
    }

    return this->GetTfShape() == input_shape;
  }

  inline const bool IsMklTensor() const { return data_.is_mkl_tensor_; }
  inline void SetMklTensor(bool is_mkl_tensor) {
    data_.is_mkl_tensor_ = is_mkl_tensor;
  }

  inline void SetDimensions(const size_t dimension) {
    data_.dimension_ = dimension;
  }
  inline size_t GetDimension(char dimension) const {
    int index = GetMklDnnTensorDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }

  inline size_t GetDimension3D(char dimension) const {
    int index = GetMklDnnTensor3DDimIndex(dimension);
    CHECK(index >= 0 && index < this->GetDimension())
        << "Invalid index from the dimension: " << index << ", " << dimension;
    return this->DimSize(index);
  }

  inline int32 GetMklDnnTensorDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return DimensionIndex::Dim_N;
      case 'C':
        return DimensionIndex::Dim_C;
      case 'H':
        return DimensionIndex::Dim_H;
      case 'W':
        return DimensionIndex::Dim_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  }

  inline int32 GetMklDnnTensor3DDimIndex(char dimension) const {
    switch (dimension) {
      case 'N':
        return DimensionIndex3D::Dim3d_N;
      case 'C':
        return DimensionIndex3D::Dim3d_C;
      case 'D':
        return DimensionIndex3D::Dim3d_D;
      case 'H':
        return DimensionIndex3D::Dim3d_H;
      case 'W':
        return DimensionIndex3D::Dim3d_W;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  }

  inline size_t GetDimension() const { return data_.dimension_; }
  inline const int* GetSizes() const {
    return reinterpret_cast<const int*>(&data_.sizes_[0]);
  }

  // Returns an dnnl::memory::dims object that contains the sizes of this
  // MklDnnShape object.
  inline dnnl::memory::dims GetSizesAsMklDnnDims() const {
    dnnl::memory::dims retVal;
    if (data_.is_mkl_tensor_) {
      size_t dimensions = sizeof(data_.sizes_) / sizeof(data_.sizes_[0]);
      for (size_t i = 0; i < dimensions; i++) {
        if (data_.sizes_[i] != -1) retVal.push_back(data_.sizes_[i]);
      }
    } else {
      CHECK_EQ(data_.is_mkl_tensor_, true);
    }
    return retVal;
  }

  inline int64 DimSize(int index) const {
    CHECK_LT(index, sizeof(data_.sizes_) / sizeof(data_.sizes_[0]));
    return data_.sizes_[index];
  }

  /// Return TensorShape that describes the Tensorflow shape of the tensor
  /// represented by this MklShape.
  inline TensorShape GetTfShape() const {
    CHECK_EQ(data_.is_mkl_tensor_, true);

    std::vector<int32> shape(data_.dimension_, -1);
    // As mentioned in the comment above, we now rely on TF's `data_format`
    // attribute to determine if TF shape is in blocked format or not.
    if (data_.tf_data_format_ != MklTensorFormat::FORMAT_BLOCKED) {
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[TfDimIdx(idx)];
      }
    } else {
      // If Tensorflow shape is in Blocked format, then we don't have dimension
      // map for it. So we just create Tensorflow shape from sizes in the
      // specified order.
      for (size_t idx = 0; idx < data_.dimension_; ++idx) {
        shape[idx] = data_.sizes_[idx];
      }
    }

    TensorShape ts;
    bool ret = TensorShapeUtils::MakeShape(shape, &ts).ok();
    CHECK_EQ(ret, true);
    return ts;
  }

  inline void SetElemType(dnnl::memory::data_type dt) { data_.T_ = dt; }
  inline const dnnl::memory::data_type GetElemType() { return data_.T_; }

  inline void SetMklLayout(dnnl::memory::desc* md) {
    CHECK_NOTNULL(md);
    data_.mkl_md_ = md->data;
  }

  inline const dnnl::memory::desc GetMklLayout() const {
    return dnnl::memory::desc(data_.mkl_md_);
  }

  inline MklTensorFormat GetTfDataFormat() const {
    return data_.tf_data_format_;
  }

  /// We don't create primitive_descriptor for TensorFlow layout now.
  /// We use lazy evaluation and create it only when needed. Input format can
  /// also be Blocked format.
  inline void SetTfLayout(size_t dims, const dnnl::memory::dims& sizes,
                          MklTensorFormat format) {
    DCHECK_EQ(dims, sizes.size())
        << "SetTfLayout: Number of dimensions does not"
           "match with dimension array";
    data_.dimension_ = dims;
    for (size_t ii = 0; ii < dims; ++ii) {
      data_.sizes_[ii] = sizes[ii];
    }
    data_.tf_data_format_ = format;
    if (format != MklTensorFormat::FORMAT_BLOCKED) {
      if (dims == 2) {
        data_.map_[0] = DimensionIndex::Dim_N;
        data_.map_[1] = DimensionIndex::Dim_C;
      } else {
        SetTfDimOrder(dims, format);
      }
    }
  }

  inline const dnnl::memory::desc GetTfLayout() const {
    dnnl::memory::dims dims;
    for (size_t ii = 0; ii < data_.dimension_; ++ii) {
      dims.push_back(data_.sizes_[ii]);
    }

    // Create Blocked memory desc if input TF format was set like that.
    if (data_.tf_data_format_ == MklTensorFormat::FORMAT_BLOCKED) {
      auto strides = CalculateTFStrides(dims);
      dnnl_memory_desc_t blocked_md;
      TF_CHECK_OK(
          CreateBlockedMemDescHelper(dims, strides, data_.T_, &blocked_md));
      return dnnl::memory::desc(blocked_md);
    } else {
      auto format_tag =
          MklTensorFormatToMklDnnDataFormat(data_.tf_data_format_);
      DCHECK_NE(format_tag, dnnl::memory::format_tag::undef);
      return dnnl::memory::desc(dims, data_.T_, format_tag);
    }
  }

  inline const dnnl::memory::desc GetCurLayout() const {
    return IsMklTensor() ? GetMklLayout() : GetTfLayout();
  }

  // We don't need a case of default dimension order because
  // when an operator that does not get data_format attribute gets all inputs
  // in Tensorflow format, it will produce output in Tensorflow format.
  inline void SetTfDimOrder(const size_t dimension, const dnnl_dims_t map) {
    CHECK(dimension == data_.dimension_);
    for (size_t ii = 0; ii < dimension; ii++) {
      data_.map_[ii] = map[ii];
    }
  }

  inline void SetTfDimOrder(const size_t dimension, TensorFormat data_format) {
    if (dimension == 5) {
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<3>(data_format, '0')] =
          DimensionIndex3D::Dim3d_D;
      data_.map_[GetTensorDimIndex<3>(data_format, '1')] =
          DimensionIndex3D::Dim3d_H;
      data_.map_[GetTensorDimIndex<3>(data_format, '2')] =
          DimensionIndex3D::Dim3d_W;
      data_.map_[GetTensorDimIndex<3>(data_format, 'C')] =
          DimensionIndex3D::Dim3d_C;
      data_.map_[GetTensorDimIndex<3>(data_format, 'N')] =
          DimensionIndex3D::Dim3d_N;
    } else {
      CHECK_EQ(dimension, 4);
      CHECK(dimension == data_.dimension_);
      data_.map_[GetTensorDimIndex<2>(data_format, 'W')] =
          DimensionIndex::Dim_W;
      data_.map_[GetTensorDimIndex<2>(data_format, 'H')] =
          DimensionIndex::Dim_H;
      data_.map_[GetTensorDimIndex<2>(data_format, 'C')] =
          DimensionIndex::Dim_C;
      data_.map_[GetTensorDimIndex<2>(data_format, 'N')] =
          DimensionIndex::Dim_N;
    }
  }

  inline void SetTfDimOrder(const size_t dimension, MklTensorFormat format) {
    TensorFormat data_format = MklDnnDataFormatToTFDataFormat(format);
    SetTfDimOrder(dimension, data_format);
  }

  inline const dnnl_dim_t* GetTfToMklDimMap() const { return &data_.map_[0]; }
  inline size_t TfDimIdx(int index) const { return data_.map_[index]; }
  inline int64 TfDimSize(int index) const {
    return data_.sizes_[TfDimIdx(index)];
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Channel dimension.
  inline bool IsMklChannelDim(int d) const {
    return TfDimIdx(d) == DimensionIndex::Dim_C;
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Batch dimension.
  inline bool IsMklBatchDim(int d) const {
    return TfDimIdx(d) == DimensionIndex::Dim_N;
  }

  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Width dimension.
  inline bool IsMklWidthDim(int d) const {
    return TfDimIdx(d) == DimensionIndex::Dim_W;
  }
  /// Query TF-MKL dimension ordering map and check if Tensorflow dimension 'd'
  /// corresponds to MKL's Height dimension.
  inline bool IsMklHeightDim(int d) const {
    return TfDimIdx(d) == DimensionIndex::Dim_H;
  }

  /// Check if the TF-MKL dimension ordering map specifies if the input
  /// tensor is in NCHW format.
  inline bool IsTensorInNCHWFormat() const {
    TensorFormat data_format = FORMAT_NCHW;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  /// Check if the TF-MKL dimension ordering map specifies if the input
  /// tensor is in NHWC format.
  inline bool IsTensorInNHWCFormat() const {
    TensorFormat data_format = FORMAT_NHWC;
    return (IsMklBatchDim(GetTensorDimIndex<2>(data_format, 'N')) &&
            IsMklChannelDim(GetTensorDimIndex<2>(data_format, 'C')) &&
            IsMklHeightDim(GetTensorDimIndex<2>(data_format, 'H')) &&
            IsMklWidthDim(GetTensorDimIndex<2>(data_format, 'W')));
  }

  /// The following methods are used for serializing and de-serializing the
  /// contents of the mklshape object.
  /// The data is serialized in this order
  /// is_mkl_tensor_ : dimension_ : sizes_ : map_: format_ : T_ : mkl_pd_;

  /// Size of buffer to hold the serialized object, the size is computed by
  /// following above mentioned order
  inline size_t GetSerializeBufferSize() const { return sizeof(MklShapeData); }

  void SerializeMklDnnShape(unsigned char* buf, size_t buf_size) const {
    CHECK(buf_size >= GetSerializeBufferSize())
        << "Buffer size is too small to SerializeMklDnnShape";
    *reinterpret_cast<MklShapeData*>(buf) = data_;
  }

  void DeSerializeMklDnnShape(const unsigned char* buf, size_t buf_size) {
    // Make sure buffer holds at least is_mkl_tensor_.
    CHECK(buf_size >= sizeof(data_.is_mkl_tensor_))
        << "Buffer size is too small in DeSerializeMklDnnShape";

    const bool is_mkl_tensor = *reinterpret_cast<const bool*>(buf);
    if (is_mkl_tensor) {  // If it is an MKL Tensor then read the rest
      CHECK(buf_size >= GetSerializeBufferSize())
          << "Buffer size is too small in DeSerializeMklDnnShape";
      data_ = *reinterpret_cast<const MklShapeData*>(buf);
    }
  }
};

// Get the MKL shape from the second string tensor
inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape,
                        bool eager_mode) {
  if (!eager_mode) {
    mklshape->DeSerializeMklDnnShape(
        ctext->input(n + ctext->num_inputs() / 2).flat<uint8>().data(),
        ctext->input(n + ctext->num_inputs() / 2).flat<uint8>().size() *
            sizeof(uint8));
  } else {
    mklshape->SetMklTensor(false);
  }
}

inline void GetMklShape(OpKernelContext* ctext, int n, MklDnnShape* mklshape) {
  GetMklShape(ctext, n, mklshape, false);
}

// Gets the actual input
inline const Tensor& MklGetInput(OpKernelContext* ctext, int n) {
  return ctext->input(n);
}

// Allocate the second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      const MklDnnShape& mkl_shape) {
  Tensor* second_tensor = nullptr;
  TensorShape second_shape;
  second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
  OP_REQUIRES_OK(ctext, ctext->allocate_output(n + ctext->num_outputs() / 2,
                                               second_shape, &second_tensor));
  mkl_shape.SerializeMklDnnShape(
      second_tensor->flat<uint8>().data(),
      second_tensor->flat<uint8>().size() * sizeof(uint8));
}

// Allocate the output tensor, create a second output tensor that will contain
// the MKL shape serialized
inline void AllocateOutputSetMklShape(OpKernelContext* ctext, int n,
                                      Tensor** output,
                                      const TensorShape& tf_shape,
                                      const MklDnnShape& mkl_shape,
                                      bool eager_mode = false) {
  OP_REQUIRES_OK(ctext, ctext->allocate_output(n, tf_shape, output));
  if (!eager_mode) {
    Tensor* second_tensor = nullptr;
    TensorShape second_shape;
    second_shape.AddDim(mkl_shape.GetSerializeBufferSize());
    OP_REQUIRES_OK(ctext, ctext->allocate_output(n + ctext->num_outputs(),
                                                 second_shape, &second_tensor));
    mkl_shape.SerializeMklDnnShape(
        second_tensor->flat<uint8>().data(),
        second_tensor->flat<uint8>().size() * sizeof(uint8));
  }
}

// Map MklTensorFormat to MKL-DNN format tag
//
// @input: MklTensorFormat i.e. TensorFlow data format
// @return: MKL-DNN's memory format tag corresponding to MklTensorFormat.
//          Fails with an error if invalid data format.
inline dnnl::memory::format_tag MklTensorFormatToMklDnnDataFormat(
    MklTensorFormat format) {
  if (format == MklTensorFormat::FORMAT_NHWC)
    return dnnl::memory::format_tag::nhwc;
  if (format == MklTensorFormat::FORMAT_NCHW)
    return dnnl::memory::format_tag::nchw;
  if (format == MklTensorFormat::FORMAT_NDHWC)
    return dnnl::memory::format_tag::ndhwc;
  if (format == MklTensorFormat::FORMAT_NCDHW)
    return dnnl::memory::format_tag::ncdhw;
  if (format == MklTensorFormat::FORMAT_X) return dnnl::memory::format_tag::x;
  if (format == MklTensorFormat::FORMAT_NC) return dnnl::memory::format_tag::nc;
  if (format == MklTensorFormat::FORMAT_TNC)
    return dnnl::memory::format_tag::tnc;
  return dnnl::memory::format_tag::undef;
}

/// Map TensorFlow data format into MKL-DNN 3D data format
/// @input: TensorFlow data format
/// @return: MKL-DNN 3D data format corresponding to TensorFlow data format;
///          Fails with an error if invalid data format.
inline MklTensorFormat TFDataFormatToMklDnn3DDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MklTensorFormat::FORMAT_NDHWC;
  if (format == FORMAT_NCHW) return MklTensorFormat::FORMAT_NCDHW;
  TF_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));
  return MklTensorFormat::FORMAT_INVALID;
}

/// Map TensorFlow data format into MKL-DNN data format
///
/// @input: TensorFlow data format
/// @return: MKL-DNN data format corresponding to TensorFlow data format;
///          Fails with an error if invalid data format.
inline MklTensorFormat TFDataFormatToMklDnnDataFormat(TensorFormat format) {
  if (format == FORMAT_NHWC) return MklTensorFormat::FORMAT_NHWC;
  if (format == FORMAT_NCHW) return MklTensorFormat::FORMAT_NCHW;
  TF_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));
  return MklTensorFormat::FORMAT_INVALID;
}

/// Map MKL-DNN data format into TensorFlow data format
///
/// @input: MKL-DNN data format
/// @return: Tensorflow data format corresponding to MKL-DNN data format;
///          Fails with an error if invalid data format.
inline TensorFormat MklDnnDataFormatToTFDataFormat(MklTensorFormat format) {
  if (format == MklTensorFormat::FORMAT_NHWC ||
      format == MklTensorFormat::FORMAT_NDHWC)
    return FORMAT_NHWC;
  if (format == MklTensorFormat::FORMAT_NCHW ||
      format == MklTensorFormat::FORMAT_NCDHW)
    return FORMAT_NCHW;
  TF_CHECK_OK(Status(TF_INVALID_ARGUMENT, "Unsupported data format"));

  // Return to prevent compiler warnings, otherwise TF_CHECK_OK will ensure
  // that we don't come here.
  return FORMAT_NHWC;
}

/// Map TensorShape object into dnnl::memory::dims required by MKL-DNN
///
/// This function will simply map input TensorShape into MKL-DNN dims
/// naively. So it will preserve the order of dimensions. E.g., if
/// input tensor is in NHWC format, then dims will be in NHWC format also.
///
/// @input TensorShape object in shape
/// @return dnnl::memory::dims corresponding to TensorShape
inline dnnl::memory::dims TFShapeToMklDnnDims(const TensorShape& shape) {
  if (shape.dims() == 0) {
    dnnl::memory::dims dims{shape.num_elements()};
    return dims;
  }
  dnnl::memory::dims dims(shape.dims());
  for (int d = 0; d < shape.dims(); ++d) {
    dims[d] = shape.dim_size(d);
  }
  return dims;
}

/// Map TensorShape object into dnnl::memory::dims in NCHW format required by
/// MKL-DNN
///
/// This function is a specific one than above function. It will map input
/// TensorShape into MKL-DNN dims in NCHW format. So it may not preserve the
/// order of dimensions. E.g., if input tensor is in NHWC format, then dims
/// will be in NCHW format, and not in NHWC format.
///
/// @input TensorShape object in shape
/// @return dnnl::memory::dims in MKL-DNN required NCHW format
inline dnnl::memory::dims TFShapeToMklDnnDimsInNCHW(const TensorShape& shape,
                                                    TensorFormat format) {
  // Check validity of format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);

  int n = shape.dim_size(GetTensorDimIndex(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex(format, 'C'));
  int h = shape.dim_size(GetTensorDimIndex(format, 'H'));
  int w = shape.dim_size(GetTensorDimIndex(format, 'W'));

  // MKL-DNN requires dimensions in NCHW format.
  return dnnl::memory::dims({n, c, h, w});
}

inline dnnl::memory::dims TFShapeToMklDnnDimsInNCDHW(const TensorShape& shape,
                                                     TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnn3DDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);

  int n = shape.dim_size(GetTensorDimIndex<3>(format, 'N'));
  int c = shape.dim_size(GetTensorDimIndex<3>(format, 'C'));
  int d = shape.dim_size(GetTensorDimIndex<3>(format, '0'));
  int h = shape.dim_size(GetTensorDimIndex<3>(format, '1'));
  int w = shape.dim_size(GetTensorDimIndex<3>(format, '2'));

  // MKL-DNN requires dimensions in NCDHW format.
  return dnnl::memory::dims({n, c, d, h, w});
}

/// Overloaded version of function TFShapeToMklDnnDimsInNCHW above.
/// Input parameters are self-explanatory.
inline dnnl::memory::dims MklDnnDimsInNCHW(const dnnl::memory::dims& in_dims,
                                           TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);

  int n = in_dims[GetTensorDimIndex(format, 'N')];
  int c = in_dims[GetTensorDimIndex(format, 'C')];
  int h = in_dims[GetTensorDimIndex(format, 'H')];
  int w = in_dims[GetTensorDimIndex(format, 'W')];

  // MKL-DNN requires dimensions in NCHW format.
  return dnnl::memory::dims({n, c, h, w});
}

/// Overloaded version of function TFShapeToMklDnnDimsInNCDHW above.
/// Input parameters are self-explanatory.
inline dnnl::memory::dims MklDnnDimsInNCDHW(const dnnl::memory::dims& in_dims,
                                            TensorFormat format) {
  // Validate format.
  DCHECK_NE(TFDataFormatToMklDnnDataFormat(format),
            MklTensorFormat::FORMAT_INVALID);

  int n = in_dims[GetTensorDimIndex<3>(format, 'N')];
  int c = in_dims[GetTensorDimIndex<3>(format, 'C')];
  int d = in_dims[GetTensorDimIndex<3>(format, '0')];
  int h = in_dims[GetTensorDimIndex<3>(format, '1')];
  int w = in_dims[GetTensorDimIndex<3>(format, '2')];

  // MKL DNN requires dimensions in NCDHW format.
  return dnnl::memory::dims({n, c, d, h, w});
}

/// Map MklDnn dnnl::memory::dims object into TensorShape object.
///
/// This function will simply map input shape in MKL-DNN dnnl::memory::dims
/// format in Tensorflow's TensorShape object by preserving dimension order.
///
/// @input MKL-DNN dnnl::memory::dims object
/// @output TensorShape corresponding to dnnl::memory::dims
inline TensorShape MklDnnDimsToTFShape(const dnnl::memory::dims& dims) {
  std::vector<int32> shape(dims.size(), -1);
  for (int d = 0; d < dims.size(); d++) {
    shape[d] = dims[d];
  }

  TensorShape ret;
  CHECK_EQ(TensorShapeUtils::MakeShape(shape, &ret).ok(), true);
  return ret;
}

/// Function to calculate strides given tensor shape in Tensorflow order
/// E.g., if dims_tf_order is {1, 2, 3, 4}, then as per Tensorflow convention,
/// dimension with size 1 is outermost dimension; while dimension with size 4 is
/// innermost dimension. So strides for this tensor would be {4 * 3 * 2,
/// 4 * 3, 4, 1}, i.e., {24, 12, 4, 1}.
///
/// @input Tensorflow shape in dnnl::memory::dims type
/// @return dnnl::memory::dims containing strides for the tensor.
inline dnnl::memory::dims CalculateTFStrides(
    const dnnl::memory::dims& dims_tf_order) {
  CHECK_GT(dims_tf_order.size(), 0);
  dnnl::memory::dims strides(dims_tf_order.size());
  int last_dim_idx = dims_tf_order.size() - 1;
  strides[last_dim_idx] = 1;
  for (int d = last_dim_idx - 1; d >= 0; d--) {
    strides[d] = strides[d + 1] * dims_tf_order[d + 1];
  }
  return strides;
}

/// Helper function to create memory descriptor in Blocked format
///
/// @input: Tensor dimensions
/// @input: strides corresponding to dimensions. One can use utility
///         function such as CalculateTFStrides to compute strides
///         for given dimensions.
/// @output: dnnl_memory_desc_t object corresponding to blocked memory
///          format for given dimensions and strides.
/// @return: Status indicating whether the blocked memory descriptor
///          was successfully created.
inline Status CreateBlockedMemDescHelper(const dnnl::memory::dims& dim,
                                         const dnnl::memory::dims& strides,
                                         dnnl::memory::data_type dtype,
                                         dnnl_memory_desc_t* blocked_md) {
  DCHECK_EQ(dim.size(), strides.size());
  const int kNumDims = dim.size();
  dnnl_dim_t input_dims[kNumDims];
  dnnl_dim_t input_strides[kNumDims];
  for (int i = 0; i < kNumDims; ++i) {
    input_dims[i] = dim[i];
    input_strides[i] = strides[i];
  }
  try {
    dnnl_memory_desc_init_by_strides(blocked_md, kNumDims, input_dims,
                                     dnnl::memory::convert_to_c(dtype),
                                     input_strides);
  } catch (dnnl::error& e) {
    return Status(TF_INTERNAL,
                  intel_plugin::strings::StrCat(
                      "Failed to create blocked memory descriptor.",
                      "Status: ", e.status, ", message: ", e.message));
  }
  return Status::OK();
}

/// Helper function to create memory descriptor in Blocked format
///
/// @input: Tensor dimensions
/// @input: strides corresponding to dimensions. One can use utility
///         function such as CalculateTFStrides to compute strides
///         for given dimensions.
/// @return: dnnl::memory::desc object corresponding to blocked memory format
///          for given dimensions and strides.
template <typename T>
inline dnnl::memory::desc CreateBlockedMemDesc(
    const dnnl::memory::dims& dim, const dnnl::memory::dims& strides) {
  dnnl_memory_desc_t blocked_md;
  TF_CHECK_OK(
      CreateBlockedMemDescHelper(dim, strides, MklDnnType<T>(), &blocked_md));
  return dnnl::memory::desc(blocked_md);
}

// Class to represent all the resources corresponding to a tensor in TensorFlow
// that are required to execute an operation (such as Convolution).
template <typename T>
class MklDnnData {
 private:
  /// MKL-DNN memory primitive for input user memory
  dnnl::memory* user_memory_;

  /// MKL-DNN memory primitive in case input or output reorder is needed.
  dnnl::memory* reorder_memory_;

  /// Operations memory descriptor
  dnnl::memory::desc* op_md_;

  // flat to indicate if data is 3D or not.
  bool bIs3D;

  // OneDNN engine.
  const dnnl::engine* onednn_engine_;

 public:
  explicit MklDnnData(const dnnl::engine* e)
      : user_memory_(nullptr),
        reorder_memory_(nullptr),
        op_md_(nullptr),
        bIs3D(false),
        onednn_engine_(e) {}

  ~MklDnnData() {
    onednn_engine_ = nullptr;  // We don't own this.
    delete (user_memory_);
    delete (reorder_memory_);
    delete (op_md_);
  }

  inline void* GetTensorBuffer(const Tensor* tensor) const {
    CHECK_NOTNULL(tensor);
    return const_cast<void*>(
        static_cast<const void*>(tensor->flat<T>().data()));
  }

  /// Set user memory primitive using specified dimensions, memory format tag
  /// and data_buffer. Function automatically uses element data type by using
  /// input type T used for creating call object.
  ///
  /// In a nutshell, function allows user to describe the input tensor to
  /// an operation. E.g., filter of Conv2D is of shape {1, 2, 3, 4}, and
  /// memory format tag HWIO, and the buffer that contains actual values is
  /// pointed by data_buffer.
  inline void SetUsrMem(const dnnl::memory::dims& dim,
                        dnnl::memory::format_tag fm,
                        void* data_buffer = nullptr) {
    auto md = dnnl::memory::desc(dim, MklDnnType<T>(), fm);
    SetUsrMem(md, data_buffer);
  }

  inline void SetUsrMem(const dnnl::memory::dims& dim,
                        dnnl::memory::format_tag fm, const Tensor* tensor) {
    DCHECK(tensor);
    SetUsrMem(dim, fm, GetTensorBuffer(tensor));
  }

  /// A version of SetUsrMem call that allows user to create memory in blocked
  /// format. So in addition to accepting dimensions, it also accepts strides.
  /// This allows user to create memory for tensor in a format that is not
  /// supported by MKLDNN. E.g., MKLDNN does not support tensor format for 6
  /// dimensional tensor as a native format. But by using blocked format, a user
  /// can create memory for 6D tensor.
  inline void SetUsrMem(const dnnl::memory::dims& dim,
                        const dnnl::memory::dims& strides,
                        void* data_buffer = nullptr) {
    CHECK_EQ(dim.size(), strides.size());
    auto blocked_md = CreateBlockedMemDesc<T>(dim, strides);
    SetUsrMem(blocked_md, data_buffer);
  }

  inline void SetUsrMem(const dnnl::memory::dims& dim,
                        const dnnl::memory::dims& strides,
                        const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(dim, strides, GetTensorBuffer(tensor));
  }

  /// A version of SetUsrMem with memory descriptor and tensor
  inline void SetUsrMem(const dnnl::memory::desc& md, const Tensor* tensor) {
    CHECK_NOTNULL(tensor);
    SetUsrMem(md, GetTensorBuffer(tensor));
  }

  /// A version of function to set user memory type that accepts memory
  /// descriptor directly, instead of accepting dimensions and format. This
  /// function is more generic than the one above, but the function above is
  /// sufficient in most cases.
  inline void SetUsrMem(const dnnl::memory::desc& pd,
                        void* data_buffer = nullptr) {
    DCHECK(onednn_engine_);
    if (user_memory_) delete user_memory_;
    // TODO(nhasabni): can we remove dynamic memory allocation?
    if (data_buffer) {
      user_memory_ = new dnnl::memory(pd, *onednn_engine_, data_buffer);
    } else {
      user_memory_ = new dnnl::memory(pd, *onednn_engine_);
    }
  }

  /// Get the memory primitive for input and output of an op. If inputs
  /// to an op require reorders, then this function returns memory primitive
  /// for reorder. Otherwise, it will return memory primitive for user memory.
  ///
  /// E.g., Conv2D(I, F) is a primitive with I and F being inputs. Then to
  /// execute Conv2D, we need memory primitive for I and F. But if reorder is
  /// required for I and F (say I_r is reorder primitive for I; F_r is reorder
  /// primitive for F), then we need I_r and F_r to perform Conv2D.
  inline const dnnl::memory& GetOpMem() const {
    return reorder_memory_ ? *reorder_memory_ : *user_memory_;
  }

  /// Set memory descriptor of an operation in terms of dimensions and memory
  /// format. E.g., For Conv2D, the dimensions would be same as user dimensions
  /// but dnnl::memory::format_tag would be dnnl::any because we want MKL-DNN to
  /// choose the best layout/format for given input dimensions.
  inline void SetOpMemDesc(const dnnl::memory::dims& dim,
                           dnnl::memory::format_tag fm) {
    // TODO(nhasabni): can we remove dynamic memory allocation?
    op_md_ = new dnnl::memory::desc(dim, MklDnnType<T>(), fm);
  }

  /// Get function for memory descriptor for an operation
  inline const dnnl::memory::desc& GetOpMemDesc() const { return *op_md_; }

  /// Predicate that checks if we need to reorder user's memory into memory
  /// pointed by op_md.
  ///
  /// @input: op_md - memory descriptor of the given input of an operation.
  /// @return: true in case reorder of input is needed; false, otherwise.
  inline bool IsReorderNeeded(const dnnl::memory::desc& op_pd) const {
    DCHECK(user_memory_);
    return op_pd != user_memory_->get_desc();
  }

  /// Function to create a reorder from memory pointed by from to memory pointed
  /// by to. Returns created primitive.
  inline dnnl::primitive CreateReorder(const dnnl::memory* from,
                                       const dnnl::memory* to) const {
    CHECK_NOTNULL(from);
    CHECK_NOTNULL(to);
    return dnnl::reorder(*from, *to);
  }

  /// Overloaded version of above function that accepts memory buffer
  /// where output of reorder needs to be stored.
  ///
  /// @input: op_pd - memory primitive descriptor (memory descriptor for v1.x)
  ///                 of the given input of an operation
  /// @reorder_data_handle - memory buffer where output of reorder needs to be
  ///                        stored. Primitive does not check if buffer has
  ///                        enough size to write.
  /// @input: net - net to which to add reorder primitive in case it is needed.
  /// @input: net_args - net to which user and reorder memories are added if
  ///                    needed. Each entry is a key-value pair of the form
  ///                    <argument-type, dnnl::memory>.
  /// @input: engine - MKL-DNN's abstraction of a computational device
  /// @return: true in case reorder of input is needed; false, otherwise.
  inline bool CheckReorderToOpMem(const dnnl::memory::desc& op_md,
                                  void* reorder_data_handle,
                                  std::vector<dnnl::primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const dnnl::engine& engine) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      // TODO(nhasabni): can we remove dynamic memory allocation?
      reorder_memory_ = GetMkldnnMemory(op_md, engine, reorder_data_handle);
      net.push_back(CreateReorder(user_memory_, reorder_memory_));
      net_args.push_back(MemoryArgsMap{{DNNL_ARG_FROM, *user_memory_},
                                       {DNNL_ARG_TO, *reorder_memory_}});

      return true;
    }
    return false;
  }

  /// This is a faster path with reorder primitive cache compared with
  /// CheckReorderToOpMem(..., std::vector<primitive>* net).
  /// The slower path will be removed in the future
  /// TODO(bhavanis): Need to use reorder cache here for better performance.
  inline bool CheckReorderToOpMem(const dnnl::memory::desc& op_md,
                                  void* reorder_data_handle,
                                  const dnnl::engine& engine,
                                  OpKernelContext* context = nullptr) {
    DCHECK(reorder_data_handle);
    DCHECK(user_memory_);
    if (IsReorderNeeded(op_md)) {
      reorder_memory_ = GetMkldnnMemory(op_md, engine, reorder_data_handle);
      auto prim = dnnl::reorder(*user_memory_, *reorder_memory_);
      auto onednn_stream = CreateDnnlStream(*context, engine);
      MemoryArgsMap net_args{{DNNL_ARG_FROM, *user_memory_},
                             {DNNL_ARG_TO, *reorder_memory_}};
      prim.execute(onednn_stream, net_args);
      return true;
    }
    return false;
  }

  /// Another overloaded version of CheckReorderToOpMem that accepts Tensor
  /// where output of reorder needs to be stored.
  ///
  /// @input: op_md - memory primitive descriptor (memory descriptor for v1.x)
  ///                 of the given input of an operation
  /// @reorder_tensor - Tensor whose buffer is to be used to store output of
  ///                   reorder. Primitive does not check if buffer is
  ///                   enough size to write.
  /// @input: net - net to which to add reorder primitive in case it is needed.
  /// @input: net_args - net to which user and reorder memories are added if
  ///                    needed. Each entry is a key-value pair of the form
  ///                    <argument-type, dnnl::memory>.
  /// @input: engine - MKL-DNN's abstraction of a computational device
  /// @return: true in case reorder of input is needed; false, otherwise.
  inline bool CheckReorderToOpMem(const dnnl::memory::desc& op_md,
                                  Tensor* reorder_tensor,
                                  std::vector<dnnl::primitive>& net,
                                  std::vector<MemoryArgsMap>& net_args,
                                  const dnnl::engine& engine) {
    DCHECK(reorder_tensor);
    return CheckReorderToOpMem(op_md, GetTensorBuffer(reorder_tensor), net,
                               net_args, engine);
  }

  /// TODO: this is a faster path with reorder primitive cache compared with
  /// CheckReorderToOpMem(op_md, reorder_tensor, net, net_args, engine), will
  /// remove
  /// slow path in the future
  inline bool CheckReorderToOpMem(const dnnl::memory::desc& op_pd,
                                  Tensor* reorder_tensor,
                                  OpKernelContext* ctx = nullptr) {
    DCHECK(reorder_tensor);
    return CheckReorderToOpMem(op_pd, GetTensorBuffer(reorder_tensor),
                               *onednn_engine_, ctx);
  }
};

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_DPCPP_UTIL_H_
