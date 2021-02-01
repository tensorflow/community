#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"

#include "tensorflow_plugin/src/xpu_core/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/xpu_core/util/tensor_shape.h"

namespace intel_plugin {

void Tensor::CheckTypeAndIsAligned(DataType expected_dtype) const {
  CHECK_EQ(dtype(), expected_dtype)
      << " " << DataTypeString(expected_dtype) << " expected, got "
      << DataTypeString(dtype());
  CHECK(IsAligned()) << "ptr = " << buf_;
}

void Tensor::CheckType(DataType expected_dtype) const {
  CHECK_EQ(dtype(), expected_dtype)
      << " " << DataTypeString(expected_dtype) << " expected, got "
      << DataTypeString(dtype());
}

void Tensor::CheckIsAlignedAndSingleElement() const {
  CHECK(IsAligned()) << "Aligned and single element";
  CHECK_EQ(1, NumElements()) << "Must have a one element tensor";
}

size_t Tensor::TotalBytes() const {
  return NumElements() * sizeof(DataTypeSize(dtype()));
}

size_t Tensor::AllocatedBytes() const { return TotalBytes(); }

Tensor::Tensor(DataType type, const TensorShape& shape, TF_Tensor* buf)
    : shape_(shape), buf_(buf) {
  shape_.set_data_type(type);
}

void Tensor::CopyFromInternal(const Tensor& other, const TensorShape& shape) {
  CHECK_EQ(shape.num_elements(), other.NumElements());
  // Data type will be overwritten if this == &other, since dtype is part of
  // shape.
  DataType other_dtype = other.dtype();
  shape_ = shape;
  set_dtype(other_dtype);
  if (buf_ != other.buf_) {
    buf_ = other.buf_;
  }
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatInnerDims(
    gtl::ArraySlice<int64> orig, int64 num_out_dims) {
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  int64 offset = orig.size() - num_out_dims;
  for (int64 out_dim = num_out_dims - 1; out_dim >= 0; --out_dim) {
    const int64 in_dim = out_dim + offset;
    out_dims[out_dim] = in_dim < 0 ? 1 : orig[in_dim];
  }
  for (int64 in_dim = 0; in_dim < offset; ++in_dim) {
    out_dims[0] *= orig[in_dim];
  }
  return out_dims;
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatOuterDims(
    gtl::ArraySlice<int64> orig, int64 num_out_dims) {
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  for (int64 out_dim = 0; out_dim <= num_out_dims - 1; ++out_dim) {
    out_dims[out_dim] = out_dim >= orig.size() ? 1 : orig[out_dim];
  }
  for (int64 in_dim = num_out_dims; in_dim < orig.size(); ++in_dim) {
    out_dims[num_out_dims - 1] *= orig[in_dim];
  }
  return out_dims;
}

static TF_Tensor* AllocateDummyTensor(DataType type) {
  const int64_t dims[1] = {1};
  TF_Tensor* tensor = TF_AllocateTensor(static_cast<TF_DataType>(type), dims, 1,
                                        DataTypeSize(type));
  return tensor;
}

Status Tensor::CopyFrom(const Tensor& other, const TensorShape& shape) {
  if (other.NumElements() != shape.num_elements()) {
    return errors::InvalidArgument("The tensor with ", other.NumElements(),
                                   " is not the same as shape's ",
                                   shape.num_elements(), " elements");
  }

  // This is a workaround because of `to` in TF_TensorBitcastFrom can't be
  // `nullptr`. Don't worry about the memory leak. It will be destructed out of
  // tensor life scope.
  if (!buf_) {
    buf_ = AllocateDummyTensor(other.dtype());
  }

  // copy the shape and dtype.
  DataType dtype = other.dtype();
  shape_ = std::move(shape);
  set_dtype(dtype);

  TF_Status* tf_status = TF_NewStatus();
  TF_TensorBitcastFrom(other.buf_, static_cast<TF_DataType>(dtype), buf_,
                       shape_.dim_sizes().data(), shape_.dims(), tf_status);

  Status status = StatusFromTF_Status(tf_status);
  TF_DeleteStatus(tf_status);
  return status;
}

bool Tensor::SharesBufferWith(const Tensor& other) {
  char* start = reinterpret_cast<char*>(data());
  char* end = start + NumElements() * DataTypeSize(dtype());

  char* other_start = reinterpret_cast<char*>(other.data());
  char* other_end =
      other_start + other.NumElements() + DataTypeSize(other.dtype());

  if (start < other_start && end > other_end) {
    return true;
  }
  return false;
}

StringPiece Tensor::tensor_data() const {
  if (GetTFTensor() == nullptr)
    return StringPiece();  // Don't die for empty tensors
  return StringPiece(static_cast<char*>(data()), TotalBytes());
}

Status MakeShape(const Tensor& shape, TensorShape* out) {
  if (!TensorShapeUtils::IsVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
}
}  // namespace intel_plugin
