#include "tensorflow_plugin/src/xpu_core/util/op_kernel.h"

#include <iostream>
#include <string>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/plugin_tensor.h"

namespace intel_plugin {

int OpKernelContext::num_inputs() const { return TF_NumInputs(ctx_); }

DataType OpKernelContext::input_dtype(int index) const {
  if (inputs_ != nullptr) {
    return static_cast<DataType>(
        TF_TensorType(inputs_->at(index).get()->GetTFTensor()));
  }
  TF_Tensor* tensor = nullptr;
  TF_GetInput(ctx_, index, &tensor, status_);
  return static_cast<DataType>(TF_TensorType(tensor));
}

// Status OpKernelContext::input_dtype(StringPiece name, DataType* dtype) const;

MemoryType OpKernelContext::input_memory_type(int index) const {
  DataType dtype = input_dtype(index);
  return MTypeFromDType(dtype);
}

int OpKernelContext::num_outputs() const { return TF_NumOutputs(ctx_); }

DataType OpKernelContext::expected_output_dtype(int index) const {
  return static_cast<DataType>(TF_ExpectedOutputDataType(ctx_, index));
};

const Tensor& OpKernelContext::input(int index) const {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_inputs());
  if (inputs_ == nullptr) {
    inputs_ = new gtl::InlinedVector<std::shared_ptr<Tensor>, 4>();
    TF_Tensor* tensor = nullptr;

    for (auto i = 0; i < num_inputs(); ++i) {
      TF_GetInput(ctx_, i, &tensor, status_);
      TensorShape shape;
      auto dims = TF_NumDims(tensor);
      for (auto j = 0; j < dims; ++j) {
        shape.AddDim(TF_Dim(tensor, j));
      }
      std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
          static_cast<DataType>(TF_TensorType(tensor)), shape, tensor);
      inputs_->push_back(std::move(ptr));
    }
  }
  CHECK_NE(inputs_, nullptr);
  return *inputs_->at(index);
}

bool OpKernelContext::ValidateInputsAreSameShape() {
  OpKernelContext ctx(ctx_);

  const size_t kNumInputs = ctx.num_inputs();
  for (size_t i = 1; i < kNumInputs; ++i) {
    if (!ctx.input(0).IsSameSize(ctx.input(i))) {
      ctx.CtxFailure(errors::InvalidArgument(
          "Inputs must have the same size and shape. Input 0: ",
          ctx.input(0).shape().DebugString(), " != input ", std::to_string(i),
          ": ", ctx.input(i).shape().DebugString()));
      return false;
    }
  }

  return true;
}

Status OpKernelContext::forward_input_or_allocate_output(
    gtl::ArraySlice<int> candidate_input_indices, int output_index,
    const TensorShape& output_shape, Tensor** output, int* forwarded_input) {
  CHECK_GE(output_index, 0);
  CHECK_LT(output_index, num_outputs());
  TF_Tensor* tensor = TF_ForwardInputOrAllocateOutput(
      ctx_, const_cast<int*>(candidate_input_indices.data()),
      candidate_input_indices.size(), output_index,
      output_shape.dim_sizes().data(), output_shape.dims(), forwarded_input,
      status_);
  if (outputs_[output_index] == nullptr) {
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(expected_output_dtype(output_index)),
        output_shape, tensor);
    outputs_[output_index] = std::move(ptr);
  }

  *output = outputs_[output_index].get();
  return StatusFromTF_Status(status_);
}

Status OpKernelContext::allocate_output(int index, const TensorShape& shape,
                                        Tensor** tensor) {
  DataType out_type = static_cast<DataType>(expected_output_dtype(index));
  TF_Tensor* output = TF_AllocateOutput(
      ctx_, index, static_cast<TF_DataType>(out_type), shape.dim_sizes().data(),
      shape.dims(), shape.num_elements() * DataTypeSize(out_type), status_);
  if (outputs_[index] == nullptr) {
    std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(
        static_cast<DataType>(expected_output_dtype(index)), shape, output);
    outputs_[index] = std::move(ptr);
  }
  *tensor = outputs_[index].get();

  return StatusFromTF_Status(status_);
}

Status OpKernelContext::allocate_temp(
    DataType type, const TensorShape& shape, Tensor* out_temp,
    AllocatorAttributes allocator_attr,
    const AllocationAttributes& allocation_attr) {
  TF_Tensor* tmp = TF_AllocateTemp(ctx_, static_cast<TF_DataType>(type),
                                   shape.dim_sizes().data(), shape.dims(),
                                   &allocator_attr.plugin_attr(), status_);
  Tensor t(type, shape, tmp);
  *out_temp = std::move(t);

  return StatusFromTF_Status(status_);
}

void OpKernelContext::set_output(int index, const Tensor& tensor) {
  CHECK(index < num_outputs()) << " Index out of range while setting output";
  TF_SetOutput(ctx_, index, tensor.GetTFTensor(), status_);
  CHECK_EQ(TF_OK, TF_GetCode(status_)) << " Error while setting output";
  return;
}

/// all below CtxFailure will pass back the TF_Status created by plugin.
/// so we need not to delete it, which will be controlled by TF.
void OpKernelContext::CtxFailure(const Status& s) {
  VLOG(1) << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}
void OpKernelContext::CtxFailure(const char* file, int line, const Status& s) {
  LOG(WARNING) << file << ": " << line << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelContext::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}
void OpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                            const Status& s) {
  LOG(WARNING) << file << line << s;
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelContext::SetStatus(const Status& s) {
  TF_OpKernelContext_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

// class OpKernelConstruction -------------------------------------------------
bool OpKernelConstruction::HasAttr(StringPiece attr_name) const {
  // note that StringPiece.data() will return a not nul-terminated char*
  // so will need std::string.c_str()
  std::string name(attr_name.data(), attr_name.size());
  bool ret = TF_OpKernelConstruction_HasAttr(ctx_, name.c_str(), status_);
  return ret;
}

template <>
Status OpKernelConstruction::GetAttr<int32_t>(StringPiece attr_name,
                                              int32_t* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrInt32(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<DataType>(StringPiece attr_name,
                                               DataType* value) const {
  TF_DataType type;
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrType(ctx_, name.c_str(), &type, status_);
  *value = static_cast<DataType>(type);
  return StatusFromTF_Status(status_);
}

template <>
Status OpKernelConstruction::GetAttr<int64_t>(StringPiece attr_name,
                                              int64_t* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrInt64(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<float>(StringPiece attr_name,
                                            float* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrFloat(ctx_, name.c_str(), value, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<bool>(StringPiece attr_name,
                                           bool* value) const {
  std::string name(attr_name.data(), attr_name.size());
  TF_OpKernelConstruction_GetAttrBool(
      ctx_, name.c_str(), reinterpret_cast<unsigned char*>(value), status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::string>(StringPiece attr_name,
                                                  std::string* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  std::vector<char> val(total_size);
  TF_OpKernelConstruction_GetAttrString(ctx_, name.c_str(), val.data(),
                                        total_size, status_);
  *value = std::string(val.data(), total_size);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<string>>(
    StringPiece attr_name, std::vector<std::string>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);

  value->resize(list_size);

  std::unique_ptr<void*[]> vals(new void*[list_size]);
  std::unique_ptr<size_t[]> lens(new size_t[list_size]);
  std::unique_ptr<char[]> storage(new char[total_size]);
  size_t storage_size(total_size);
  TF_OpKernelConstruction_GetAttrStringList(
      ctx_, name.c_str(), reinterpret_cast<char**>(vals.get()), lens.get(),
      list_size, storage.get(), storage_size, status_);

  for (int32_t i = 0; i < list_size; ++i) {
    (*value)[i] = string(static_cast<const char*>(vals[i]), lens[i]);
  }

  return StatusFromTF_Status(status_);
}
// TODO(yangshe1): Update if these apis are changed.
template <>
Status OpKernelConstruction::GetAttr<std::vector<int32_t>>(
    StringPiece attr_name, std::vector<int32_t>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;
  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrInt32List(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<DataType>>(
    StringPiece attr_name, std::vector<DataType>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  std::vector<TF_DataType> val(list_size);
  TF_OpKernelConstruction_GetAttrTypeList(ctx_, name.c_str(), val.data(),
                                          list_size, status_);
  for (int i = 0; i < list_size; i++)
    (*value)[i] = static_cast<DataType>(val[i]);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<int64_t>>(
    StringPiece attr_name, std::vector<int64_t>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrInt64List(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}
template <>
Status OpKernelConstruction::GetAttr<std::vector<float>>(
    StringPiece attr_name, std::vector<float>* value) const {
  std::string name(attr_name.data(), attr_name.size());
  int32_t list_size = 0;
  int32_t total_size = 0;

  TF_OpKernelConstruction_GetAttrSize(ctx_, name.c_str(), &list_size,
                                      &total_size, status_);
  value->resize(list_size);
  TF_OpKernelConstruction_GetAttrFloatList(ctx_, name.c_str(), value->data(),
                                           list_size, status_);
  return StatusFromTF_Status(status_);
}

void OpKernelConstruction::CtxFailure(const Status& s) {
  VLOG(1) << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailure(const char* file, int line,
                                      const Status& s) {
  LOG(WARNING) << file << ":" << line << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailureWithWarning(const Status& s) {
  LOG(WARNING) << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                 const Status& s) {
  LOG(WARNING) << file << ": " << line << s;
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

void OpKernelConstruction::SetStatus(const Status& s) {
  TF_OpKernelConstruction_Failure(ctx_, TF_StatusFromStatus(s, status_));
}

const char* OpKernelConstruction::OpName() const {
  return TF_OpKernelConstruction_GetName(ctx_).data;
}

OpKernel::OpKernel(OpKernelConstruction* context) {}

OpKernel::~OpKernel() {}

KernelDefBuilder& KernelDefBuilder::Device(const char* backend) {
  backend_ = std::string(backend);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::HostMemory(const char* host) {
  host_memorys_.push_back(std::string(host));
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Priority(const int32 priority_number) {
  priority_ = priority_number;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterCreate(KernelCreateFunc func) {
  create_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterCompute(KernelComputeFunc func) {
  compute_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::RegisterDelete(KernelDeleteFunc func) {
  delete_func_ = func;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::KernelClassName(
    const char* kernel_class_name) {
  kernel_class_name_ = std::string(kernel_class_name);
  return *this;
}

Name::Name(const char* op_name) { op_name_ = std::string(op_name); }

void Name::Build(const char* device_name, const char* backend) {
  if (backend != backend_) {
    return;
  }

  StatusUniquePtr status(TF_NewStatus());
  {
    auto builder =
        TF_NewKernelBuilder(op_name_.c_str(), device_name, create_func_,
                            compute_func_, delete_func_);

    auto check_type_constraint = [&builder, &status, this](DataType dtype,
                                                           const char* name) {
      auto data_type = static_cast<TF_DataType>(dtype);
      TF_KernelBuilder_TypeConstraint(builder, name, data_type, status.get());
      CHECK_EQ(TF_OK, TF_GetCode(status.get()))
          << "Error while registering " << kernel_class_name_
          << " kernel with attribute " << name;
    };

    for (int i = 0; i < type_constraints_.size(); i++) {
      auto& type_constraint = type_constraints_[i];
      auto& type_value = type_values_[i];
      check_type_constraint(type_value, type_constraint.c_str());
    }

    for (auto const& host_memory : host_memorys_) {
      TF_KernelBuilder_HostMemory(builder, host_memory.c_str());
    }

    if (priority_ > 0) {
      TF_KernelBuilder_Priority(builder, priority_);
    }

    TF_RegisterKernelBuilder(kernel_class_name_.c_str(), builder, status.get());
    CHECK_EQ(TF_OK, TF_GetCode(status.get()))
        << "Error while registering " << kernel_class_name_ << " kernel.";
  }
}

namespace register_kernel {
Registrar::Registrar(std::string key, KernelRegisterFunc func) {
  auto global_registry = GlobalKernelRegistry();
  mutex_lock l(global_registry->mu);
  global_registry->registry.push_back(std::make_pair(key, func));
}

KernelRegistry* GlobalKernelRegistry() {
  static KernelRegistry* global_kernel_registry = new KernelRegistry;
  return global_kernel_registry;
}

void RegisterCPUKernels(const char* device_name) {
  for (auto const& x : GlobalKernelRegistry()->registry) {
    KernelRegisterFunc func = x.second;
    func(device_name, DEVICE_CPU);
  }
}

void RegisterGPUKernels(const char* device_name) {
  for (auto const& x : GlobalKernelRegistry()->registry) {
    KernelRegisterFunc func = x.second;
    func(device_name, DEVICE_GPU);
  }
}

}  // namespace register_kernel

// PersistentTensor ----------------------------------------------------------

Tensor* PersistentTensor::AccessTensor(OpKernelConstruction* context) {
  // the caller has to have a valid context
  CHECK(context);
  return tensor_;
}

Tensor* PersistentTensor::AccessTensor(OpKernelContext* context) {
  return tensor_;
}
void CheckNotInComputeAsync(OpKernelContext* ctx,
                            const char* correct_macro_name) {
  VLOG(0) << __FILE__ << ":" << __LINE__ << "need impl!";
}
}  // namespace intel_plugin
