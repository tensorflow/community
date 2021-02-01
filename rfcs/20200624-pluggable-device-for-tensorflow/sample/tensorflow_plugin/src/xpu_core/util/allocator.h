#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ALLOCATOR_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ALLOCATOR_H_

#include "tensorflow/c/tf_tensor.h"
#include "tensorflow_plugin/src/xpu_core/util/integral_types.h"
#include "tensorflow_plugin/src/xpu_core/util/macros.h"

namespace intel_plugin {

struct AllocationAttributes {
  AllocationAttributes() = default;

  AllocationAttributes(bool retry_on_failure, bool allocation_will_be_logged,
                       std::function<uint64()>* freed_by_func)
      : retry_on_failure(retry_on_failure),
        allocation_will_be_logged(allocation_will_be_logged),
        freed_by_func(freed_by_func) {}

  // If the first attempt to allocate the memory fails, the allocation should
  // wait and retry (with a timeout).
  //
  // This is usually set to true, but we may set it to false in cases where a
  // failure has only performance impact (e.g. optional scratch space
  // allocation).
  bool retry_on_failure = true;
  // If a Tensor is allocated without the following set to true, then
  // it is logged as an unknown allocation. During execution Tensors
  // should be allocated through the OpKernelContext which records
  // which Op is performing the allocation, and sets this flag to
  // true.
  bool allocation_will_be_logged = false;
  // EXPERIMENTAL: If provided, then evaluates to a timing count such that only
  // a memory chunk whose freed_at_count is at this value or earlier may be
  // returned.
  std::function<uint64()>* freed_by_func = nullptr;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(AllocationAttributes);
};

struct AllocatorAttributes {
  AllocatorAttributes() : attr_({TF_ALLOCATOR_ATTRIBUTES_STRUCT_SIZE, 0}) {}
  void set_on_host(bool v) { attr_.on_host = 1; }
  bool on_host() const { return attr_.on_host == 1; }
  TF_AllocatorAttributes& plugin_attr() const { return attr_; }
  mutable TF_AllocatorAttributes attr_;
};

}  // namespace intel_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_ALLOCATOR_H_
