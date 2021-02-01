#ifndef TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OP_REQUIRES_H_
#define TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OP_REQUIRES_H_

#include "tensorflow_plugin/src/xpu_core/util/macros.h"

namespace intel_plugin {

// Convenience macros for asserting and handling exceptional conditions.
// Analogous to the CHECK* macros provided by logging.h.
//
// Example use:
// void Compute(OperationContext* context) {
//   OP_REQUIRES(context, context->num_inputs() == 2,
//               errors::InvalidArgument("FooOp requires 2 arguments"));
//   ...
//   Status status = SomeUncertainMethod();
//   OP_REQUIRES_OK(context, status);
//   ...
// }
//
// These macros depend on CheckNotInComputeAsync, which must be defined before
// invoking the macro. We specifically don't include op_kernel.h from this
// header to reduce this header's dependencies. These macros may be used with
// alternative implementations of OpKernelContext with fewer dependencies.

#define OP_REQUIRES(CTX, EXP, STATUS)                     \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return;                                             \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK(CTX, ...)                             \
  do {                                                       \
    ::intel_plugin::Status _s(__VA_ARGS__);                  \
    if (!TF_PREDICT_TRUE(_s.ok())) {                         \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return;                                                \
    }                                                        \
  } while (0)

#define OP_REQUIRES_ASYNC(CTX, EXP, STATUS, CALLBACK)  \
  do {                                                 \
    if (!TF_PREDICT_TRUE(EXP)) {                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      (CALLBACK)();                                    \
      return;                                          \
    }                                                  \
  } while (0)

#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)         \
  do {                                                      \
    ::intel_plugin::Status _s(STATUS);                      \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      (CALLBACK)();                                         \
      return;                                               \
    }                                                       \
  } while (0)

#define OP_REQUIRES_PTR(CTX, EXP, STATUS)                 \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return nullptr;                                     \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK_PTR(CTX, ...)                         \
  do {                                                       \
    ::intel_plugin::Status _s(__VA_ARGS__);                  \
    if (!TF_PREDICT_TRUE(_s.ok())) {                         \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return nullptr;                                        \
    }                                                        \
  } while (0)

#define OP_REQUIRES_ASYNC_PTR(CTX, EXP, STATUS, CALLBACK) \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      (CALLBACK)();                                       \
      return nullptr;                                     \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK_ASYNC_PTR(CTX, STATUS, CALLBACK)     \
  do {                                                      \
    ::intel_plugin::Status _s(STATUS);                      \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      (CALLBACK)();                                         \
      return nullptr;                                       \
    }                                                       \
  } while (0)

}  // namespace intel_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_XPU_CORE_UTIL_OP_REQUIRES_H_
