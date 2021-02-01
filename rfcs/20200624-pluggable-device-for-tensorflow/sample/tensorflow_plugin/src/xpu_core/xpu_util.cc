#include "xpu_util.h"
#include <cstring>

XPU_MODE get_xpu_mode() {
  const char* env = std::getenv("INTEL_XPU_MODE");
  if (env == nullptr) {
    return XPU_MODE_DEFAULT;
  } else if (strcmp(env, "GPU") == 0) {
    return XPU_MODE_GPU;
  } else if (strcmp(env, "XPU") == 0) {
    return XPU_MODE_XPU;
  } else {
    return XPU_MODE_DEFAULT;
  }
}

XPU_MAPPING get_xpu_mapping() {
  const char* env = std::getenv("INTEL_XPU_MAPPING");
  if (env == nullptr) {
    return XPU_MAPPING_DEFAULT;
  } else if (strcmp(env, "GPU") == 0) {
    return XPU_MAPPING_GPU;
  } else if (strcmp(env, "CPU") == 0) {
    return XPU_MAPPING_CPU;
  } else if (strcmp(env, "AUTO") == 0) {
    return XPU_MAPPING_AUTO;
  } else {
    return XPU_MAPPING_DEFAULT;
  }
}
