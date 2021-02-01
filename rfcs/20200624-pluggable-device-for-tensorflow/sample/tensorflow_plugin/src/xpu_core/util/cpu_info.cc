#include "tensorflow_plugin/src/xpu_core/util/cpu_info.h"

namespace intel_plugin {
namespace port {
// SIMD extension querying is only available on x86.
#ifdef PLATFORM_IS_X86
#ifdef PLATFORM_WINDOWS
// Visual Studio defines a builtin function for CPUID, so use that if possible.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  {                                        \
    int cpu_info[4] = {-1};                \
    __cpuidex(cpu_info, a_inp, c_inp);     \
    a = cpu_info[0];                       \
    b = cpu_info[1];                       \
    c = cpu_info[2];                       \
    d = cpu_info[3];                       \
  }
#else
// Otherwise use gcc-format assembler to implement the underlying instructions.
#define GETCPUID(a, b, c, d, a_inp, c_inp) \
  asm("mov %%rbx, %%rdi\n"                 \
      "cpuid\n"                            \
      "xchg %%rdi, %%rbx\n"                \
      : "=a"(a), "=D"(b), "=c"(c), "=d"(d) \
      : "a"(a_inp), "2"(c_inp))
#endif
#endif

int CPUIDNumSMT() {
#ifdef PLATFORM_IS_X86
  // https://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration
  // https://software.intel.com/en-us/articles/intel-sdm (Vol 3A)
  // Section: Detecting Hardware Multi-threads Support and Topology
  // Uses CPUID Leaf 11 to enumerate system topology on Intel x86 architectures
  // Other cases not supported
  uint32 eax, ebx, ecx, edx;
  // Check if system supports Leaf 11
  GETCPUID(eax, ebx, ecx, edx, 0, 0);
  if (eax >= 11) {
    // 1) Leaf 11 available? CPUID.(EAX=11, ECX=0):EBX != 0
    // 2) SMT_Mask_Width = CPUID.(EAX=11, ECX=0):EAX[4:0] if CPUID.(EAX=11,
    // ECX=0):ECX[15:8] is 1
    GETCPUID(eax, ebx, ecx, edx, 11, 0);
    if (ebx != 0 && ((ecx & 0xff00) >> 8) == 1) {
      return 1 << (eax & 0x1f);  // 2 ^ SMT_Mask_Width
    }
  }
#endif  // PLATFORM_IS_X86
  return 0;
}
}  // namespace port
}  // namespace intel_plugin
