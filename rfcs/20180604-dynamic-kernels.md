# Dynamic Loading of Kernels in TensorFlow
 | Status | Proposed |
:-------------- | :-------------------------------------------------|
| **Author(s)** | Gunhan Gulsoy (Google) |
| **Sponsor**   | Martin Wicke (Google)  |
| **Updated**   | 2018-06-04             |
 ## Objective
 This document describes a new way to create and deploy new kernels for
TensorFlow. We propose deploying kernels in separate shared libraries (dso,
dylib or dll) and loading these at runtime. While at the moment the scope of
this document only covers **TensorFlow Python distribution**, we aim to
generalize this approach for all TF distributions. With this mechanism, we
would like to create the following capabilities:
 * Loading kernels dynamically at runtime from shared libraries.
* Being able to load multiple kernels for the same op/device pair, and pick the
  best one in terms of hardware compatibility and performance.
* Check the hardware and load the compatible kernels.
* Check compiler options used and load the compatible kernels.
 ## Overview
 For an Op, we need three pieces:
 * Python bindings, to make them accessible in the Python API
* C++ op implementation
* C++ Kernel implementation(s)
 This document proposes a new way on how **kernels** can be deployed and loaded.
 In the current mechanism, the only constraint is Python bindings have to be
executed/loaded after C++ op implementation is loaded. Kernels can be loaded at
any time. This makes our task easier. When a kernel is loaded, it registers
itself in the global registry with a string key. The string key is constructed
as follows: `op_name:device_name:(optional)label`
 To start this project off, what we propose is the following:
* Create a new API, `tf.load_kernel_library`
* Use the new API to load kernels from a different shared object.
 Then, we will start to build checks, to be more picky about the kernels we load.
* Build handling for loading multiple kernels for the same op and device pair.
* Enhance Global Kernel Registry to allow cleanup of registered kernels when a
  library is unloaded.
* Build the library compatibility checking mechanism, and unload libraries when
  they are found to be incompatible
 Finally, we will add the following advanced checks
* Keep track of which libraries provide which kernels
* Garbage collection of unqualified kernels, and their libraries.
 ## Detailed Current State
 While this document proposes a new way to **load kernels**, there is a lot of
ideas we would like to adopt from the way ops are loaded. Therefore, current
op loading mechanism is also described in this section.
 ### Op loading
 Currently, we can load op libraries from shared objects. When loading custom or
contrib ops, we also load their kernels. The following pseudocode describes how
the current custom/contrib op loading mechanism works:
 * Custom contrib op Python bindings are not loaded until they are accessed.
* At the first access, the `__init__` file of the custom op module calls `tf.load_op_library`
* `load_op_library` loads the shared object using `TF_LoadLibrary` in the C API
* Once the shared object is loaded, `load_op_library` now executes and loads the rest of the Python code in the op library.
 Now, diving deep into `TF_LoadLibrary`
* `TF_LoadLibrary` is called. This is just a thin wrapper and status checker around `tensorflow::LoadLibrary`
* `tensorflow::LoadLibrary` checks first if this shared object is already loaded
* In a serial way, making sure only one library is processed at a time:
  * It starts a watcher for `OpRegistry`, to get a list of ops included in the library
  * Try loading the library using `Environment::LoadLibrary`
  * Which just calls `tensorflow::internal::LoadLibrary`
  * Which is essentially just `dlopen`.
 ### Kernel loading
 Currently, kernel loading mechanism is simpler than the op loading mechanism, at least at loading time. The mechanism can be summarized as follows:
 * Kernels use `REGISTER_KERNEL_BUILDER` macro to create a static initializer
* The static initializer is just an object of type `OpKernelRegistrar`
* Which calls `OpKernelRegistrar::InitInternal`
* Which saves the kernel in the `GlobalKernelRegistry`, with a factory method.
* Kernel is read from the registry and instantiated when op tries to be executed.
 ## Design
 Here we will describe the details of the work we plan to perform. The work will be divided into three milestones:
 ### Milestone 1: Load kernels from shared objects
 This phase will just be a simple proof of concept, to show that loading kernels
from shared objects will work. The deliverables of this phase are:
 1. `tf.load_kernel_library` api. This new method on our API will be responsible
  for loading kernels from given shared objects, or folders containing shared
  objects. It will:
  * Load the given shared object, if it is an `.so` file
  * If a folder is given, load all `libtfkernel-*` shared object files in the folder
2. Split one or more kernels into a different shared object. This will involve:
  * Resolve the `BUILD` dependency mess to be able to create a reasonably small
    shared object for a kernel (size will be optimized later).
  * Resolve all symbol collisions stemming from the different shared objects,
    potentially both depending on core TF framework.
  * Finally, on the Python side of the op whose kernel is being split out, add
    the directive: `tf.load_kernel_library(“libtfkernel_kernel_name.so”)`
3. Get a bazel test to pass with a split kernel library
4. Get a working Python wheel file with a split kernel library, and run the
   kernel from the shared object.
 To simplify the proof of concept, at this stage we will only do this on linux.
 ### Milestone 2: Enable kernel compatibility checks
 Once the proof of concept is ready, we need to start building the fancier
features of the proposal. These will be:
 1. Create a mechanism to save the compiler options from bazel side, and make
  them available to read in C++ runtime.
2. Create a mechanism in addition to `KernelDef` to be stored in the
  `GlobalKernelRegistry` to help decide which kernels should be loaded. The
  following is the data structure we propose for this information:
 ```c
typedef struct TF_DsoDef {
  const char* name;
  const char* version;
};
 typedef struct TF_HardwareDef {
  const char** SIMD_ISA;  // Or enum
  int SIMD_ISA_length;
   char* cpu_arch;
   const char** accelerator;
  int accelerator_length;  
};
 typedef struct TF_CompilerDef {
  const char* compiler;
  const char* compiler_version;
  
  const char** compiler_options;
  int compiler_options_length;
   int memory_alignment;
};
 typedef struct TF_KernelBuildInfo {
  TF_DsoDef* dependencies;
  int dependencies_list;
   TF_HardwareDef hardware_def;
  TF_CompilerDef compiler_def;
};
```
3. Create Methods to extract all the above information from the core runtime,
  to check for compatibility with any given kernel library.
4. During kernel registration, implement checks for the following:
  * Is this kernel compatible with the given hardware
  * Is this kernel compatible with the software available on the system
  * Is this kernel ABI compatible with the core runtime
  * Is this kernel faster than any other kernels that are loaded. In this context faster means one of the following:
    * Better optimized for the hardware
    * Uses a special acceleration library such as MKL
5. Provide means to override some of the above checks for loading experimental kernels
6. Expand Global kernel registry to be functionally similar to the op registry. Op registry can unregister ops if there are any problems during the object loading, kernel registry should be able to do the same.
 ### Milestone 3: Make it work on different OSs
 While the above will be done on linux, we will have to get things to work on all operating systems we support. For macos, the issues are mainly around bazel bugs. For windows, we will have to be more careful about symbol collisions, and partial lockdown of symbol exports may be required to get things working.
 ### Milestone 4: Memory and performance optimizations
 When we load multiple shared objects, we can easily have some bloat in memory
usage, or performance hits. The simplest things we can foresee are:
 1. Multiple kernel registry entries that are retained when multiple kernels for
  the same op and device pair are loaded.
2. Some shared object may only include slow kernels, and they may just be
  included in the distribution for compatibility. We can unload shared objects
  from memory if none of the kernels in it are useful.
3. Minimize the total size of the shared libraries created. Currently, tf
  framework is this big monolithic build rule everyone ends up depending on.
  Try to slim down the kernels, and get them to a size that makes sense to be
  included in tf lite packages.
4. Make sure there are only kernels in the given shared object. Error out if
  someone sneaks in ops in kernel libraries.
 ## Alternatives considered
 A number of alternatives have been considered before deciding on this route:
 1. Create and distribute the whole package with different compiler options.
  While this is the path of least resistance, the monolithic package that needs
  to be tested fully on different hardware and compiler options is becoming
  unmanageable. The simplest example is, we have a lot of code that needs to be
  tested with GPU compilers only once, but we end up having to run similar tests
  with 5+ different compiler options. Such issues drive up our testing costs in
  terrms of both resources, and developer time.
2. Splitting kernels into different binaries rather than different shared
  objects. While this will protect us from symbol collisions, ODR violations, or
  other classical headaches that plague shared objects, this will make things
  slower. Also, we would need to implement shared memory pages to share data
  across different processes, which will incur a similar engineering cost to the
  proposed approach. Therefore, we decided on using shared libraries instead.
