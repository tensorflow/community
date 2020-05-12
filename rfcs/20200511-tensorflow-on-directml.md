# TensorFlow on DirectML

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Chai Chaoweeraprasit (wchao@microsoft.com), Justin Stoecker (justoeck@microsoft.com), Adrian Tsai (adtsai@microsoft.com), Patrice Vignola (pavignol@microsoft.com) |
| **Sponsor**   | \[interim\] Alexandre Passos (apassos@google.com)  |
| **Updated**   | 2020-05-11                                           |

## Objective
Implement a new TensorFlow device type and a new set of kernels based on [DirectML](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-intro), a hardware-accelerated machine learning library on the DirectX 12 Compute platform. This change broadens the reach of TensorFlow beyond its existing GPU footprint and enables high-performance training and inferencing on Windows devices with any DirectX12-capable GPU.

## Motivation
TensorFlow training and inferencing on the GPU have so far been limited to Nvidia CUDA and AMD ROCm platform, with limited availability on ROCm only on Linux. Bringing the full machine learning training capability to Windows on any GPU has been one of the most requested features from the Windows developer community in our recent survey. By implementing a new set of TensorFlow kernels based on DirectML, Windows developers, professionals, and enthusiasts will be able to realize the full breadth of TensorFlow functionality on the vast hardware ecosystem of Windows devices, significantly expanding TensorFlow's availability on both the edge and the cloud.

## User Benefit
1. Users will be able to use TensorFlow on any DirectX12-capable GPU on any Windows device.
2. It works right out of the box. DirectML is an operating system component that works with any GPU on the device. The users do not need to go through the process of finding the right combination of CUDA/cuDNN runtime dependencies and graphics driver that works with a version of TensorFlow.
3. The size of our current TensorFlow-DirectML packages is roughly 20% of the current TensorFlow-GPU 1.15 package size of the comparable build. We expect our package size to double at launch, which will still be no bigger than half the current size. The smaller package size simplifies the requirements for the containers and other deployment mechanisms.
4. DirectML starts up considerably faster than the CUDA runtime. This improves the developer's experience in the model development process.

## Design Proposal
We propose:
1. A new device type called `DmlDevice` alongside the current `ThreadPoolDevice` and `GPUDevice`. This new device is a primary interface for executing kernel operations on DirectML and managing Direct3D resources for it.
2. An implementation for a new set of kernels based on DirectML. These new kernels are `OpKernel` instances that are registered, instantiated, and executed by the DirectML device (1) through its kernel manager interface. Each kernel calls into DirectML API to compute the output of the operation at model execution time. 
3. A new build and deployment PyPI package that combines the existing CPU kernels with the DirectML device (1) and kernels (2), as an alternative GPU package for TensorFlow users.

### Alternatives Considered
The vast majority of both the CPU and CUDA kernels are implemented using Eigen. This is only possible because of the single-source programming model, which mixes the host and the device code in a single file. It is also the same approach taken by ROCm. Only a handful of CUDA kernels are implemented directly using the cuDNN library. This model doesn't work for DirectCompute/HLSL as the HLSL provides a different level of abstraction for the underlying GPU programmable shaders. For this reason, the DirectML kernels will be using DirectML operators explicitly, either by forwarding the calls to the corresponding DirectML operators in simple cases, or composing the kernel operations as DirectML graphs for the more complex ones. See the [detailed design](#detailed-design) section for a sample kernel implementation.

### Performance Implications
There will be no performance implications for the existing TensorFlow's use cases through the existing TensorFlow's deployment packages. The new TensorFlow-DirectML package and its corresponding new device and kernels will have its own performance characteristics. There will be no interoperability between the existing CUDA kernels and the newly added DirectML kernels, as the two kernel implementations will not co-exist within the same package.

### Dependencies
The DirectML device and kernels take a dependency on the DirectML library and API. This dependency is specific to the new TensorFlow-DirectML package.

DirectML is a Windows system component, available in Windows 10 version 1903 (build 18362) or newer. It is also available as a stand-alone NuGet package that can be obtained through a public NuGet feed. TensorFlow can be configured to use with either the OS version or the stand-alone version of DirectML depending on the framework's need.

The current TensorFlow Python tests will be extended to support the execution of the DirectML device and kernels. It will, therefore, be affected by this additional dependency when running the tests against the new DirectML backend.

### Engineering Impact
The DirectML builds of TensorFlow are produced with a newly added Bazel build option `--config=dml` but without the existing `--config=cuda` option. A full build takes roughly 30 minutes to complete. Our current package sizes are 54 MB and 88 MB for the Windows and Linux wheel, compared to the current [TensorFlow-GPU 1.15](https://pypi.org/project/tensorflow-gpu/1.15.0/#files) wheels at 294 MB and 411.5 MB. Our final package size will most definitely be bigger than what it is today, as new kernels are added to the package. However, we believe our final size should still be significantly smaller than the current TensorFlow-GPU package size. This is because DirectML kernels are largely just a set of API calls to DirectML while CUDA kernels are templatized for every single data type that it supports.

Microsoft is committed to maintaining this code and its corresponding test collaterals as part of the TensorFlow 1.15 codebase. We augmented the existing Python tests to test against DirectML devices, as we hold the DirectML device to the same quality bar as to how the CUDA devices have been tested.

### Platforms and Environments
The DirectML device and kernels are fully supported in both the Windows and Linux build of TensorFlow.

XLA graph compilation will not be supported initially.

TPU support is out of scope for this change.

### Best Practices
This change is designed to be a drop-in replacement of the current TensorFlow-GPU 1.15 package with the following exceptions:
- Use cases taking direct dependencies on CUDA-specific kernels are not supported. No interoperability between the CUDA and DirectML kernels is allowed.
- Use cases requiring XLA compilation are not supported by the DirectML device and kernels.

### Tutorials and Examples
No change is required to activate the DirectML device in the Python script as the default device priority in the TensorFlow-DirectML build is the DirectML device. To explicitly activate the DirectML device, use `with tf.device("DML")`.

### Compatibility
The change in this proposal concerns the low-level constructs inside the TensorFlow runtime with minimal to no impact to the high-level exposures and API.

We expect existing models relying on hardware acceleration with CUDA to behave the same way on the same hardware with DirectML. The change has no impact on SavedModel and model conversions. The new TensorFlow-DirectML package will be available on PyPI.org and installable via the `pip install` command in the same way the existing TensorFlow packages are.

### User Impact
This change will be available to the users as a new TensorFlow-DirectML package available on PyPI.org, installable via the `pip install` command.

## Detailed Design
The following pseudo-code illustrates how a kernel for the TensorFlow Keras [BatchNormalization](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/layers/BatchNormalization) may be implemented with DirectML:

    auto x = dml::InputTensor(scope, 0, input_descs[0]);
    auto scale = dml::InputTensor(scope, 1, input_descs[1]);
    auto offset = dml::InputTensor(scope, 2, input_descs[2]);

    DML_TENSOR_DATA_TYPE input_type = x.GetOutputDesc().dataType;
    auto input_sizes = x.GetOutputDesc().sizes;

    // We normalize the input for each channel, so the nuMBer of elements per
    // normalization is N * H * W
    uint32_t norm_elem_count = input_sizes[0] * input_sizes[2] * input_sizes[3];

    // Compute the mean of the input for each channel. We do this with an
    // AVERAGE reduction across all axes except C.
    auto mean = dml::Reduce(x, DML_REDUCE_FUNCTION_AVERAGE, {0, 2, 3});

    // The strides we need to set to broadcast C across an entire tensor
    dml::TensorDesc::Dimensions broadcast_c_strides = {/*N*/0, /*C*/1, /*H*/0, /*W*/0};

    // Broadcasts the C dimension across the entire input tensor
    auto broadcasted_mean = dml::Reinterpret(mean, input_sizes, broadcast_c_strides);

    // Compute the variance of the input for each channel.
    auto x_centered = x - broadcasted_mean;
    auto variance = dml::Reduce(x_centered, DML_REDUCE_FUNCTION_SUM_SQUARE, {0, 2, 3});
    variance /= norm_elem_count;

    // Given the mean/variance, use DirectML's built-in BatchNorm operator to compute the final output.
    const bool is_spatial = true;
    auto normalized_output = 
        dml::BatchNormalization(x, mean, variance, scale, offset, is_spatial, epsilon);

## Questions and Discussion Topics
- What is the long-term goal for XLA in TensorFlow?
- What is the path to integrate this change to TensorFlow 2.0? What about TensorFlow Runtime?
- How do MLIR and its development affect the future direction of TensorFlow?