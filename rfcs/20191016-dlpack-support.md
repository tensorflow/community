# dlpack support for interoperability with other GPU frameworks

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | eoldridge@nvidia.com, AN Other (you@example.org) |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                 |
| **Updated**   | 2019-10-16                                           |

## Objective

This document proposes the adoption of dlpack (https://github.com/dmlc/dlpack) as way of passing tensor data to other frameworks without leaving the GPU and without a copy.  dlpack is a community effort to define a common tensor data structure that can be shared by different frameworks. dlpack is currently supported by cuPy, cuDF, DGM, TGL, PyTorch, MxNet. 

The interoperability of dlpack would allow for fast on-GPU communication between TensorFlow and these frameworks opening up a wide range of use cases outlined below.  It would further enable \_\_cuda_array_interface\_\_ interoperability through cuPy/cuDF which support both methods providing a way to transfer data to Numba, PyArrow and other frameworks that have adopted that method.

## Motivation

Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?

Which users are affected by the problem? Why is it a problem? What data supports
this? What related work exists?

DLPack is a community effort to define a common tensor data structure that can be shared by different frameworks allowing data to be quickly shared often with zero or minimal copy. One of the main bottlenecks when trying to achieve GPU performance is I/O.  The transfer of data between GPU and CPU or between formats is costly to the point where many operations become faster to simply run on the CPU because of the additional costs associated with moving the data.  Even when mechanisms exist to copy data directly on GPU, memory constraints limit the application because two copies of the data are required.  By implementing dlpack within TensorFlow there would be a way to transfer data directly between frameworks, enabling the development of a range of applications that weren't previously possible but are currently available in PyTorch and MxNet.

Applications that take advantage of dlpack include: (adding my own, other contributions needed)
 - Inline on-gpu preprocessing of tabular data using cuDF to prepare it for the model (normalization, categorical encoding, etc) improving preprocessing performance by 10x
 - Larger than cpu memory dataloader that iterates over parquet files and batch loads tensors, providing a significant speedup over traditional dataloaders for tabular data
 - End to end acceleration of training on GPU (https://medium.com/rapids-ai/accelerating-deep-learning-recommender-systems-by-15x-using-rapids-fastai-and-pytorch-b50b4d8568d1)

Beyond the application benefit, Tensorflow's adoption of dlpack would further incentivize other frameworks considering its adoption as all three major DL frameworks would now be supporting it.  That would improve the availability of other frameworks for the Tensorflow userbase.

## User Benefit

How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post?

Users who wish to utilize other GPU accelerated frameworks like cuDF, cuPy, etc would be able to do so without expensive copy operations.  By doing direct dataloading, feature engineering and preprocessing on GPU we see 10-15x speedups over traditional workflows involving CPUs to prepare the data for model readiness.  

## Design Proposal

This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Factors to consider include:

* performance implications
* dependencies
* maintenance
* platforms and environments impacted (e.g. hardware, cloud, other software
  ecosystems)
* [compatibility](https://www.tensorflow.org/programmers_guide/version_compat)
* how will this change impact users, and how will that be managed?

The TensorFlow Tensor object provide API to get the array data of tensor.

input_tensor.flat<T>().data()
Can we get the data from the above function and use to initialize the DLTensor like this?

  DLTensor* x;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;

  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &x);
  static_cast<float*>(x->data) = data;

Workflow proposed by @futurely

Tensor constructs Buffer with Allocator. Buffer calls TypedAllocator::Allocate to allocate memory through Allocator::AllocateRaw. GPUcudaMallocAllocator::AllocateRaw calls cudaMalloc.

Maybe cuda.synchronize related API should be added to every class involved in the above pipeline.

Runtime uses DeviceFactory to CreateDevices. BaseGPUDeviceFactory::CreateGPUDevice allocates GPUDevice containing GPUAllocator. GPUProcessState::GetGPUAllocator composes AllocatorParts including GPUcudaMallocAllocator.

GpuExecutor::Allocate is a simple wrapper of GpuDriver::DeviceAllocate which utilizes cuMemAlloc.

TensorFlow has added conversion between CPU Tensor and numpy array.

## Questions and Discussion Topics

https://github.com/tensorflow/tensorflow/issues/29039#issuecomment-527520270 Outlines the key issues that need to be addressed, namely that a synch is required to ensure the tensor information is valid.  Supporting \_\_cuda_array_interface\_\_ is another option as well, although cuPy and cuDF have opted to support both.
