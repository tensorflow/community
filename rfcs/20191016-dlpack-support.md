# dlpack support for interoperability with other GPU frameworks

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | 180 (https://github.com/tensorflow/community/pull/180) (update when you have community PR #)|
| **Author(s)** | eoldridge@nvidia.com, wmjlyjemaine@gmail.com, zhoujinjing09@gmail.com |
| **Sponsor**   | apassos@google.com, sanjoy@google.com                 |
| **Updated**   | 2019-11-26                                           |

## Objective

This document proposes the adoption of dlpack (https://github.com/dmlc/dlpack) as way of passing tensor data to other frameworks without leaving the GPU and without a copy per [24453](https://github.com/tensorflow/tensorflow/issues/24453).  dlpack is a community effort to define a common tensor data structure that can be shared by different frameworks. dlpack is currently supported by cuPy, cuDF, DGM, TGL, PyTorch, and MxNet. 

The interoperability of dlpack would allow for fast on-GPU communication between TensorFlow and these frameworks opening up a wide range of use cases outlined below.  It would further enable \_\_cuda_array_interface\_\_ interoperability through cuPy/cuDF which support both methods providing a way to transfer data to Numba, PyArrow and other frameworks that have adopted that method, although [a similar request has been made to support that method of interoperability](https://github.com/tensorflow/tensorflow/issues/29039) and ideally both would be supported.

A solution has already been developed by @VoVAllen and @jermainewang (coauthored above) as an external python package.  This RFC would see the concepts from the package integrated into Tensorflow Core, and reviewed and enhanced by the TF team so that dlpack support is native.

## Motivation

Why this is a valuable problem to solve? What background information is needed
to show how this design addresses the problem?

Which users are affected by the problem? Why is it a problem? What data supports
this? What related work exists?

DLPack is a community effort to define a common tensor data structure that can be shared by different frameworks allowing data to be quickly shared often with zero or minimal copy. One of the main bottlenecks when trying to achieve GPU performance when operating across different frameworks is I/O and data formatting.  The transfer of data between GPU and CPU or between formats is costly to the point where many operations become faster to simply run on the CPU because of the additional costs associated with moving/transforming the data.  Even when mechanisms exist to copy data without leaving the GPU, memory constraints limit the application because two copies of the data are required.  By implementing dlpack within TensorFlow there would be a way to transfer data directly between frameworks, enabling the development of a range of applications that weren't previously possible.

Existing applications that take advantage of dlpack include: (adding my own and those listed in , other contributions needed)
 - Inline on-gpu preprocessing of tabular data using cuDF to prepare it for deep learning models (continuous normalization, categorical encoding, etc) improving preprocessing performance by 10x over pandas and CPU
 - Larger than cpu memory dataloader that iterates over parquet files and batch loads tensors, providing a significant speedup over traditional dataloaders for tabular data
 - [End to end acceleration of training on GPU](https://medium.com/rapids-ai/accelerating-deep-learning-recommender-systems-by-15x-using-rapids-fastai-and-pytorch-b50b4d8568d1); 
 - Use of Tensorflow in conjunction with [tvm](https://github.com/dmlc/tvm); [TF custom op implementation of TVM](https://github.com/tobegit3hub/tftvm)
 - Use of Tensorflow in conjunction with [dgl](https://github.com/dmlc/dgl)
 - Zero copy transfer of data in [DALI](https://github.com/NVIDIA/DALI) reducing memory requirements.

Beyond the benefit of specific applications, Tensorflow's adoption of dlpack would further incentivize other frameworks considering its adoption as all three major DL frameworks would now be supporting it.  Finally, it would also make the development of applications that operate upstream and downstream of deep learning frameworks easier to develop as a single framework agnostic method could be used in conjunction all DL frameworks.  

## User Benefit

How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post?

Users who wish to utilize other GPU accelerated frameworks like cuDF, cuPy, etc would be able to do so without expensive copy operations.  By doing direct dataloading, feature engineering and preprocessing on GPU we see 10-15x speedups over traditional workflows involving CPUs to prepare the data for model readiness in other frameworks and they would be immediately available in tensorflow.

More generally, users would be able to develop preprocessing or other GPU based functionality and be able to support integration with all dl frameworks simplifying development efforts when creating solutions that are upstream or downstream from deep learning models.

A blog post or release notes headline could read "Tensorflow now supports dlpack enabling interoperability with other GPU powered frameworks like cuPy, cuDF, DGM, TGL, PyTorch, and MxNet."

## Design Proposal

Notes from @alextp:

AFAICT it should be easy to take cuda pointers in and out of TF and use them to build dlpack structures from tensors or vice versa. The tricky part is that TF does not use cudamalloc to allocate memory but its own allocator whose internal state is stored on the CPU and matches the head of TF's compute stream, so we need to sync TF's stream before the memory is usable from dlpack and similarly sync other cuda streams before memory is made usable by TF tensors (and similarly we need to sync the streams when trying to free the buffers).

A working version of dlpack integration has been released as a package by coauthors @jermainewang and @VoVAllen here:
https://github.com/VoVAllen/tf-dlpack/issues/3

This proposal would leverage that solution and integrate it into TF so that the operations could be performed natively.

User experience
We plan to release a python package tfdlpack, containing two APIs:
```
to_dlpack: Given a tensorflow tensor, return a DLPack tensor contain.
from_dlpack: Given a DLPack-compatible python capsule, return a tensorflow tensor.
```

Example code of converting a Tensorflow tensor to Torch tensor using DLPack using the package:
```python
import numpy as np
import tensorflow as tf
import torch.utils.dlpack as thdlpack
import tfdlpack

t1 = tf.constant([1, 2, 3], dtype=np.float32)
dlpack = tfdlpack.to_dlpack(t1)  # tf tensor -> dlpack
t2 = thdlpack.from_dlpack(dlpack)  # dlpack -> th tensor
print(t2)
dlpack = thdlpack.to_dlpack(t2)  # th tensor -> dlpack
t3 = tfdlpack.from_dlpack(dlpack)  # dlpack -> tf tensor
print(t3)
```
You will find that t1, t2 and t3 all have the same values, shape, and device contexts.
Package dependency: tensorflow>=2.0

Proposed code of converting a Tensorflow tensor to Torch tensor using DLPack natively:
```python
import numpy as np
import tensorflow as tf
import tensorflow.experimental.dlpack as tfdlpack
import torch.utils.dlpack as thdlpack


t1 = tf.constant([1, 2, 3], dtype=np.float32)
dlpack = tfdlpack.to_dlpack(t1)  # tf tensor -> dlpack
t2 = thdlpack.from_dlpack(dlpack)  # dlpack -> th tensor
print(t2)
dlpack = thdlpack.to_dlpack(t2)  # th tensor -> dlpack
t3 = tfdlpack.from_dlpack(dlpack)  # dlpack -> tf tensor
print(t3)
```

Proposed API implementation details:
There two critical parts for this API:
1. Memory usability on async device (to_dlpack)
As mentioned by @alextp
> TF does not use cudamalloc to allocate memory but its own allocator whose internal state is stored on the CPU and matches the head of TF's compute stream, so we need to sync TF's stream before the memory is usable from dlpack and similarly sync other cuda streams before memory is made usable by TF tensors (and similarly we need to sync the streams when trying to free the buffers).
Here we decide to manunally sync the device when exporting TF tensor to dlpack. The sync behavior is done in the `TFE_TensorHandleDevicePointer` API, which returns the pointer to the underlying memory.

2. Memory management (avoid leak) (to_dlpack/from_dlpack)
As the design of dlpack, the framework constructing tensor from dlpack is responsible to call the dlpack's deleter, which is usually dereferencing the underlying buffer, when destructing the constructed tensor. 
For `from_dlpack`, a deleter function is registered when constructing the TF tensor, and would be called upon destruction.
For `to_dlpack`, the dlpack data structure will hold a reference (by `TensorReference`) to the underlying buffer, and `unref` it in the dlpack's deleter function. 


## Questions and Discussion Topics

https://github.com/tensorflow/tensorflow/issues/29039#issuecomment-527520270 Outlines the key issues that need to be addressed, namely that a synch is required to ensure the tensor information is valid.  Supporting \_\_cuda_array_interface\_\_ is another option as well, although cuPy and cuDF have opted to support both and ideally Tensorflow would as well.

## Reference

### tfdlpack package implementation detail

The first design consideration is that we want to avoid any modification to the main Tensorflow library, so to get around the potential long delay of PR, code review, and release cycle of Tensorflow main package. Inspired by the solution from https://github.com/tobegit3hub/tftvm, we decide to implement the functionality as two custom tensor ops: to_dlpack and from_dlpack.

Besides, we want this feature to be plugged into other projects quite easily. For example, any project that relies on this feature is able to run without compiling against Tensorflow's header files. Not only that an extra dependency usually means extra effort, but also that such maintenance is repetitive and should be handled by the feature developer (i.e., us) alone. To this end, we have an idea of releasing it as a python package. However, the question is how to invoke the two custom tensor ops in python? The challenge is that Tensorflow's custom op interface has a limited support of argument and return types, while to_dlpack and from_dlpack should have an argument/return type of DLPack object. We work around this by encoding the address of an DLPack object as an integer, so it can be accepted/returned by the custom op interface. Then, we decode it in python or C depending on whether we return it (to_dlpack) or consume it (from_dlpack).

Finally, to achieve the maximal efficiency, we want the conversion happens without memory copy.

For to_dlpack, the returned DLPack tensor shares the same memory address of the input Tensorflow tensor and holds a reference to it. Upon the destruction of the DLPack tensor, it will dereference the Tensorflow tensor, so it can be collected by Tensorflow's memory management. (inspired by PyTorch's DLPack implementation).
For from_dlpack, it first creates an allocator object (subclass Tensorflow's allocator interface) that holds the reference to the DLPack tensor. The AllocateRaw function directly returns the memory it holds without creating any new buffer. Upon destruction, the DeallocateRaw function just calls the deletor of the DLPack tensor. (inspired by Tensorflow's immutable_constant_op).
