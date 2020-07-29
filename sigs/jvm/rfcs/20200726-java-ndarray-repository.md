# Java NdArray Repository
| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Karl Lessard (karl.lessard@gmail.com) |
| **Sponsored** | Joana Carrasqueira (joanafilipa@google.com), Rajagopal Ananthanarayanan (ananthr@google.com) |
| **Updated**   | 2020-07-26                                          |

## Objective

Create a new repository under the TensorFlow GitHub organization to host the code of the NdArray Java library used by TensorFlow and 
maintained by SIG JVM.

## Motivation

SIG JVM has developed a library called [`NdArray`](https://github.com/tensorflow/java/tree/master/ndarray) to improve the 
support of n-dimensional data structures on the JVM (e.g. tensors in TensorFlow).

While this library is intensively used by the TensorFlow Java bindings, it does not have any dependency to the 
TensorFlow runtime itself, making it an interesting option for any Java developer interested in fast and efficient I/O accesses 
to continuous native memory in a n-dimensional space.

To promote the adoption of this library beyond the scope of TensorFlow users, SIG JVM would like to move it into a distinct 
GitHub repository.

## User Benefit

N-dimensional memory access in Java is poorly supported by the JDK and while there are already a few initiatives involving different 
parties to standardize such API, it will take time before it gets released to the public. The NdArray Java Library is now a good option for 
all kind of Java projects and since it does not depend on the TensorFlow runtime, it is very lightweight and portable.

Also, having a distinct repository for the NdArray Java Library development will simplify the task for its collaborators as they will not need
anymore to checkout and build the whole TensorFlow Java project, a process which tends to be heavy since it also runs a Bazel build 
of the TensorFlow core sources that can take many hours.

## Design Proposal

The current request is for the creation of the following repository: `/tensorflow/java-ndarray`

This new repository will only host the code of the NdArray Java Library, which will be moved out of `tensorflow/java`. 
Everything else related to TensorFlow Java depends on the TensorFlow runtime and shall remain under their actual
repositories.

The NdArray Java Library will now have its own release cycle and group of collaborators, independently from the TensorFlow Java bindings, while
remaining under the jurisdiction of SIG JVM.
