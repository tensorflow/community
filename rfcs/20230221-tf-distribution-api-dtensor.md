# A new TensorFlow Distribution API with DTensor

Status                    | (Proposed / Accepted / Implemented / Obsolete)
:------------------------ | :---------------------------------------------
**RFC #**                 | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)
**Author(s)**             | Yu Feng(feyu@google.com), Bruce Fontain (bfontain@google.com) Yuefeng Zhou(yuefengz@google.com) Scott
Zhu (scottzhu@google.com) |
**Sponsor**               | A N Expert (whomever@tensorflow.org)
**Updated**               | 2023-02-21

## Objective

The new TensorFlow Distribution API extends the TensorFlow Core API, such as
tf.Variable, tf.function, and tf.Module with distributed tensor computation.

The low level component of the Distribution API, built with DTensor, provides a
uniform SPMD semantics across CPU, GPU, and TPU device types. DTensor is an
intrinsic part of TensorFlow that defines a representation of distributed
Tensors with Mesh and Layout data structures. Users and high level libraries
(such as Keras) can depend on Mesh and Layout just as other components of the
TensorFlow low level API. An initial experimental implementation is covered here
(on TensorFlow.org): DTensor Concepts; DTensor ML Tutorial; DTensor Keras
Tutorial

This RFC defines the integration between TensorFlow and DTensor, the low level
of TensorFlow's Next generation Distribution API. DTensor defines a uniform and
generic API for composing distributed TensorFlow programs for accelerator types
supported by TensorFlow. Common distribution patterns in machine learning,
including data and model parallelism, spatial partitioning, and pipelining can
all be expressed with primitives offered in this RFC. A very basic form of
interoperability with other ML frameworks, such as JAX is also supported in the
API described in this RFC. This document also demonstrates a potential path for
integration with the Keras modeling primitives in the form of DTensorStrategy, a
new subclass of `tf.distribute.Strategy`.

## Disclaimer

This is a draft RFC, API endpoints in this document are not final. This document
may refer to but does not cover the following topics for which we expect follow
on RFCs:

*   Keras integration

*   Saved Model/C++ API

*   Automatic sharding

*   Mixed use of DTensor and non-DTensor execution in tf.functions

*   tf.data integration

*   Representation of DTensor primitives in the TF Graph

## Glossary

Distributed Tensor: A Tensor that is distributed to multiple devices. A
Distributed Tensor can be viewed as a single Tensor, in the 'Global
Perspective'; or in the 'Local Perspective', as a collection of Tensors, one per
device.

The 'Global Tensor Perspective' to the 'Local Tensor Perspective' are related by
Layout and Mesh, which are defined below. A distributed Tensor can be 'shard',
'replicated', or a hybrid of both, although sometimes the terminology 'shard' is
used interchangeably with 'distribute'.

*   **Mesh**: A cartesian grid of devices. The dimensions of the grid are named.
    The following example is a 2 by 3 mesh on 6 CPU devices.

```python
mesh = tf.dtensor.Mesh({'x': 2, 'y' : 3}, devices=[
    'CPU:0', 'CPU:1', 'CPU:2',
    'CPU:3', 'CPU:4', 'CPU:5'])
```

*   **Layout**: A list of sharding specifications. A sharding specification is a
    string that controls how an axis of Tensor is distributed. The axis can be
    either sharded to a mesh dimension, or replicated. A sharded axis is always
    associated with a mesh dimension. The Global Perspective Tensor is evenly
    split along all of the sharded axes. Each segment of the split becomes a
    Local Perspective Tensor component. A replicated axis is not associated with
    a mesh dimension. The axis is present in all of the Local Perspective Tensor
    components. For example, on the 2 by 3 mesh defined above,

```python
tf.Tensor(shape=[5, 4, 6],
       layout=tf.dtensor.Layout([tf.dtensor.Mesh.UNSHARDED, 'x', 'y'], mesh))
```

The Global Perspective Tensor has a shape of `[5, 4, 6]`. The 6 components in the
Local Perspective all have the shape of `[5=5, 2=4//2, 2=6//3]`.

### Runtime Architectures:

The Distribution API supports the following runtime architectures:

*   Local mode: 1 local Process with multiple (local) accelerator devices.

*   Multi-client Distributed mode: N Processes each with one or more local
    accelerator devices, running identical programs. The user application runs
    as the N processes, which are also called clients. A typical example of a
    multi-client mode is a multinode system, where one or more processes are
    launched per machine (node).

*   Single-controller Distributed mode: 1 Process as the Controller + N generic
    Processes as Workers. Each worker has one or more local accelerator devices.
    The user application runs as the controller that coordinates the workers. A
    typical example of a single-controller mode is to deploy the controller to
    an external machine with a slower connection to a cluster that runs N of the
    workers.

## Design Proposal

### Dependencies

*   Dependencies: does this proposal add any new dependencies to TensorFlow?
*   Dependent projects: are there other areas of TensorFlow or things that use
    TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have
    you identified these dependencies and are you sure they are complete? If
    there are dependencies, how are you managing those changes?

### Engineering Impact

*   Do you expect changes to binary size / startup time / build time / test
    times?
*   Who will maintain this code? Is this code in its own buildable unit? Can
    this code be tested in its own? Is visibility suitably restricted to only a
    small API surface for others to use?

### Platforms and Environments

*   Platforms: does this work on all platforms supported by TensorFlow? If not,
    why is that ok? Will it work on embedded/mobile? Does it impact automatic
    code generation or mobile stripping tooling? Will it work with
    transformation tools?
*   Execution environments (Cloud services, accelerator hardware): what impact
    do you expect and how will you confirm?

### Best Practices

*   Does this proposal change best practices for some aspect of using/developing
    TensorFlow? How will these changes be communicated/enforced?

### Compatibility

*   Does the design conform to the backwards & forwards compatibility
    [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
*   How will this proposal interact with other parts of the TensorFlow
    Ecosystem?
    -   How will it work with TFLite?
    -   How will it work with distribution strategies?
    -   How will it interact with tf.function?
    -   Will this work on GPU/TPU?
    -   How will it serialize to a SavedModel?

### User Impact

*   What are the user-facing changes? How will this feature be rolled out?

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
