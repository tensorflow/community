# RFC: Enabling Determinism in TensorFlow
  
| Status        | Proposed                                             |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Pankaj Kanwar (Google), Reed Wanderman-Milne (Google)|
| **Sponsor**   | Sanjoy Das (Google)                                  |
| **Updated**   | 2021-01-31                                           |


## Objective
Allow users to enable deterministic behavior in TensorFlow. This means if the user runs a TensorFlow program multiple times, the model outputs and weights will be the same each time. Determinism will be supported on CPUs and GPUs.
 
To get deterministic behavior, users must do the following:

* Enable determinism using the API proposed in this doc.
* Use same hardware in every run.
* Use the same software environment every run (OS, checkpoints, version of TF, environmental variables, etc).
* Not use constructs outside TensorFlow that are nondeterministic, such as Python’s `random` module or using multiple threads/processes in ways that influence TensorFlow’s behavior.
* Do not use nondeterministic custom ops.

## Motivation
There are several mission critical applications in life sciences, finance and automation that require deterministic behavior. Determinism is required so that the behavior of these applications can be accurately predicted & demonstrated in a variety of scenarios.

Lack of determinism prevents companies from launching products using models developed in TF. For a subset of these industries having deterministic behavior is a regulatory requirement. 

In addition, determinism increases model velocity development by reducing noise, while also simplifying the debugging workflow.

## Design Proposal
We will create a new flag with the default value of ‘False’ which enables determinism.  We will define 2 functions:

* `tf.config.enable_deterministic_execution(enabled)`
* `tf.config.deterministic_execution_enabled()`

The first function takes in a boolean value, and allows the model developer to enable/disable determinism. The second function returns a bool indicating whether determinism is enabled.
In some cases, we have deterministic and nondeterministic versions of the kernel. In such cases, we will use this flag to run the appropriate kernels.
For ops which do not yet have a deterministic implementation, TensorFlow will raise a `tf.errors.UnimplementedError` if the flag is enabled.

Enabling deterministic execution does not automatically cause a user’s program to become deterministic. If users use nondeterministic constructs outside TensorFlow, such as threads/process, in ways that influence TensorFlow’s behavior, their program will not be deterministic. In order for a user to ensure their program is deterministic, users must both enable deterministic execution within TensorFlow and remove any sources of nondeterminism outside TensorFlow.

### Existing Flags
Multiple environmental variables exist today that control determinism. As part of this change, we will deprecate then remove the following:

* TF_DETERMINISTIC_OPS
* TF_CUDNN_DETERMINISTIC

tf.data also has flags for determinism. The system will throw an error message if flags are out of sync i.e. if deterministic_execution_enabled is enabled but if the tf.data option is set to ‘false’, we will throw an error. (`tf.data.Options.experimental_deterministic`). We’ll also add the necessary checks for Dataset.map and Dataset.interleave. See the [Random ops](#random-ops) section for how random Datasets, such as `tf.data.experimental.RandomDataset`, are handled.

### Grappler changes
Grappler graph optimizations may add nondeterministic behavior. In particular some optimizations will time out if they take too long to run. When determinism is enabled, these timeouts will be disabled.

### Random ops
Legacy random ops, such as `tf.random.normal`, are not deterministic if no seed is set, and so such ops will raise an error when determinism is enabled. To fix, the user should set a global seed with `tf.random.set_seed`. Since most models use legacy random ops, in practice users must call `tf.random.set_seed` when enabling deterministic behavior. Alternatively, users can pass a seed to every individual random operation, but doing so is more inconvenient.

Certain random ops, such as `tf.image.sample_distorted_bounding_box` and `tf.nn.fractional_max_pool`, ignore the global seed if a seed is not explicitly passed. For such ops, setting the global seed is not enough to avoid the error, so users must pass a seed directly to the op.

As for TensorFlow 2 random number generation, `tf.random.Generator.from_non_deterministic_state` will raise an error if called when determinism is enabled. In such cases, users should check if determinism is enabled and if so, use a different generator from a deterministic source. `tf.random.get_global_generator` implicitly calls `from_non_deterministic_state` if no global generator is set, and so will also raise an error if a global generator is not set with `tf.random.set_global_generator`.

Stateless random functions, such as `tf.random.stateless_normal`, are always deterministic and so will never raise determinism-related errors.

In graph mode, ops will raise an error message when the random op is created. If a random op is created in graph mode without determinism being enabled but then later runs when determinism is enabled, it will also raise an error.

No error will be raised if a random op or generator is run before determinism is enabled (as is true for any other op), so users should take care to enable determinism before running any random ops or generators.

### Parameter Server
Use of parameter servers adds nondeterministic behavior. In case a model constructs a ParameterServerStrategy, TensorFlow will throw an error. We’ll also document this in the documentation for the flag.

### Op Review and changes
As part of the implementation, we will review all ops to make a determination of their behavior (deterministic vs nondeterministic). Some of the ops that are known to be nondeterministic, at least when running on a GPU, include:

* `tf.nn.softmax_cross_entropy_with_logits`
* `tf.nn.sparse_softmax_cross_entropy_with_logits`
* `tf.image.resize` gradient with `method=ResizeMethod.NEAREST`
* `tf.math.segment_sum`, `tf.math.unsorted_segment_sum` forward
* `tf.image.crop_and_resize` gradient to both image and boxes 
* `tf.sparse.sparse_dense_matmul` forward
* `tf.math.unsorted_segment_mean`, `tf.math.unsorted_segment_prod` and `tf.math.unsorted_segment_sqrt`; all foward
* `tf.sparse.sparse_dense_matmul`


`tf.image.sample_distorted_bounding_box` has been observed to behave nondeterministically unless you set its seed parameter, even if you call tf.random.set_seed. We will review this Op as part the change. Another case that needs review is "pulling a random number from a PRNG before its state has been initialized".

Given the large number of ops involved, there is a chance that we might omit raising an error for a nondeterministic Op.
