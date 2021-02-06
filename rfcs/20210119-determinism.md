# RFC: Enabling Determinism in TensorFlow
  
| Status        | Proposed                                                                    |
:---------------|:----------------------------------------------------------------------------|
| **Author(s)** | Pankaj Kanwar (Google),Reed Wanderman-Milne (Google), Duncan Riach (NVIDIA) |
| **Sponsor**   | Sanjoy Das (Google)                                                         |
| **Updated**   | 2021-01-31                                                                  |

## Objective
Allow users to enable deterministic behavior in TensorFlow. This means if the user runs a TensorFlow program multiple times, the model outputs and weights will be the same each time. Determinism will be supported on CPUs and GPUs.

To get deterministic behavior, users must do the following:

* Enable determinism using the API proposed in this doc.
* Use same hardware configuration in every run.
* Use the same software environment every run (OS, checkpoints, version of CUDA and TF, environmental variables, etc).
* Not use constructs outside TensorFlow that are nondeterministic, such as Python’s `random` module (without a fixed seed) or using multiple threads/processes in ways that influence TensorFlow’s behavior.
* Not use nondeterministic custom ops.

## Motivation
There are several mission critical applications in medicine, finance and automation that require deterministic behavior. Determinism is required so that the behavior of these applications can be accurately predicted & demonstrated in a variety of scenarios.

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
There are currently two environment variables in TensorFlow to enable deterministic op functionality.

The first environment variable is `TF_CUDNN_DETERMINISTIC`. When set to `'true'` or `'1'`, this,

* makes the selection of cuDNN convolution algorithms deterministic,
* selects deterministic gradient algorithms for `tf.nn.conv*d` and `tf.keras.layers.Conv*D`,
* selects deterministic gradient algorithms for `tf.nn.max_pool*d` and `tf.keras.layers.MaxPool*D`, and
* selects a deterministic gradient algorithm for `tf.nn.ctc_loss`.

The second environment variable is `TF_DETERMINISTIC_OPS`. This supercedes and replaces `TF_CUDNN_DETERMINISTIC` by having the same functionality and also (when set to `'true'` or `'1'`),

* selects deterministic gradient kernels for `tf.nn.bias_add` and the many Keras layers that apply a bias,
* selects a deterministic algorithm for XLA reductions on GPU, and
* selects a deterministic gradient algorithm for `tf.image.resize` with `method=ResizeMethod.BILINEAR` and `tf.keras.layers.UpSampling2D` with `interpolation='bilinear'`

Calling `tf.config.enable_deterministic_execution(True)` will be equivalent to setting `TF_DETERMINISTIC_OPS` to `'true'` or `'1'` plus the additional functionality described in this RFC.

The two environment variables will be first deprecated and then removed.

tf.data also has flags for determinism. The system will throw an error message if flags are out of sync i.e. if deterministic_execution_enabled is enabled but if the tf.data option is set to ‘false’, we will throw an error. (`tf.data.Options.experimental_deterministic`). We’ll also add the necessary checks for Dataset.map and Dataset.interleave. See the [Random ops](#random-ops) section for how random Datasets, such as `tf.data.experimental.RandomDataset`, are handled.

### Grappler changes
Grappler graph optimizations may add nondeterministic behavior. In particular some optimizations will time out if they take too long to run. When determinism is enabled, these timeouts will be disabled.

### Random ops
Legacy random ops, such as `tf.random.normal`, are not deterministic if no seed is set, and so such ops will raise an error when determinism is enabled. To fix, the user should set a global seed with `tf.random.set_seed`. Since most models use legacy random ops (for variable initialization and various other uses), in practice users must call `tf.random.set_seed` when enabling deterministic behavior. Alternatively, users can pass a seed to every individual random operation, but doing so is more inconvenient.

Certain random ops, such as `tf.image.sample_distorted_bounding_box` and `tf.nn.fractional_max_pool`, ignore the global seed if a seed is not explicitly passed. For such ops, setting the global seed is not enough to avoid the error, so users must pass a seed directly to the op.

As for TensorFlow 2 random number generation, `tf.random.Generator.from_non_deterministic_state` will raise an error if called when determinism is enabled. In such cases, users should check if determinism is enabled and if so, use a different generator from a deterministic source. `tf.random.get_global_generator` implicitly calls `from_non_deterministic_state` if no global generator is set, and so will also raise an error if a global generator is not set with `tf.random.set_global_generator`.

Stateless random functions, such as `tf.random.stateless_normal`, are always deterministic and so will never raise determinism-related errors.

In graph mode, ops will raise an error message when the random op is created. If a random op is created in graph mode without determinism being enabled but then later runs when determinism is enabled, it will also raise an error.

No error will be raised if a random op or generator is run before determinism is enabled (as is true for any other op), so users should take care to enable determinism before running any random ops or generators.

### Parameter Server
Use of parameter servers adds nondeterministic behavior. In case a model constructs a ParameterServerStrategy, TensorFlow will throw an error. We’ll also document this in the documentation for the flag.

### Op Review and changes
As part of the implementation, we will review all ops to make a determination of their behavior (deterministic vs nondeterministic). Ops that are known to operate nondeterministically, at least when running on a GPU, include the following:

* `tf.nn.softmax_cross_entropy_with_logits`
* `tf.nn.sparse_softmax_cross_entropy_with_logits`
* `tf.image.resize` gradient with `method=ResizeMethod.NEAREST`
* `tf.math.segment_sum`, `tf.math.unsorted_segment_sum` forward
* `tf.image.crop_and_resize` gradient to both image and boxes 
* `tf.math.unsorted_segment_mean`, `tf.math.unsorted_segment_prod` and `tf.math.unsorted_segment_sqrt`; all foward
* `tf.sparse.sparse_dense_matmul`

We have a list of other ops that use CUDA's `atomicAdd` and are therefore likely to be sources of nondeterminism. Once it has been confirmed that those ops function nondeterministically, they will be made to throw errors when determinism is enabled. In the long term, we can add a deterministic implementation to such ops.

Given the large number of ops involved, there is a chance that we might omit raising an error for a nondeterministic Op. We will fix such cases as they arise.

## Discussion

This section describes some topics that were discussed during the design review on February 4, 2021. The RFC was not yet approved due to the concerns described in the next section.

### CPU Support

If we implement determinism, we do not want to leave it half completed due to a lack of time to work on it. As CPU support may be time consuming, we could start by having the API only supported for GPU and TPU users. However, GPU models may still use some CPU ops. For example, all `tf.data` ops only support the CPU, and most int32 ops run on the CPU, [even for GPU kernels](https://cs.opensource.google/tensorflow/tensorflow/+/master:tensorflow/core/kernels/identity_op.cc;l=94;drc=3cbb50768909c585d33e99ba10172d1c44c04d6f). To support GPU models, we must also support determinism in these CPU ops. To avoid having to modify most CPU ops to raise a determinism error, we can have an allowlist of allowed CPU ops which are deterministic.

However, supporting a subset of CPU ops is problematic: what if a rewrite pass converts Op A (which supports determinism) to Op B (which does not). Similarly, Placer may be modified in the future to place some small ops on the CPU by default instead of the GPU, which can break determinism. In general, modifications to TensorFlow that affect ops can potentially break determinism, and therefore break backwards compatibility.

We don’t want TensorFlow developers to have to worry about breaking determinism when modifying TensorFlow. We could potentially allow a model to start raising a determinism error in minor releases of TF, but this is a bad user experience. Alternatively, we could rely on unit tests to catch cases where developers break determinism. Another alternative is to fully support determinism on the CPU. I and others will try to think of other ways to avoid developers inadvertently breaking determinism when modifying TensorFlow.

### Other points of discussion

* Performance is important even when determinism is enabled. We should ensure determinism is fast in the long term, although not necessarily as fast as nondeterminism. In the short term, it’s acceptable for it to be slow.
* The current semantics of tf.data’s `experimental_deterministic` option may be unacceptable slow, as it involves reading from input files in a serial order instead of in parallel. The semantics should be changed to allow files to be read in parallel, while still guaranteeing the order of items read from the files is deterministic
* Perhaps `enable_deterministic_execution` should take no arguments, and instead a `disable_deterministic_execution` function should be added. Should be consistent with other functions which we can also change, such as `enable_tensor_float_32_execution`.
* Sessions are nondeterminism and making them determinism requires having the executor run ops in a consistent order. It is probably not worth making sessions deterministic.
* If performant, we could potentially have determinism be enabled by default, but not raising an error for nondeterministic ops.
