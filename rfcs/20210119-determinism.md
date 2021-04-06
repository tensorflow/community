# RFC: Enabling Determinism in TensorFlow
  
| Status        | Accepted                                                                    |
:---------------|:-----------------------------------------------------------------------------|
| **Author(s)** | Pankaj Kanwar (Google), Duncan Riach (NVIDIA), Reed Wanderman-Milne (Google) |
| **Sponsor**   | Sanjoy Das (Google)                                                          |
| **Updated**   | 2021-03-12                                                                   |

## Objective

Add an API which will make all ops deterministic. This means if a user runs an op multiple times with the same inputs, the outputs will be the same each time. Users can use this API to create deterministic models.

## Motivation

There are several mission critical applications in medicine, finance and automation that require deterministic behavior. Determinism is required so that the behavior of these applications can be accurately predicted & demonstrated in a variety of scenarios.

The lack of determinism in certain ops prevents companies from launching products using models developed in TF. For a subset of these industries having deterministic behavior is a regulatory requirement.

In addition, deterministic functionality, enabled by deterministic ops, increases model velocity development by reducing noise, while also simplifying the debugging workflow.

## Design Proposal

We will create a new flag with the default value of "False" which enables deterministic ops.  We will define 2 functions:

* `tf.config.enable_deterministic_ops(enabled)`
* `tf.config.deterministic_ops_enabled()`

The first function takes in a boolean value, and allows the model developer to enable/disable deterministic ops. The second function returns a bool indicating whether deterministic ops is enabled.

Once enabled, every built-in op will either be made deterministic or raise an error if determinism is not supported. A `tf.errors.UnimplementedError` will be raised by ops for which we have not yet implemented a deterministic version. In the long term, we plan on adding a deterministic version to all such ops. For ops which are inherently nondeterministic such as `tf.random.normal` without a seed, a `tf.errors.FailedPreconditionError` will be raised (the precondition being that determinism must be disabled). Some ops will only raise an error on a subset of input shapes, attributes, data types, or codepaths through the op. Depending on the op, in graph mode, the error will either be raised when the op is constructed or when the op is run.

By "deterministic", we mean that if an op is run multiple times with the same inputs and attributes, it produces the same outputs. The op must be run with the same hardware configuration on the same device each time. The software environment must be the same every run as well (OS, TF and CUDA version, environmental variables, etc). For stateful ops, the all relevant state must be identical each run (values of `tf.Variable`s, checkpoints, etc).

In most cases, ops will be unconditionally deterministic, but a few will have a separate deterministic and nondeterministic codepath when the nondeterministic codepath is faster.

This API only makes ops deterministic, not other parts of TensorFlow. For example, `ParameterServerStrategy` will not be made deterministic or raise an error when deterministic ops is enabled. The reason the API only affects ops is that TensorFlow has a large number of components, and it is infeasible to find all sources of nondeterminism, fix them, and ensure no TensorFlow developer ever accidentally introduces nondeterminism again. Currently, the only known sources of nondeterminism outside ops in TensorFlow 2 is `ParameterServerStrategy` and Grappler. In TensorFlow 1, sessions are additionally nondeterministic. Separately from the `enable_deterministic_ops` API, we plan on making Grappler deterministic, which is described in the "Grappler changes" section.

The API allows users to write deterministic models. To do so, users must:

* Enable deterministic ops with `tf.config.enable_deterministic_ops`.
* Use the same hardware configuration in every run.
* Use the same software environment in every run (OS, checkpoints, version of CUDA and TF, environmental variables, etc).
* Not use nondeterministic parts of TensorFlow (besides ops), such as `ParameterServerStrategy`.
* Not use constructs outside TensorFlow that are nondeterministic, such as Python’s `random` module (without a fixed seed) or using multiple threads/processes in ways that influence TensorFlow’s behavior.
* Not use nondeterministic custom ops.

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

Calling `tf.config.enable_deterministic_ops(True)` will be equivalent to setting `TF_DETERMINISTIC_OPS` to `'true'` or `'1'` plus additionally making all other ops deterministic. In the short term, we will continue making more ops deterministic with `TF_DETERMINISTIC_OPS`, so that we can implement and test determinism to ensure it works fully before introducing the `enable_deterministic_ops` API. Once the `enable_deterministic_ops` API is implemented, the two environment variables will be first deprecated and then removed.

tf.data also has flags for determinism. The system will throw an error message if flags are out of sync i.e. if deterministic_execution_enabled is enabled but if the tf.data option is set to ‘false’, we will throw an error. (`tf.data.Options.experimental_deterministic`). We’ll also add the necessary checks for Dataset.map and Dataset.interleave. See the [Random ops](#random-ops) section for how random Datasets, such as `tf.data.experimental.RandomDataset`, are handled.

### Grappler changes

Grappler graph optimizations may add nondeterministic behavior. In particular, some optimizations will time out if they take too long to run. Grappler will not be affected by the `enable_deterministic_ops` API, as the API only affects ops. However, the purpose of the API is to allow users to write deterministic models, and this is impossible if Grappler is nondeterministic. Therefore, we will either make Grappler deterministic by default, or if that is infeasible, add a separate flag to make Grappler deterministic.

To make Grappler deterministic by default, all timeouts must be removed. Typically, a Grappler pass will check if the timeout is exceeded once or several times per iteration of a loop. These timeouts can be replaced with a limit in the number of iterations the loop can execute for. This approximates a timeout while being deterministic.

If removing timeouts is infeasible, an option will be added to disable timeouts, which will be specified with the existing [`tf.config.optimizer.set_experimental_options`](https://www.tensorflow.org/api_docs/python/tf/config/optimizer/set_experimental_options) function. The timeout can already be disabled with a [RewriteConfig](https://github.com/tensorflow/tensorflow/blob/07d3ea8f99bd838e7e47c248babfe21372cbe62a/tensorflow/core/protobuf/rewriter_config.proto#L173) option, but the RewriteConfig is not exposed in TensorFlow 2. To create deterministic models, users must both must call `enable_deterministic_ops` and disable the Grappler timeout. A warning will be issued if deterministic ops is enabled but the Grappler timeout has not been disabled.

It is also possible Grappler is nondeterministic due to nondeterministic iteration order of certain sets, especially sets of graph nodes. We will investigate and fix such cases of nondeterminism. Users will not be required to set the `PYTHONHASHSEED` environmental variable.

### Random ops

Legacy random ops, such as `tf.random.normal`, are not deterministic if no seed is set, and so such ops will raise a `tf.errors.FailedPreconditionError` when determinism is enabled. To fix, the user should set a global seed with `tf.random.set_seed`. Since most models use legacy random ops (for variable initialization and various other uses), in practice users must call `tf.random.set_seed` when enabling deterministic ops. Alternatively, users can pass a seed to every individual random operation, but doing so is more inconvenient.

Certain random ops, such as `tf.image.sample_distorted_bounding_box` and `tf.nn.fractional_max_pool`, ignore the global seed if a seed is not explicitly passed. For such ops, setting the global seed is not enough to avoid the error, so users must pass a seed directly to the op.

As for TensorFlow 2 random number generation, `tf.random.Generator.from_non_deterministic_state` will raise an error if called when determinism is enabled. In such cases, users should check if determinism is enabled and if so, use a different generator from a deterministic source. `tf.random.get_global_generator` implicitly calls `from_non_deterministic_state` if no global generator is set, and so will also raise an error if a global generator is not set with `tf.random.set_global_generator`.

Stateless random functions, such as `tf.random.stateless_normal`, are always deterministic and so will never raise determinism-related errors.

No error will be raised if a random op or generator is run before determinism is enabled (as is true for any other op), so users should take care to enable determinism before running any random ops or generators.

### Testing plan

We must ensure that every op will either run deterministically or raise an error if `enable_deterministic_ops` has been called. In order to do this, we must test for determinism, both to check we don't accidentally miss any nondeterministic ops and to ensure any previously-deterministic ops are not accidentally made nondeterministic in the future. To accomplish this, we will have the following tests:

1. We will add tests to several of the [official models](https://github.com/tensorflow/models/tree/master/official) to ensure they run deterministically. In particular, each test will train a model for several steps, then retrain it from scratch several times. The final weights after training will be asserted to be the same each time. This tests not only the `enable_deterministic_ops` API but that the entire model is deterministic. This only tests ops that the official models use.

2. We will add a special mode to TensorFlow where every time a non-stateful op is run, TensorFlow will rerun the op several times and assert the outputs are the same each time. We will then run the TensorFlow unit tests with this mode as part of the nightly tests. Doing so ensures that for each op that is run as part of a unit test, it will be tested for determinism.

3. When adding determinism to an op which previously was nondeterministic, an explicit unit test will be added that checks for determinism. Unlike running unit tests with the special mode above, the explicit unit tests can be part of the presubmit tests instead of the nightly tests, and can test on inputs that are very likely to demonstrate nondeterminism if it exists.

### Op Review and changes

As part of the implementation, we will review all ops to make a determination of their behavior (deterministic vs nondeterministic). Ops that are known to operate nondeterministically, at least when running on a GPU, include the following:

* `tfa.image.dense_image_warp` gradient w.r.t `image`
* `tf.compat.v1.nn.fused_batch_norm` gradient w.r.t. `offset`
* `tf.convert_to_tensor` forward, when `value` is of type `tf.IndexedSlices`
* `tf.gather` gradient w.r.t dense `params`
* `tf.image.resize` gradient w.r.t `image`, when `method=ResizeMethod.NEAREST`
* `tf.image.crop_and_resize` gradient w.r.t both `image` and `boxes`
* `tf.image.adjust_contrast` forward
* `tf.math_segment_prod` forward
* `tf.math.segment_sum` forward
* `tf.math.unsorted_segment_mean` forward
* `tf.math.unsorted_segment_sqrt_n` forward
* `tf.math.unsorted_segment_prod` forward
* `tf.math.unsorted_segment_sum` forward
* `tf.nn.softmax_cross_entropy_with_logits` forward
* `tf.nn.sparse_softmax_cross_entropy_with_logits` forward
* `tf.sparse.sparse_dense_matmul` forward

We have a list of other ops that use CUDA's `atomicAdd` and are therefore likely to be sources of nondeterminism. Once it has been confirmed that those ops function nondeterministically, they will be made to throw errors when determinism is enabled. In the long term, we can add a deterministic implementation to such ops.

Given the large number of ops involved, there is a chance that we might omit raising an error for a nondeterministic op. While we plan on testing every op (See section ["Testing plan"](#testing-plan)), we will likely miss several ops which are only nondeterministic in practice on large inputs. We will fix such cases as they arise.

### Alternatives considered

#### Making all of TensorFlow deterministic, instead of just ops

This RFC proposes a function which makes each TensorFlow op deterministic. Alternatively, the function could make all of TensorFlow deterministic. The [original version of this RFC](https://github.com/tensorflow/community/blob/b3a8cd8a190daa56d20b0eecdb1efaa91b237eb8/rfcs/20210119-determinism.md) proposed this, but it was later changed to affect only ops.

The reason the RFC was changed to only affect ops is that determining every TensorFlow component which is current nondeterministic is infeasible. There is no effective way to find every nondeterministic part of TensorFlow. Additionally, it is likely developers would accidentally introduce nondeterminism back into TensorFlow, especially those who are unaware of the determinism API. Limiting the scope of the API to ops means we only have to worry about a fraction of the TensorFlow codebase, and additionally makes determinism far easier to test. It is feasible to test almost every op for determinism, but doing so for the entirety of TensorFlow component is impossible.

In the future, if we do decide to have an API which makes all of TensorFlow deterministic, we can add one. The advantage of making every part of TensorFlow either deterministic or raise an error is that users interested in determinism typically want their entire model to be deterministic. A model is nondeterministic if any part of it is nondeterministic, so we cannot guarentee determinism simply by making each op deterministic.
