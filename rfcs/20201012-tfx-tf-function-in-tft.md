# Supporting tf.function in tf.transform

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) |
:               : (update when you have community PR #)                   :
| **Author(s)** | Varshaa Naganathan (varshaan@google.com)                |
| **Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com), Jiri     |
:               : Simsa (jsimsa@google.com), Zohar Yahav (zoy@google.com) :
| **Updated**   | 2020-10-12                                              |

## Objective

For tf.transform users who have TF 2 features enabled, tf.transform will
internally use tf.function to trace the user defined preprocessing function.
This is expected to work for the majority of existing use cases with no user
visible impact. However, it is possible that the user defined preprocessing
function is not traceable by tf.function (i.e. not compatible with TF 2 graph
mode execution). To accommodate such cases, we will provide a
`force_tf_compat_v1` escape hatch to allow users to fall back to TF 1
compatibility mode. Keeping the legacy behavior will continue to write the
output artifact as a TF 1 SavedModel and is not guaranteed to be compatible with
V2 Trainer.

To support this change, we need to allow tf.transform analyzers to be traceable
by tf.function. Additionally, for tf.transform to support tf.function usage
completely, it needs to handle nested tf.functions that could by themselves
contain tf.transform analyzers. tf.transform currently only traverses a single
graph generated from the user defined preprocessing function. To support nested
tf.functions, tf.transform will need to recursively traverse all the nested
FuncGraphs, identifying tf.transform analyzer nodes and replacing them with
their final computed value.

A side-effect of this proposal is that the output artifact created by
tf.transform when tf.function is used to trace the preprocessing function will
be a V2 SavedModel.

## Motivation

tf.transform today is not fully TF 2 compatible. It first traces the user
defined preprocessing function, inserting placeholders for any tf.transform
analyzers into the traced graph using a custom tracing implementation. These
analyzers are represented and evaluated using Beam and the corresponding
analyzer placeholder is replaced with a constant tensor containing the result of
the Beam computation. With TF 2, we propose to use tf.function -- an idiomatic
way to express graph computation in TF 2 -- and its tracing implementation to
handle the initial tracing of the user defined function. Besides providing TF 2
compatibilty, this reduces tf.transform’s dependence on deprecated tf.compat.v1
APIs and allows us to tighten the contract that tf.transform maintains with its
users from "supporting arbitrary Python functions" (which is generally
infeasible) to "supporting functions that can be traced with tf.function".

Another long-term advantage of tracing using tf.function is moving tf.transform
closer to supporting preprocessing of features represented by arbitrary
CompositeTensors (namely, native support for sparse and ragged data). Today
tf.transform has support only for dense and sparse data; ragged data support is
very narrow and requires workarounds.

## User Benefit

This change allows users, with TF 2 behaviors enabled, to provide tf.transform
with a preprocessing function using idiomatic TF 2 features such as tf.function.
It also guarantees that the output artifact produced by tf.transform is
compatible with V2 Trainer.

## Design Proposal

tf.transform currently traces the preprocessing function in several places. The
most important one is the `tft_beam.AnalyzeDataset` API which analyzes the
provided preprocessing function and returns the Tensorflow graph with the
computed tf.transform analyzer values. Other such APIs are
`tft.get_analyze_input_columns` and `tft.get_transform_input_columns`.

The design proposal below primarily uses `tft_beam.AnalyzeDataset` as a working
example to outline what the required changes will be to use tf.function to trace
the preprocessing function. However, the same changes will be needed in each of
the other APIs that trace the preprocessing function. Wherever relevant, any
details specific to these other APIs have been mentioned.

As tracing is an expensive operation, the preprocessing function provided by the
user should be traced minimally. To guarantee this, we need to ensure that the
FuncGraph obtained can be invoked on all possible inputs that tf.transform is
expected to feed it with. The input to the preprocessing function provided by
the user is expected to be a batch of examples. Thus, during analysis,
tf.transform invokes the graph with batches containing a varying number of
examples. This implies that the batch dimension for the tensor inputs to the
preprocessing function will vary.

For tensor inputs to a function, tf.function determines whether to re-trace or
not based on the shape and dtype of the tensor input. It will raise an exception
if the FuncGraph is invoked with tensor inputs with shape and/or dtype different
from the inputs/input_signature specified during tracing. For a dimension we
know to be varying for different inputs, we can ask the tf.function tracing
implementation to relax the strict requirement on shape. This is done by
specifying that dimension as None in the input_signature.

Note: tf.transform refers to the OSS tf.transform library while TFX Transform
refers to the TFX component.

### Recommended design

#### tf.transform internally decides whether to use tf.function based tracing or not depending on certain environment variables.

This design considers the choice of the mechanism used for tracing a
preprocessing function as an implementation detail. If a user has TF 2 behavior
enabled, it will default to using tf.function to trace the provided user defined
function. The input_signature for this tf.function will be inferred from the
schema provided to tf.transform by the user. An optional `force_tf_compat_v1`
flag will be added to the `tft_beam.Context` to allow for users to override
tf.transform’s choice of tracing mechanism as an escape hatch.

tf.transform changes:

AnalyzeDatasetCommon

```py
def expand():
  ⁞
  if not force_tf_compat_v1 and tf2.enabled():
   type_spec = TensorAdapter.type_specs(...)
   tf_fn = tf.function(self._preprocessing_fn, input_signature=[type_spec])
   graph = tf_fn.get_concrete_function().graph
  ⁞
```

tf.transform Usage:

No changes to the API exposed. If users want to opt-out of using tf.function
based tracing, they can set the `force_tf_compat_v1` flag as follows:

```
def preprocessing_fn(inputs):
  ⁞
  return {...}

with tft_beam.Context(temp_dir=tempfile.mkdtemp(), force_tf_compat_v1=True):
    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn))
```

TFX Transform Usage:

An additional field will be exposed in the TFX Transform component that
translates to the `force_tf_compat_v1` flag. The user can set this field if they
want to opt-out of using tf.function based tracing.

Pros:

*   No user friction. Most users need not understand the internal tracing
    mechanism, unless they need to override it.
*   No special handling to support TFX Transform.
*   In TFX, the preprocessing function is traced multiple times. Any side
    effects present in these functions will be invoked each time. The current
    behavior will not change with this option.

Cons:

*   No clear indication to user to define the preprocessing function as
    traceable by tf.function. Can lead to users seeing behavior they didn’t
    expect. Would need clear documentation to indicate this switch in
    tf.transform behavior with TF 2.

Long-term transition plan:

Over time, the expectation is that none or very few of our users will need to
use the legacy graph tracing path when using TF 2. The legacy path will continue
to be supported (as the default) for TF 1 users as well as TF 2 users who
explicitly disable TF 2 behavior.

We will have this functionality available for at least one minor release as an
opt-in feature (default value of `force_tf_compat_v1` will be True). It will be
announced with the release that the default behavior will be switched in the
future, allowing anyone who wants to test this out to do so and explicitly set
their choice for `force_tf_compat_v1`.

### Alternatives Considered

#### The user indicates which tracing mechanism should be used

Here the user explicitly indicates that the function they are passing is
traceable using tf.function. It is no longer an internal implementation detail
and requires the user to understand what is supported and what isn’t in this
mode.

#### Sub option a: User passes in a traced tf.function object to the tf.transform APIs

tf.transform changes:

Similar to tf.function, provide a `tft.function` decorator. It’s underlying
implementation essentially wraps the provided preprocessing_fn in a tf.function
with the specified input_signature.

```py
def function(preprocesssing_fn, type_spec):
  """Takes a user defined python preprocessing function and the type_spec for the
  inputs to it. Returns a tf.function object.

  Args:
    preprocesssing_fn: A function traceable by tf.function.
    type_spec: a Dict[Text, tf.TypeSpec] for each feature in the inputs to the preprocessing_fn.
  """
  return tf.function(preprocesssing_fn, input_signature=[type_spec])
```

AnalyzeDatasetCommon

```py
import tensorflow.python.eager.def_function as def_function
⁞
def expand():
  ⁞
  if isinstance(self._preprocessing_fn, def_function.Function):
    graph = self._preprocessing_fn.get_concrete_function().graph
  else:
    with tf.compat.v1.Graph().as_default() as graph:
      with tf.compat.v1.name_scope('inputs'):
        ⁞
     output_signature = self._preprocessing_fn(copied_inputs)
  ⁞
```

tf.transform Usage:

```py
@tft.function({
    'x': tf.TensorSpec(shape=[None,], dtype=tf.float32),
    'y': tf.SparseTensorSpec(shape=[None, 4], dtype=tf.float32)})
def preprocessing_fn(inputs):
  x = inputs['x']
  y = inputs['y']
  x_scaled = tft.scale_to_0_1(x)
  z_score = tf.sparse.to_dense(
         tft.scale_to_z_score(
             tf.cast(y, tf.float32), elementwise=True),
         default_value=np.nan)
  z_score.set_shape([None, 4])
  return {'x_scaled': x_scaled,
          'y_scaled': tf.cast(z_score, tf.float32)}

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (
       (raw_data, raw_data_metadata)
         | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
```

TFX Transform Usage:

The TFX component takes in a path to a module file that contains a function
named “preprocessing_fn”. This function should be decorated with the
`tft.function` decorator.

module_file.py:

```py
@tft.function(type_spec)
def preprocessing_fn(inputs):
  ⁞
  return {...}
```

Pros:

*   User explicitly opts in to and is aware of using tf.function based tracing.

Cons:

*   Introduces user friction to migrate to this option as the user now needs to
    modify any existing pipelines.
*   For any new pipelines that a user is creating, introduces the problem of
    discoverability. The user may just copy code from a previous pipeline which
    did not need these additional steps.
*   In TFX, the preprocessing function is traced multiple times. Any side
    effects present in these functions will be invoked each time. With this
    option, since a tf.function object is passed in, the tracing will occur only
    once and hence the side-effects will also be invoked only once.
    Additionally, the APIs that are responsible for the multiple tracings (such
    as `tft.get_analysis_dataset_keys`) will need to be modified to accept
    tf.function objects or traced graph objects.

#### Sub Option b: Add a flag to the AnalyzeDataset API to opt in to tf.function based tracing or Introduce a v2 AnalyzeDataset API that uses tf.function to trace the preprocessing function.

The examples in this section are for adding a flag to the current AnalyzeDataset
API, but the Pros and Cons and all the arguments are the same for the option of
introducing a v2 version of the AnalyzeDataset API.

tf.transform changes:

AnalyzeDatasetCommon

```py
def expand():
  ⁞
  if self._is_tf_function_traceable:
    type_spec = TensorAdapter.type_specs(...)
    tft_fn = tf.function(self._preprocessing_fn, input_signature=[type_specs])
    graph = tft_fn.get_concrete_function().graph
  else:
    with tf.compat.v1.Graph().as_default() as graph:
      with tf.compat.v1.name_scope('inputs'):
        ⁞
      output_signature = self._preprocessing_fn(copied_inputs)
  ⁞
```

tf.transform Usage

```py
def preprocessing_fn(inputs):
  ⁞
  return {...}

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn, tf_function_traceable = True))
```

TFX Transform Usage:

An additional field will be exposed in the TFX Transform component that
translates to the `tf_function_traceable` flag (or usage of a v2 API). The user
can set this field if they want tf.transform to use tf.function based tracing.

Pros:

*   User explicitly opts in to and is aware of using tf.function based tracing.
*   In TFX, the preprocessing function is traced multiple times. Any side
    effects present in these functions will be invoked each time. The current
    behavior will not change with this option.

Cons:

*   Introduces user friction to migrate to this option as the user now needs to
    modify any existing pipelines.
*   For any new pipelines that a user is creating, introduces the problem of
    discoverability. The user may just copy code from a previous pipeline which
    did not need these additional steps.
*   While the tracing behavior is unchanged for TFX Transform (Pros#2), the APIs
    that are responsible for the multiple tracings (such as
    `tft.get_analysis_dataset_keys`) will need to be modified to accept an
    equivalent flag to `tf_function_traceable` (or provide v2 versions of each
    of these APIs). There could be some unexpected behavior observed if the
    value for this flag was different between the different APIs (or different
    versions of the API were invoked) for the same pipeline.

#### Sub Option c: Add a parameter to the `tft_beam.Context` that users can use to opt-in to tf.function based tracing.

tf.transform changes:

AnalyzeDatasetCommon

```
def __init__(self, preprocessing_fn, pipeline=None):
  ⁞
  self._is_tf_function_traceable = Context.get_is_tf_function_traceable()
  ⁞

def expand():
  ⁞
  if self._is_tf_function_traceable:
    type_spec = TensorAdapter.type_specs(...)
    tft_fn = tf.function(self._preprocessing_fn, input_signature=[type_specs])
    graph = tft_fn.get_concrete_function().graph
  else:
    with tf.compat.v1.Graph().as_default() as graph:
      with tf.compat.v1.name_scope('inputs'):
        ⁞
     output_signature = self._preprocessing_fn(copied_inputs)
  ⁞
```

tf.transform Usage

```
def preprocessing_fn(inputs):
  ⁞
  return {...}

with tft_beam.Context(temp_dir=tempfile.mkdtemp(), tf_function_traceable = True):
    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
            preprocessing_fn))
```

TFX Transform Usage:

An additional field will be exposed in the TFX Transform component that
translates to the tf_function_traceable parameter in the Context. The user can
set this field if they want tf.transform to use tf.function based tracing.

This option has all the pros and cons of Option b. In addition to that:

Pros:

*   The API complexity is reduced by moving this knob to the Context.

Long-term transition plan for Sub option a, b and c:

Since there is user friction introduced to using the new APIs, we need to take
some explicit steps to move users to using the tf.function based tracing.
Initially, we will show deprecation warnings to users on TF 2 who do not use the
new API, prompting them to use the newer APIs instead. Documentation and
examples will be updated to make this the recommended API with TF 2.

The transition path for this option is likely to be slower than with the
recommended design as it requires explicit user action.

### Performance Implications

Performance impact should be neutral.

### Dependencies

No new dependencies introduced.

### Engineering Impact

Moves tf.transform closer to providing native TF 2 support and removing usage of
any deprecated or non-public APIs. This should make code maintenance easier.

### Platforms and Environments

No special considerations across different platforms and environments.

### Best Practices

With this change, tf.transform's contract with users on preprocessing functions
that it can handle is tightened and clearly defined to only handle functions
that tf.function can trace. This change will be communicated in documentation
and over time be enforced.

### Tutorials and Examples

The only API change is a flag for opt-out. This will be updated in the API docs.

### Compatibility

Since Transform's support for TF 2 behaviors is currently experimental, there is
a possibility that some existing preprocessing functions can no longer be traced
by tf.transform. In such cases, if the users have TF 2 behavior enabled, we
provide the `force_tf_compat_v1` flag to allow them to retain the deprecated
tracing behavior.

When this proposal has been implemented, we will do an initial release with
`force_tf_compat_v1` set to `True` by default. With this release, we will
communicate a plan to switch the default value for `force_tf_compat_v1` to
`False` in a future release. This will give users with TF 2 behaviors enabled
the chance to test their code and migrate or explicitly opt-out of this feature.

### User Impact

Since this is a step towards native TF 2 support in tf.transform, this will only
impact users who are currently using tf.transform with TF 2 features enabled.
This change will allow such users to use idiomatic TF 2 features such as
tf.function with tf.transform. It also tightens tf.transform's contract to only
support preprocessing functions that can be traced with tf.function as opposed
to arbitrary python functions.

## Questions and Discussion Topics
