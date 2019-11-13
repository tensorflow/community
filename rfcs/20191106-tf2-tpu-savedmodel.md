# TPU SavedModel Export API for TF2.x

Status        | Proposed
:------------ | :-----------------------------------------------------------
**RFC #**     | [171](https://github.com/tensorflow/community/pull/171)
**Author(s)** | Zhuoran Liu (lzr@google.com), Youlong Cheng (ylc@google.com)
**Sponsor**   | Jonathan Hseu (jhseu@google.com)
**Updated**   | 2019-11-06

## Objective

Provide an API to allow TF2 users to export TPU saved models <b>for
inference</b>, which:

+   Provide a user-friendly way to specify which function to run on TPU;
+   Hides Graph construction and TPU inference specific logic (multi-core
    support, etc) from users;
+   Allows specifying tags in SavedModel.

## Motivation

### TPU Serving Requirement

Serving a model on TPU is not as straightforward as serving on CPU and GPU,
because TPU serving has special requirements, listed as follows:

+   Contract between TensorFlow graph and TF2XLA Bridge. The new bridge will
    still respect this contract. The information of “which part of computation
    should run on TPU” is conveyed from Graph to Bridge by tagging a special
    Node attribute `_tpu_replicate`. Because of this, we need to provide
    information during Function object instantiation in order for this attribute
    to be correctly attached to Nodes during Graph building;

+   Multi-core TPU serving. TPU has various deployment configurations, for
    example 1x1 Dragonfish chip has 2 cores, 2x2 Dragonfish chip has 8 cores.
    The exported saved model should be able to run on different configurations
    and can leverage all the TPU cores.

    -   When users write their model code, they likely don’t have information
        about how many TPU they have for serving / which core they can use.
        Therefore we need a Graph level abstraction to express graph
        partitioning information. tf.device() cannot serve this purpose, because
        it requires users to have knowledge about the physical device they have
        during serving;
    -   To make efficient usage of multicore TPUs, we need to encapsulate TPU
        computations as FunctionDef, and construct TPUPartitionedCall /
        TPUOrdinalSelector to perform round-robin core selection;

+   Tagging system of SavedModel. Users rely on a tagging system to load their
    models for serving. E.g. CPU MetaGraphs have one tag ‘serve’, while TPU
    MetaGraphs have two tags ‘serve’ and ‘tpu’. Only with correct tags can
    SavedModels be loaded correctly.

Below is an intuitive example of how a TPU graph is different from a CPU one:

![Original CPU Graph](20191106-tf2-tpu-savedmodel/cpu_graph.png)
<center>Original CPU Graph.</center>

![TPU Graph](20191106-tf2-tpu-savedmodel/tpu_graph.png)
<center>TPU Graph.</center>

### Limitation of current `tf.saved_model.save()`

MetaGraphDef allows saving customized tags. Current downstream components like
TPU model-server, TFX infra-validator use the tags to load the specific
MetaGraph. However tf.saved_model.save() does not allow users to specify the set
of tags in MetaGraphDef, but hard-coded the MetaGraph to have only one ‘serve’
tag.

### User Control of Device Placement

There has to be a way for users to specify which part of computation should be
placed on TPU, because there’s no perfect device placement policy that can work
for every use case. For example even though dense embedding ops are allowed on
TPU, serving models might still want to run embedding lookups on CPU because the
embeddings are too big to fit on TPU.

![Customized Embeddings](20191106-tf2-tpu-savedmodel/customized_embeddings.png)
<center>Example of user control. In this graph, both ‘custom_embedding’ and
‘dense’ can run on TPU. But users want ‘custom_embedding’ to run on CPU for
whatever reason, e.g. CPU computations can be parallelized, users don’t have
enough TPU resources, etc. In this case, there has to be a way for them to tell
SavedModel that only ‘dense’ is to run on TPU.</center>

## Design Proposal

### Caveat

`@tf.tpu.function` should only be used for serving. It should never appear in
training code.

### User Facing API

<b>For General TF2 Users</b>

Under the proposed design, users will need to do the following things to export
a TPU SavedModel in TF2.x:

1.  Replace @tf.function with @tf.tpu.function for functions they wish to run on
    TPU;

    ```python
    # `model` can be any Python Callable. E.g. A Keras Model.
    @tf.tpu.function
    def predict_step(image_tensors):
      return model(image_tensors)
    ```

2.  Create main serving function and call the tpu function above. The main
    function might have additional TF ops which can’t run on TPU (e.g.
    `tf.decode_image`:

    ```python
    @tf.function
    def serve(images):
      image_tensors = tf.decode_image(images)
      return predict_step(image_tensors)
    ```

    And then create a signature:

    ```python
    signatures = {
        'serving_default':
            serve.get_concrete_function(...),
    }
    tags = [tag_constants.SERVING, tag_constants.TPU]
    ```

3.  Pass the both signatures to `tf.saved_model.save()`:

    ```python
    tf.saved_model.save(
         model,
         export_dir='...',
         signatures=signatures,
         tags=tags)
    ```

The resulting TPU inference graph looks like this:

![Resulting TPU Graph](20191106-tf2-tpu-savedmodel/tpu_result.png)
<center>Resulting TPU Graph.</center>

<b>For Advanced Users who need customized Ops</b>

In such cases, we provide the flexibility for users to tweak `@tf.tpu.function`.

1.  If users wish not to use TPUPartitionedCall, they can disable using
    TPUPartitionedCall:

    ```python
    @tf.tpu.function(use_tpu_partitioned_call=False)
    def predict_step(images):
      ...
    ```

2.  Users can also nest TPU functions within BatchFunction:

    ```python
    @tf.tpu.function(use_batch_function=True,
                     # Below arguments for BatchFunction
                     # are optional
                     max_batch_size=...,
                     allowed_batch_sizes=...
                     ...)
    def predict_step(images):
      ...
    ```

3.  User can also customize their TPUPartitionedCallOp:

    ```python
    @tf.tpu.function(use_tpu_partitioned_call=True,
                     device_ordinal=0)
    def predict_step(images):
      ...
    ```

<b>For Keras Users</b>

Option 1:

Introduce argument `export_to_tpu`. For Keras users, they will only need to pass
`export_to_tpu=True` to save to TPU SavedModel. (Currently, we require the graph
defined by `model` to be completely TPU-compatible.)

```python
tf.keras.models.save_model(
    model,
    filepath='...',
    export_to_tpu=True)
```

Option 2:

Keep tf.keras.models.save_model() unchanged. Users use a keras model as if they
were using a TF2 Function.

```python
# isinstance(model, (tf.keras.Model, tf.keras.layers.Layer)) == True
@tf.tpu.function
def predict_step(image_tensors):
  return model(image_tensors)
```

### Changes to TF2.x API

1.  `tf.saved_model.save()` will take an optional argument `tags`.

    `tags` is an optional argument which represents a list of tags. This allows
    users to specify customized tags. For example, Servomatic or model server
    requires both ‘tpu’ and ‘serve’ tags to load TPU saved model.

2.  Implement an additional `@tf.tpu.function` decorator in
    `tensorflow/python/tpu/tpu.py`. This decorator handles TPU rewriting under
    the hood.

    `tf.tpu.function()` takes the following optional arguments:

    -   `func`: A Python function. If not set, will return a wrapper that takes
        a Python function. This allows @tf.tpu.function to be called w/ or w/o
        arguments;
    -   `use_tpu_partitioned_call`: boolean. Controls whether TPUPartitionedCall
        will be used;
    -   `device_ordinal`: Used in conjunction with `use_tpu_partitioned_call`. A
        tensor or a TF Function object that returns a tensor, designating the
        device ordinal. Default to tpu_ordinal_selector();
    -   `use_batch_function`: boolean. Controls whether BatchFunction will be
        used;
    -   `num_batch_threads`, `max_batch_size`, `batch_timeout_micros`,
        `allowed_batch_sizes`, `max_enqueued_batches`: arguments used to
        configure BatchFunction.

### Changes to Keras API

<b>Option 1</b>

If Keras users would like `tf.keras.models.save_model()` to work directly for
exporting TPU SavedModel, without having knowledge of tf.function / tags /
signatures. The only way to achieve this is to hide those logics under
`tf.keras.models.save_model()`.

After the change, `tf.keras.models.save_model()` will have two additional
arguments:

1.  `export_to_tpu`: Simply setting this to `True` will export TPU model;
2.  `tags`: Optionally for advanced users, if they want to have more control of
    what tags they are using, they can use this argument as if they are using
    TF2.x saving API.

<b>Option 2</b>

No change. Users can save a keras model for TPU inference with
tf.saved_model.save().

## Detailed Design

### TF2.x API

Under the hood, exporter API is doing the following things:

+   The @tf.tpu.function wraps user-specified function;
+   Tag the MetaGraph with user-defined tags.

<b>Step 1:</b> Use a new decorator to wrap TPU version of the user-specified TPU
function. It calls tpu.rewrite inside the original function to generate a TPU
version of graph. By default, this will create a tpu function. If users wish to
preserve both CPU and TPU function, they can set ‘preserve_cpu_fn=True’.
Optionally, they can use `use_tpu_partitioned_call` and `use_batch_function` to
customize the Function object they get.

```python
# tensorflow/python/tpu/tpu.py

def _tpu_partitioned_call_wrapper(tf_func, device_ordinal):
  ...

def _batch_function_wrapper(tf_func,
                            num_batch_threads,
                            max_batch_size,
                            batch_timeout_micros,
                            allowed_batch_sizes,
                            max_enqueued_batches):
  ...

def _rewrite_func_wrapper(func):
  ...

@tf_export("tpu.function")
def tpu_function(func=None, *args, **kwargs):
    ...
    tpu_func = _rewrite_func_wrapper(func)
    ...
    if use_tpu_partitioned_call:
      tpu_fn = _tpu_partitioned_call_wrapper(tpu_fn, device_ordinal)
      ...
    if use_batch_function:
      tpu_fn = _batch_function_wrapper(tpu_fn,
                                       num_batch_threads,
                                       max_batch_size,
                                       batch_timeout_micros,
                                       allowed_batch_sizes,
                                       max_enqueued_batches)
      ...
```

<b>Step 2:</b> Create a MetaGraph with designated tags for the SavedModel.

```python
# tensorflow/python/saved_model/save.py

saved_model = saved_model_pb2.SavedModel()
...
meta_graph_def = saved_model.meta_graphs.add()
asset_info, exported_graph = _fill_meta_graph_def(
    meta_graph_def, saveable_view, signatures,
    options.namespace_whitelist,
    tags=list(tags))
...
```

### Support for Keras saving API (Under option 1 for Keras)

Adding an argument `export_to_tpu` for `tf.keras.models.save_model()`, which if
set to true will rewrite the model for TPU inference.

Adding an argument `tags` for `tf.keras.models.save_model()` which has the same
semantics as that in `tf.saved_model.save()`.

```python
# tensorflow/python/keras/saving/save.py

@keras_export('keras.models.save_model')
def save_model(model,
               filepath,
               overwrite=True,
               include_optimizer=True,
               save_format=None,
               signatures=None,
               tags=None,
               export_to_tpu=False,
               options=None):
  ...
    if (export_to_tpu and
        (not tags
         or tag_constants.TPU not in tags)):
      checkpoint_graph_view = save_lib._AugmentedGraphView(model)
      signatures = find_function_to_export_tpu(checkpoint_graph_view)
      tags = [tag_constants.SERVING, tag_constants.TPU]

  saved_model_save.save(model, filepath, overwrite,
                        include_optimizer,
                        signatures,
                        tags,
                        options)
```
