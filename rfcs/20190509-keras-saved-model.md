# Keras SavedModel saving/loading 

| Status        | Accepted                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Kathy Wu (kathywu@google.com)                        |
| **Sponsor**   | Karmel (karmel@google.com)                           |
| **Updated**   | 2019-05-30                                           |

## Objective

Add ability to (1) save a 
[TensorFlow SavedModel](https://www.tensorflow.org/alpha/guide/saved_model) 
from a Keras model and (2) load a Keras model from a SavedModel. This is a 
standalone serialization format, which allows models with custom objects to 
be saved/loaded without having the original code (currently required by the HDF5 and JSON formats). 

**Goals:**

- Add SavedModel format for saving Keras models. 
- Support comprehensive model serialization/deserialization 
  - Define model serialization coverage (what parts of the model are saved 
    to the SavedModel) 
  - Address serialization pattern that is reusable across different
    frameworks (TensorFlow Hub, tf.module, etc.) 
  - Define SavedModel deserialization (properties of the reconstructed 
    model) 

**Non-goals:**

- Serializing custom training loops. Only the standard `model.compile` to `model.fit` use case is covered. Custom training loops may still be saved by wrapping it in a tf.function, and passing it to the `signatures` argument or setting it as an attribute of the model (e.g. model.train_loop = ...). 
- Changing existing serialization formats

## Motivation

Keras is a high level API for defining models while using a TensorFlow graph in the backend. The current model serialization format stores details for recreating the model object and its internal layers. The Python objects define the operations to add to the graph. This is great for saving and loading models that solely use pre-defined layers and follows the Keras Sequential or Functional API. If the model contains custom objects, their implementations must be passed to the `tf.keras.model.load_models` function. 

The SavedModel format saves the TensorFlow graph, so it is capable of serializing custom objects, and deserializing without needing the original implementation. This allows Keras models to move freely between different systems and languages (Python, JS, and C++ to an extent).

Familiarity with the terminology used in [RFC: SavedModel saving/loading in 2.X](https://github.com/tensorflow/community/blob/master/rfcs/20181116-saved-model.md) is recommended.

Related works:
- [RFC: SavedModel saving/loading in 2.X](https://github.com/tensorflow/community/blob/master/rfcs/20181116-saved-model.md)
- [Well-defined Keras JSON Spec in development](https://github.com/tensorflow/tfjs-layers/tree/master/src/keras_format)

## Design Proposal

### API changes
**Symbol changes**

- `tf.keras.Model.save` or `tf.keras.models.save_model`: Add additional 
   arguments `save_format` and `signatures`.

  ```
  model.save(path, include_optimizer=None, save_format=None, signatures=None)
  ```
  - include_optimizer: (for SavedModel format) Whether or not the SavedModel should contain the optimizer. Note that even if this argument is `False`, any compiled losses and metrics are saved, since they are separate from the optimizer.
  - save_format: Either `h5` or `tf`. Specifies the format for saving the
    model. If left as None, the save format will default to tf unless the 
    path ends with `.h5`, `.hdf5`, or `.keras`.
  - signatures: Signatures to attach to the exported SavedModel. If `None`
    (default), the model's call() will be traced and used as the default 
    signature. The `signatures` argument in `tf.saved_model.save` has a more
    comprehensive description.

**Behavior changes**

- `tf.keras.models.load_model`: Currently only loads h5 files. This will be 
  modified to automatically detect SavedModels. In both cases, a Keras model 
  object is returned. The root object in the SavedModel must at least contain
  [shared endpoints](#shared-endpoints), otherwise, an error is raised. 
- `tf.saved_model.save` and `tf.keras.models.save_model` and `model.save`: These functions have consistent saving results. Additional dependencies/functions are serialized to the SavedModel. See
  [Serialization coverage](#serialization-coverage)

**Note**

- `tf.saved_model.load` (tf.saved_model.load_v2 in TF 1.13+) will *not* 
   change. Instead of returning a Keras model like 
   `tf.keras.models.load_model`, this will return a generic object with 
   similar attributes ([Deserialization details](#deserialization-details) 
   section briefly describes the generic object).

#### Alternatives considered

- `model.export`: As an alternative to `model.save`, which contains 
  export-exclusive arguments such as optimization (precision, model pruning, 
  frozen graphs), or target framework (TF JS, TF Lite, TF Hub, etc.). Perhaps
  this method will be added in the future, but for saving/loading using 
  SavedModel, modifying `model.save` is more intuitive.
- Directly return a keras model instead of a generic object from 
  `tf.saved_model.load`. This makes it difficult for other frameworks built 
  on top of TensorFlow to work with SavedModels saved from Keras. Handling
  Keras object comes with subtle issues when not using the Keras API (for
  example, Keras objects often utilize a backend graph/session, even in eager 
  mode).

### SavedModel Signatures

The `signatures` argument in `model.save` and `tf.saved_model.save` allows models saved to SavedModel to define signatures (see [RFC: SavedModel saving/loading in 2.X](https://github.com/tensorflow/community/blob/master/rfcs/20181116-saved-model.md)). Signatures are primarily used for serving, but can be used to save and load unattached tf.functions.

The following example saves a model with signatures to predict from raw inputs, or from tf.Examples:

```
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, input_shape=(32,)))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax)) 

@tf.function(input_signature=[tf.TensorSpec([None, 32], dtype=tf.float32)])
def model_predict(input_batch):
  return {'outputs': model(input_batch, training=False)}

@tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string)])
def classify(serialized):
  feature_spec = {'input_batch': tf.io.FixedLenFeature([32], tf.float32)}
  deserialized = tf.io.parse_example(serialized, feature_spec)
  scores = model_predict(deserialized['input_batch'])
  return {'classes': tf.argmax(scores, 1), 'scores': scores}

model.save(
  '/tmp/keras_model', signatures={'classify': classify, 'predict': model_predict})
```

And later, loading the SavedModel:

```
model = tf.keras.load_model('/tmp/keras_model')
model.signatures  #  {'classify' → tf.function, 'predict' → tf.function} 
```

If the `signatures` argument is left empty, then a default signature is created with the traced model call function. In the future, we may consider always exporting the default signature (even if the `signatures` argument is set).

### Serialization coverage

The goal is simple - save a Keras layer or model object to SavedModel, and load it back in the same state. Same state is defined as all of the below are equivalent- 
- variables (list of all variables, trainable variables, and non-trainable variables)
- forward pass (call function)
- losses
- metrics
- child layers (and their states)
- python attributes
  - name
  - trainable
  - config
  - dtype
  - batch_input_shape 
  - input_spec
- (if compiled model) compiled arguments
  - optimizer (hyperparameters, iterations)
  - loss functions
  - metrics

All of the above attributes are serialized in the SavedModel as either metadata, checkpointable objects, or  functions.

Checkpointable objects and functions are as defined in [RFC: SavedModel Save/Load in 2.x](https://github.com/tensorflow/community/blob/master/rfcs/20181116-saved-model.md) (see *Concrete function graph* and *Checkpointable* in the Terminology section).

Metadata will be a newly added property of checkpointable objects in the SavedModel. Checkpointable objects may save arbitrary strings to this field. Python attributes such as layer names are saved to the metadata.

See [Detailed Design](#detailed-design) for more info on how each attribute is saved. 

### Model deserialization
Reconstructing a Keras model is a matter of remapping the saved attributes (listed above) to the original attributes. A few more details are in the [Detailed design](#detailed-design).

## Detailed Design

### Serialization details
Keras models contain many Python-defined components. For example, the `trainable_variable` property lists the model's trainable variables by recursively retrieving the trainable variables from each of the child layers. Another example is `model.call`, a python function that calls child layers and adds ops to the backend graph.

Only Tensorflow checkpointable objects and functions can be serialized to SavedModel. Serializing a Keras model as-is results in a checkpointable object that does not resemble a Keras model at all. Thus, extra checkpointable objects and functions must be created during serialization. 

The following checkpointable objects and functions are attached to the saved Keras model:

- `variables`: List of all variables in this layer and sublayers.
- `trainable_variables`: List of all trainable variables in this layer and sublayers.
- `non_trainable_variables`: List of all non-trainable variables in this layer and sublayers.
- `regularization_losses`: List of unconditional loss functions in this layer 
  and sublayers. Each function takes no arguments, and returns a scalar 
  tensor. 
- `layers`: Flat list of all sublayers (does not include metrics, even though Metric subclasses Layer).
- `metrics`: List of all metric layers attached to this layer and sublayers.
- `_variables`: List of all variables owned by this object (and not sublayers)
- `__call__`: Returns the outputs of the call function.
- `call_and_return_conditional_losses`: Returns the outputs of the call function, as well as a list input-dependent losses (does not include the activity regularizer loss).
- `call_and_return_all_conditional_losses`: A function that calls the model and returns returns outputs and returns all input-dependent losses. Unlike `call_and_return_conditional_losses`, the losses returned in this function includes the activity regularizer and any compiled losses.
- `activity_regularizer_fn`: Activity regularization function
- `compile_losses`: List of loss functions added during `model.compile`.
- `compile_metrics`: List of metric objects added during `model.compile`.

The optimizer is a checkpointable object, so it is automatically saved to the SavedModel.

**Public vs private variables (variables vs _variables)**

The public attributes are exported so that the all variables/trainable variables/etc. may be accessed without the Keras python logic to recursively traverse all the sublayers. The private variable attribute is exported so that when the model is deserialized:
1. it is clear which objects own variables
2. variables are guaranteed to be in the same order. This is important for Keras models, which uses layer and weight order for certain operations (e.g. saving/loading to HDF5).

**Call functions**

Two versions of the call function are exported. Exporting `__call__` enables the `model(inputs)` function to be retained in the generic object created by `tf.saved_model.load`. `call_and_return_conditional_losses` is exported for model deserialization.

### Deserialization details
The generic loader, `tf.saved_model.load`, creates a generic object with attributes as saved in the SavedModel. Loading the saved Keras model (with added checkpointable objects and functions as listed above) will produce:

```
GenericObject obj → .variables, .trainable_variables, etc.
```

`GenericObject` is similar to a Keras model object, but lacks the `.fit`, `.test` and `.predict` methods.

When reconstructing a Keras model, the saved attributes are remapped to the original names. The exception is the call function, which uses `call_and_return_conditional_losses` instead of `__call__`.

### Shared endpoints

Having shared endpoints allows models to be used interchangeably between different frameworks. The following properties were proposed by arnoegw@google.com and andresp@google.com, from the TF Hub team, as common endpoints shared by all modules:

- `__call__`: A function that takes inputs to the model and returns outputs
- `variables`: List of all variables in the model
- `trainable_variables`: List of all trainable variables in the model
- `regularization_losses`: List of callables that return a scalar tensor.
- `call_and_return_all_conditional_losses`: A function that calls the model and returns returns outputs and returns all input-dependent losses. Unlike `call_and_return_conditional_losses`, the losses returned in this function includes the activity regularizer and any compiled losses.

All losses contained in the model are split between `regularization_losses` and `call_and_return_all_conditional_losses`. 

Any SavedModel with these endpoints defined may be loaded as a Keras model using `tf.keras.models.load_model`.

### Compatibility Guarantees
Keras SavedModels are backward and forward compatible across minor TensorFlow versions (similar to [GraphDef](https://www.tensorflow.org/alpha/guide/version_compat)). Therefore, checkpointable object and tf.function attributes will not be removed from the SavedModel. New attributes can be added if additional serialization is requested. 

## Questions and Discussion Topics

1. Do we all agree on the API changes?
   - Regarding the `include_optimizer` argument: If this is set to False, then the optimizer will not be included. If the model has been compiled, then the eval graphs.
   - Should we always save the default model signature? No (at least not initially). This can be added later on, but the change should also be added to `tf.saved_model.save`.
2. Are there other aspects of the model that should be serialized/deserialized?
   - Add an additional method that includes all input-dependent losses (`call_and_return_all_conditional_losses`). This combines the losses generated from `call_and_return_conditional_losses`, `compile_losses`, and `activity_regularizer_fn`.
3. Should `tf.saved_model.load` return a generic object or Keras model?
   **Generic object**
4. Syncing common endpoints with tf.module
   tf.module is extremely open-ended, so just the `.variables` attribute should be synced between tf.module SavedModel and Keras SavedModel. Note about `.trainable_variables` -- Keras layers and tf.modules have different definitions of trainable variables.
