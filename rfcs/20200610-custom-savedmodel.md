# Building Custom SavedModels

| Status        | (Proposed / Accepted / Implemented / Obsolete)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Kathy Wu (kathywu@google.com)                        |
| **Sponsor**   | karmel (karmel@google.com)                           |
| **Updated**   | 2020-06-10                                           |

## Objective

Add a way to implement serialization of high-level frameworks using SavedModel.

## Motivation

High level library developers do not have a good way of utilizing the TensorFlow SavedModel format. Other than pure TensorFlow models, which use tf.Variables and define the model in tf.functions, it is hard to serialize advanced models and training frameworks. I often get questions such as:

*  I've written a custom layer, how do I make sure it can be serialized to *  SavedModel? Also: Why am I getting Error: _____ when I try to save this layer?
*  How do I save python functions with my Model?
*  How do I save resources and assets?
*  How do I decrease the SavedModel size/loading time?

The main problem is the knowledge barrier -- in order to become familiar with SavedModel enough to debug it, this requires learning about the format itself and a mixture of TensorFlow internals (tf.Function tracing, checkpointing, TF 1.X concepts). 

Part of the issue lies in lack of good developer documentation. There is a growing collection of docs and guides, however many of these are not public, nor is it obvious which docs are relevant to SavedModel serialization. Thus, one way I'll be addressing the problem is by compiling the information into a series of developer docs. 

Improving the docs can help, but doesn't really resolve the issue that a large time investment is needed. Serialization should be a painless process, not something that framework developers have to worry about. I propose adding an API for defining custom saving and loading logic that abstracts away most of the internal implementation details.

API has the following goals:

*  Allow serialization and deserialization of high-level modules and resources
  *  Version management stored in the VersionedTypeRegistration.
*  Integration with other frameworks (e.g. Hub with Keras, Keras with XAI?)
  *  Hub with Keras use case: Save from Keras, load into Hub. 
  *  Another idea is allow SavedModels with a specific spec to be loaded into Keras
  *  Other frameworks built on Keras/SavedModel (e.g. TFMA) 
*  Configuring multiple levels of saving. (Examples of Keras multi-level save)
  *  Comprehensive save: every layer call function is saved individually, a
  *  Lightweight save: Only Config is saved for each layer
  *  Inference-only export: The SavedModel size is decreased as much as possible, and no extra functions and objects are saved.
*  Improve debugging and testing
  *  Transparent saving and loading process
  *  Base TestClass that provides 
  *  Easy inspection of the SavedModel/Guides explaining the process
*  Help Keras move off of private APIs


## Background
### Core/low level TensorFlow SavedModel serialization

The details are thoroughly explained in Object-based Checkpointing and TF 2.X export and import APIs. 

To summarize, Checkpoints and SavedModel traverse an object graph, defined by a root Trackable object and its children (other trackable objects). Child objects are primarily tracked through attributes, where assigning one Trackable object to another creates a dependency that is saved by the parent object. 

When saving checkpoints, this object graph is traversed to find all resource values that the root object references, e.g. Variables or tables. The object graph is also saved to the Checkpoint and used when restoring the values to another root object.

In a SavedModel, the object graph is stored so that an object with the same structure may be loaded. The SavedModel object graph contains not only other Trackable objects, but also tf.functions.

When loading from a SavedModel, the object graph will consist of these types:
* tf.Variable
* Function
* UserObject: A generic Trackable object. All attributes in the object are generated from the edges in the object graph. The object also contains an extra .signatures attribute which maps SignatureDef keys to their corresponding functions.
* Data structures: ListWrapper, DictWrapper, etc. Trackable classes of python data structures)
* TrackableAsset: Stores the path to an external file. When saved, these files are copied into the SavedModel. When loaded, the TrackableAsset stores the path to the file in the SavedModel. Used by Table classes in lookup_ops.py to save/load vocabulary files
* TrackableResource: Stores tensors (e.g. table handles).

### Higher Level Serialization
SavedModel is defined by TensorFlow object and op graphs. In the past (before TF 1.15), there was no object graph in the SavedModel -- only the op graph, and signatures for executing parts of the graph. The addition of the object graph introduced the ability to serialize high-level objects and structures.

In this doc, a high-level object is defined as an object that cannot be saved or recreated solely from tf.functions (defined in the op graph), and connections between itself and its children (defined in the object graph). In other words, an object that is reliant on:
logic that can't be expressed in a TensorFlow graph (e.g. keeping a master list of all variables that the object has access to)
uncheckpointed values (e.g. specific parameters that must be used when initializing the object)

Serialization of high-level objects into the SavedModel format can mean different things. First we must consider the purpose. How/where will the SavedModels be loaded?
* From the same framework
* From a different high-level framework
* From another language
* From TensorFlow Serving and other inference-only consumers
* A combination of the above

Here's an example to help clarify the different purposes:

```
class MyModule(tf.Module):
  def __init__(self, a, training):
    self.a = a
    self.b = tf.Variable(1.0)
    self.training = training

  def ab(self):
    return self.a * self.b
  
  def __call__(self, inputs):
    if self.training:
      return inputs + self.ab
    else:
      return inputs
```

When loading within the **same framework**, the loaded object is expected to have type `MyModule`. Thus, the attributes used during initialization -- `a, training` must be saved to the SavedModel. Of course, all checkpointed values such as variable `b` must also be saved. When loading, a `MyModule` object should be recreated using the saved `a` and `training` arguments, and the value of b should be restored from the checkpoint.

When loading using a **different high-level framework**, or from the core tf.saved_model.load, it is up to the serialization implementor to decide what is saved. For example, one goal is to expose the same endpoints as `MyModule`, so the loaded object can be used the same way without the class being defined. To accomplish this, the functions `ab` and `__call__` should be converted to tf.functions, and added to the object graph when saving. The caveat is that the value of a will be frozen, unless converted to a variable.
Note that when saving/loading within the same framework, the extra functions and variables do not need to be added.

In the **inference-only case**, let's say the call function is intended to be used for serving. Like with the previous case, the call function needs to be converted, and added to the SavedModel as a signature.

The requirements of high-level serialization can be summed up as:
1. A way to add metadata to the saved object
2. A way to modify the saved object graph and add functions and new checkpointable objects.
3. A way to change how an object is loaded based on the saved metadata and object graph.



## Design Proposal

The proposal follows the Keras `get_config` and `from_config` construct, where one method gathers the data needed to recreate the object, while the other method uses the data to create a new instance of the object. Classes that implement these methods must be registered to a named package. Users can choose which packages to enable during serialization and deserialization. 

### Example 
To help illustate this concept, here is an example where we create a custom class that lightly wraps tf.Module. The custom serializer and deserializer saves and restores the object name and variables.

This object is registered to the "Example" package, so the custom serializer and deserializer functions are only used with the package Serializer (which can be created by calling Serializer('Example').

```python
  @register_serializable(package='Example')
  class CustomSerializable(tf.Module):
    def __init__(self, name):
      super(CustomSerializable, self).__init__(name=name)
      self.vars = [tf.Variable(0.0), tf.Variable(1.0)]

    def saved_model_serializer(self, **unused_kwargs):
      metadata = {'name': self.name}
      return SaveSpec(
        metadata=json.dumps(metadata))

    @classmethod
    def saved_model_deserializer(cls, loader, load_spec):
      metadata = json.loads(load_spec.metadata)
      return cls(metadata['name'])

  obj = CustomSerializable('Obj')

  # Create a serializer for the 'Example' package.
  Serializer('Example').save(obj, path)

 
  # To load the 
  loaded = Serializer('Example').load(path)
  assert loaded.name == 'Obj'
  assert len(loaded.vars) == 2
```

The serializer and deserializer functions can be later updated if necessary:
```python
  @register_serializable(package='Example', version=2)
  class CustomSerializable(tf.Module):
    def __init__(self, name):
      super(CustomSerializable, self).__init__(name=name)

    def build(self):
      # Vars are now listed separately.
      self.var_1 = tf.Variable(0.0)
      self.var_2 = tf.Variable(1.0)

    @tf.function
    def call(self, inputs):
     return inputs

    def saved_model_serializer(self, **unused_kwargs):
      metadata = json.dumps({'name': self.name})

      # Use the default serializer to automatically create a dictionary listing
      # all dependencies and functions attached to this object:
      #   {'var_1': self.var_1, 'var_2': self.var_2, 'call': self.call}
      children = Serializer().serialize(self).children

      return SaveSpec(metadata=metadata, children=children)

    @classmethod
    def saved_model_deserializer(cls, load_spec):
      metadata = json.loads(load_spec.metadata)
      children = load_spec.children
      version = load_spec.version
      if version == 2:
        obj = cls(metadata['name'])

        # Call `deserialize` to load the `load_spec` saved in the children
        obj.var_1 = load_spec.deserialize(children['var_1'])
        obj.var_2 = load_spec.deserialize(children['var_2'])

        return obj

      elif version == 1:
        # Example of maintaining backwards compatibility.
        obj = cls(metadata['name'])
        obj.build()

        # The checkpoint saved from version 1 is incompatible with the current
        # object. The v1 checkpoint has the structure:
        v1_checkpoint = tf.train.Checkpoint(vars=[obj.var_1, obj.var_2])

        # While the current object's checkpoint is:
        #   tf.train.Checkpoint(var_1=self.var_1, var_2=self.var_2])

        # Call set_checkpoint to ensure that the checkpoint from V1 can be
        # loaded into the object.
        load_spec.set_checkpoint(v1_checkpoint)
        return obj
```


### API

#### Registration
This proposal adds a global registry of serializers and deserializers, each of which are tied to a class. When serializing an object or deserializing the SavedModel proto, this registry is used to find the right serializer and deserializer functions to call. See the [Custom Serializer](#Custom-serializer) section for more information about how the registry is used.

```python
@register_serializable(
    package="Custom", name=None, version=1, alternate_ids=None)
```
This decorater is applied to the class to be serialized, and it registers the serializer and deserializer functions of the class under the identifier "`{package}.{name}"`. (See Serializer and Deserialization specification for details)

* `package`: Name of the package. Can be `None`, in which case the class serializer and deserializer are registered under the global namespace used by `tf.saved_model.save` and `tf.save_model.load`. `tf.Variable`, `tf.saved_model.Asset`, `tf.keras.Optimizer` are registered under the `None` (global) namespace. Keras models and layers are registered under the `keras` namespace. The package comes into play when creating a custom `Serializer`.
* `name`: Name to use (defaults to the class name). The identifier string stored in the SavedModel is composed of both package and name.
* `version`: Producer version
* `alternate_ids`: Other IDs that the package may be saved as. This is useful for Keras backwards compatibility, which currently uses identifiers like _tf_keras_layer which will change to keras.Layer once the registration system is added.

`register_serializable` is similar to the method `tf.keras.utils.register_keras_serializable`, which registers objects `get_config` and `from_config` methods. 


#### Serializer specification 
To define the serializer for a class, create a method named `saved_model_serializer` which returns a `SaveSpec`. When creating the SavedModel, this function is first called on the saved object, forming the root of the object graph. The returned SaveSpec produces the following proto in the SavedModel:

```
SavedObject {
  user_object: {
    identifier: "{package}.{name}"
    metadata: "SaveSpec.metadata"
  },
  children: [SaveSpec.children]
}
```

The object graph in the SavedModel is constructed by serializing the children returned by `SaveSpec` and subsequently  their children and so on. 

The `tf.train.Checkpoint` stored in the SavedModel `variables/` directory uses the same object graph. This has important implications when when loading the SavedModel.


**Specification**
```python
def saved_model_serializer(
    self, save_options=tf.saved_model.SaveOptions, serialization_cache=dict()):
```
Keyword arguments:

* save_options: A `tf.saved_model.SaveOptions` object that is created by the user and passed to the saving function.
* `serialization_cache`: A dictionary passed to every serializer function. Should only be used in advanced cases (e.g. when serializing a Keras layer/model, its call function is included in `SaveSpec.children` as a `tf.function` object. This function is saved to the cache, so that when serializing a Keras model, if the model calls one of its layers, then it directly calls the serialized `tf.function`. The result is that instead of storing the layer call ops twice in the SavedModel, the layer ops are stored once).

Returns: This function must return a `SaveSpec`.


#### SaveSpec
Contains attributes that should be saved to the SavedModel.
* metadata: A string
* children: A dictionary that maps strings to (possibly nested) TensorFlow functions or objects (e.g. other `tf.Module`, `tf.keras.Layer`, `tf.Variable`, etc.). If left as `None`, the children will default to dependencies that are tracked by the `Trackable` class. The user can also use the default `Serializer().serialize(obj)` to list all of the tracked children. 


#### Deserializer specification
The deserializer method uses the SavedModel proto (presented as a `LoadSpec` object) to recreate the originally saved object. 

Serializers are called recursively for each of the children returned in the `SaveSpec`. Likewise, deserializers can be called recursively to rebuild each of the children in the `LoadSpec` by using the `LoadSpec.deserialize(load_spec)` function. 

Also noted in the serializer section, the checkpoint in the SavedModel uses the object graph defined by the serializer function. If the deserialized object doesn't maintain the same structure (e.g. by changing the names of the children), then it is necessary to recreate the checkpoint structure and pass the checkpoint to `LoadSpec.set_checkpoint`. The [Example](#Example) (version 2) demonstrates this.


**Specification**
```python
@classmethod
def saved_model_deserializer(cls, load_spec=LoadSpec):
```
Keyword arguments: load_spec: A `LoadSpec` object.

Returns: A deserialized object.


#### LoadSpec
Properties
* `identifier`: String identifier stored in the proto. This string is used to determine which deserializer method to call from the ones registered using `register_serializable`.
* `version`: Int producer version.
* `metadata`: String metadata stored in the proto.
* `children`: Dictionary mapping strings to (possibly nested) LoadSpecs.

Methods
* `set_checkpoint(checkpoint)`: If the object graph stored in the SavedModel is incompatible with the deserialized object, then the user should create a tf.train.Checkpoint that matches the checkpoint graph, and call set_checkpoint. An example is shown below.
* `deserialize(load_spec)`: Can be called to deserialize child `LoadSpec` objects.


### Alternatives Considered

**Design: Should the user have full control of the object graph?**

Considering the question of which object graph modifications to allow -- additions to the object graph are certainly allowed. It does not conflict with the existing graph so there are no noticeable issues (other than checkpoint keys being different when saved from tf.saved_model.save vs tf.train.Checkpoint.save).

The issue to consider is whether to allow overwrites and deletions. One obvious downside is that users may accidentally modify the object graph in a bad way (e.g. deleting a variable that's used in a function). This can be remedied by providing a method that retrieves all existing dependencies which the user can call in the serializer. 

Another downside is that arbitrary graph modifications can make checkpoint restoration difficult. There is no good solution to this, but we already assume that object structures do not change much for the sake of regular training checkpointing.

The benefit of allowing overwrites and deletions is that this allows the user to shave the object graph into a bare minimum variables + function holder that saves and loads fast. 

The proposed changes allow **all modifications** to the object graph.


### Performance Implications
* I am adding benchmarks that time how long it takes to save and load the Keras application models which will measure any performance changes.


### Platforms and Environments
* Although the resulting SavedModel will not change from the proposal, adding the ability to create custom serializers and deserialiers may complicate moving the SavedModel implementation to wrap the C API. 

## Questions and Discussion Topics

* Should this impact `tf.train.Checkpoints`? i.e. Should we allow tf.Checkpoint to save and restore checkpoints using the dependencies listed in the custom serializer?
