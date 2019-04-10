# SavedModel Save/Load in 2.x

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Allen Lavoie (allenl@google.com), André Susano Pinto (andresp@google.com), Arno Eigenwillig (arnoegw@google.com), Rohan Jain (rohanj@google.com) |
| **Sponsor**   | Karmel Allison (karmel@google.com) |
| **Updated**   | 2019-02-28                                           |


## Objective

Provide an API for serialization/deserialization in TF-2.0 that supports both serving and reuse use-cases.

TensorFlow 2 will include neither `tf.Session` nor `tf.train.Saver` as public symbols. Current SavedModel export and import workflows rely heavily on these symbols. This document proposes adding `tf.saved_model.save` and `tf.saved_model.load` as 2.x-compatible replacements. These symbols are mentioned at a high level in [the Functions, not Sessions RFC](https://github.com/tensorflow/community/pull/20).

## Motivation and introduction


### Serialization use-cases

There are several reasons to serialize state in programs which use TensorFlow, each with different requirements. This proposal addresses serving and sharing.


#### Training

When writing a new model, the first need users run into is a way to continue training after restarting a program. The original code still exists, although it may be modified slightly.

[tf.train.Checkpoint](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint) handles this case. TensorFlow state referenced by Python objects is recorded and can be restored later in time. In order to be used the user needs to have access to the code that constructs the objects.


#### Serving

Once trained, users may want to serve requests using their model. Ideally for this use-case the representation should be a hermetic and stateless black box, usable through a stable interface, and with minimal dependencies.

SavedModel as a format satisfies this use-case. Various APIs for creating SavedModels exist in TensorFlow 1.x (SavedModelBuilder, Estimator, others). Generally the only interface to the model needed for this use-case is a signature specifying what goes into the model and what comes out. This use-case is well supported in TF 1.x and will continue to be supported under this proposal.


#### Sharing

Beyond serving, users may want to reuse parts of a trained model in building new models. SavedModel allows saving the computation together with its pre-trained weights, without depending on the model definition in Python and its particular framework. This helps reproducibility (say, for results reported in a paper) and reuse (say, for importing a pre-trained embedding into a model that uses it on a new task). While serving and reproducibility call for a complete model, reuse typically concerns part of a trained model and its composition. That means loading the SavedModel must restore enough Python state to allow building on top of it.

These use-cases are not well addressed in core TensorFlow 1.x APIs. Export APIs have been complicated by concerns relevant to the serving use-case, and not much time has been spent on usability for re-importing models into Python. TensorFlow Hub has solved this (esp. the import workflow) for sharing graphs, but this needs significant redesign for TF 2.x in light of [the Functions, not Sessions RFC](https://github.com/tensorflow/community/pull/20).


### Goals



*   Support use-cases covered by existing SavedModel export/import APIs and existing Hub module APIs: serving and sharing
    *   Export a smaller part of a full model.
    *   Import a part into a larger model.
    *   Fine-tune an imported part (requires defining "trainable" set of variables, running update ops for batch normalization, exposing regularization losses).
    *   Re-export an imported part (including its new fine-tuned state).
    *   Programmatically inspect signatures/functions
    *   Import state once but data flow using that state multiple times
*   Export to SavedModel in a way that is compatible with existing serving infrastructure. Extensions to the format may be required for 2.x compatibility, but existing loading procedures will continue to work.
*   Reimport TensorFlow functionality of exported objects back into Python
    *   With minimal special casing for Keras types, support a SavedModel implementation of `tf.keras.Model.save` and `tf.keras.model.load_model`. Details of this special casing will be left to another document.
    *   Exporting and importing should be idempotent (reimported representations are saveable)
*   Importing existing SavedModels into Python in TensorFlow 2.x. This will share the `tf.saved_model.load` symbol, but will lack any object structure.


### Non-goals



*   Export arbitrary SavedModels. Use-cases will be covered. For example SavedModel supports multiple MetaGraphs, but the APIs proposed here may only ever export SavedModels with a single MetaGraph.
*   Usable Python interfaces for all symbols in the TensorFlow API on import. Everything will of course be usable when graph building, but objects may not have many features when imported back into Python. The set of types with "nice" import representations is expected to increase over time.


### Terminology



*   A _signature_ identifies inputs and outputs for a computation, roughly the feeds and fetches of a single `session.run` call in TensorFlow 1.x.
    *   In a SavedModel or MetaGraph, SignatureDefs identify input and output tensors in the GraphDef, possibly overlapping.
*   _Concrete function graph_: A subgraph with a single signature which TensorFlow can execute natively to compute Tensor outputs from Tensor inputs.
*   _Polymorphic function_: A Python callable that encapsulates several concrete function graphs behind one, more natural API.  For example it may use non-Tensor arguments to dispatch between the concrete graphs that compute outputs from the Tensor arguments.


*   _FunctionDef:_ A protocol buffer representing a concrete TensorFlow function.
*   _FunctionDefLibrary_: A protocol buffer containing multiple FunctionDefs and their gradient functions.
*   _GraphDef_: A serialized "v1-style" TensorFlow graph. Includes a FunctionDefLibrary.
*   _MetaGraph:_ A GraphDef + training checkpoint. Contains fields for signatures and other metadata, although these are often blank.
*   [SavedModel: A collection of MetaGraphs with additional assets](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md). Signatures and other MetaGraph fields are always filled.


*   _Checkpointable_: A mixin which manages dependencies between objects. Checkpointable objects have named dependencies on other checkpointable objects. Most TensorFlow objects already participate in this scheme.
*   _tf.train.Checkpoint_: A checkpointable object with a save() method, used for writing training checkpoints. Used in several examples in this document as an easy way to make a checkpointable object, since there are no plans for a separate "Checkpointable" symbol.


*   _Resource_: A data type (dtype) for tensors which point to state. Used to implement variables and tables. In TensorFlow 2.x, resources persist as eager tensors attached to a single Python object (e.g. a `tf.Variable` object), sharing a lifetime with that object. Operations take resource-dtype tensors in order to read or mutate the pointed-to state. The [RFC for variables in 2.x](https://github.com/tensorflow/community/blob/master/rfcs/20180817-variables-20.md) has more detail.
*   _Capture_: An implicit tensor input to a function, typically a resource. Variables and other resources which are used or created inside a function are not owned by that function, and are instead lifted out as eager tensors. When a function is called, these tensors are automatically collected and passed in.


## Design Proposal

The `tf.saved_model` additions proposed here handle serialization of Python objects, attributes and functions as a graph of objects, variables, resources and functions backed by polymorphic TensorFlow functions. These objects can be used without access to the original code. It can also store higher level information that allows the objects to be reconstructed assuming a factory of "revivable names"->"revivable classes".  

New user-facing symbols will be `tf.saved_model.save` and `tf.saved_model.load`. These are mentioned at a high level in [the Functions, not Sessions RFC](https://github.com/tensorflow/community/pull/20).


```
# Serialize objects reachable from root into a SavedModel.
tf.saved_model.save(
  obj : Checkpointable,
  export_dir : str
  signatures=None : map[str->Function])
```



```
# Load the root object from a SavedModel.
tf.saved_model.load(
  export_dir: str,
  tags=None : list[str]) : Checkpointable
```


 The remainder of this section defines their behavior.


### Serialization primitives

This section describes with examples the basic primitives that are needed to load a SavedModel for reusing it in python without depending on its original code. Note that in many cases, reviving the original class would provide much more functionality than what can be serialized.

Those primitive types are: Variable, CheckpointableBase, PolymorphicFunction, TrackableResource, and any nest of those, including nesting of Checkpointable attributes in an object.

Individual examples:

| Save                                         | Load                                |
:----------------------------------------------|:------------------------------------|
| `obj.x = tf.Variable(...)`                   | `type(obj.x) => tf.Variable`        |
| `obj.x = tf.Checkpoint(...)`                 | `type(obj.x) => CheckpointableBase` |
| `obj.x = tf.function(...)`                   | `type(obj.x) => PolymorphicFunction`|
| `obj.x = tf.lookup.Table(...)`               | `type(obj.x) => TrackableResource`  |
| `obj.x = [tf.Variable(), tf.Variable(), ...]`| `type(obj.x) => [tf.Variable]`      |

As a preview to the rest of this design, consider the following rough outline of how this would handle a basic text embedding model. First user1 has code that defines a model object that is a `CheckpointableBase` that has its resources (variables, tables and assets files) declared as members. Additionally the user took care to annotate the "embed" method with `tf.function` decorator and provide a signature (not providing a signature is also possible, [but leads to many technicalities](#finding-functions-and-methods)).


```
class Model(tf.Module):

  def __init__(self, vocab_file, dim):
    # The table object tracks its asset file automatically
    self.table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file,
        num_oov_buckets=1,
    )
    vocab_size = self.table.size()
    self.embeddings = tf.Variable(tf.random.uniform(shape=[vocab_size, dim]))
   
  def tokenize(self, sentences):
    sparse_words = tf.string_split(sentences)
    token_ids = self.table.lookup(sparse_words)
    return token_ids
  
  @tf.function(signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
  def embed(self, sentences):
    token_ids = self.tokenize(sentences)
    combined = tf.nn.embedding_lookup_sparse(self.embeddings, token_ids, None)
    return combined

root = Model("/tmp/vocab", 64)
tf.saved_model.save(root, "/user1/tmp/model")
```


Another user2 could then inspect and call into parts of this SavedModel without having access to the original code. Note how initialization of resources has happened during `load` without user action.


```
obj = tf.saved_model.load("/user2/download/model")
obj.embed(["hello world"]) => <tf.Tensor: shape=(1, 64), dtype=, numpy=...>
obj.embeddings => <tf.Variable...>
obj.table => <TrackableResource> (the method .lookup would not be present).
```



### Finding functions and methods

`tf.saved_model.save` will save the same set of stateful objects as `tf.train.Checkpoint` would given the same root object. In addition, `save` will iterate over each checkpointable object's attributes and find functions (`self.f = tf.function(...)`) and methods (`@tf.function`-decorated). 

Collected functions and methods are polymorphic, having one or more "concrete" functions, each corresponding to a [FunctionDef](https://github.com/tensorflow/tensorflow/blob/b7a2128b28a2755dba2a51d065c429c0cfb0a0f5/tensorflow/core/framework/function.proto#L25) with tensor inputs and outputs. For methods, we look up the function definitions corresponding to the object we're saving. After that they're similar to attribute-assigned functions.

Polymorphic functions with no signature specified, and which have not been called, have zero concrete functions associated with them. Saving an object with such a polymorphic function will raise an exception.

Each concrete function may reference variables or other stateful objects. Any variables referenced this way imply a dependency of the function's object on the variable. If no transitive dependency exists at export time, an exception will be raised. An automatic dependency scheme may be considered if there is a strong use-case. Such a scheme would be a backwards-compatible addition.

Concrete functions corresponding to signatures which can not be serialized (see [Serialization formats](#polymorphicfunctions)) will raise an exception on export.

#### Functions

```
has_fns = tf.Module()
has_fns.v = tf.Variable(1.)
has_fns.a = tf.function(lambda x: x + has_fns.v + 1.)
has_fns.b = tf.function(lambda x: x + has_fns.v + 2.)
has_fns.c_dep = tf.function(lambda x: x + 3.)
has_fns.c = tf.function(
    lambda x: has_fns.v + has_fns.c_dep(x),
    input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
has_fns.a(tf.constant(2.))  # 4.
has_fns.python_attribute = 12  # Not exported
# Error: exporting a function without a trace (for b)
# "c" can be traced for export since it has an input signature specified
# "c_dep" is traced for export as a result of "c"'s trace
tf.saved_model.save(has_fns, "/tmp/fns")
has_fns.b(tf.constant(3.))
tf.saved_model.save(has_fns, "/tmp/fns")
```



```
imported_fns = tf.saved_model.load("/tmp/fns")
print(imported_fns.v)  # 1.
imported_fns.a(tf.constant(1.))  # 3.
imported_fns.b(tf.constant(1.))  # 4.
imported_fns.c(tf.constant([1., 2.]))  # [5., 6.]
imported_fns.c_dep(tf.constant([1.]))  # [4.]
imported_fns.c_dep(tf.constant(1.))  # Error: no trace for scalar inputs

imported_fns.python_attribute  # Attribute error; Python attributes are not saved

# Exported SavedModels are also usable as training checkpoints
training_checkpoint = tf.train.Checkpoint(v=tf.Variable(2.))
training_checkpoint.load("/tmp/fns")
print(training_checkpoint.v)  # 1.
training_checkpoint.a  # Attribute error; functions are not restored
```



#### Methods

`tf.function`, when decorating a method, allows variable creation for each new `self` argument. When saving and restoring objects, methods behave like functions assigned to attributes (with the `self` argument bound).


```
class Net(tf.Module):

  def __init__(self):
    self.y = None

  @tf.function
  def add(self, x):
    if self.y is None:
      self.y = tf.Variable(2.)
    return x + self.y

net = Net()
net.add(3.)  # Variable created
net.add([3.])  # A second concrete function
tf.saved_model.save(net, "/tmp/net")
```



```
imported_net = tf.saved_model.load("/tmp/net")
print(imported_net)  # <Checkpointable object at 0x7faddc343278>,
                     # type not preserved by default.

print(imported_net.y)  # 2.

imported_net.add(3.)  # 5.
imported_net.add([3.])  # [5.]

imported_net.y.assign(3.)
imported_net.add(3.)  # 6.
imported_net.add([3.])  # [6.]
```



### Python objects and nests in function signatures

Limited serialization of Python objects will support common non-Tensor types in function signatures.


```
@tf.function
def f(x, training):
  return x if training else 2.
f(-1., training=True)
f(-1., training=False)
obj = tf.train.Checkpoint(f=f)  # save() exports objects, so we wrap f
tf.saved_model.save(obj, "/tmp/f")
```



```
imported = tf.saved_model.load("/tmp/f")
imported.f(10., training=True)  # 10.
imported.f(10., training=False)  # 2.
```


Lists, tuples, namedtuples, and dictionaries may be nested in function input signatures and return values. For example:


```
@tf.function
def g(x):
  return [x[0] + 0.1, x[1]["a"] + 0.2]
print(g((tf.constant(1.), {"a": tf.constant(2.)})))  # [1.1, 2.2]
obj = tf.train.Checkpoint(g=g)
tf.saved_model.save(obj, "/tmp/g")
```



```
imported = tf.saved_model.load("/tmp/g")
print(imported.g((tf.constant(-1.), {"a": tf.constant(-2.)})))  # [-0.9, -1.8]
```


 

See [input signature serialization](#polymorphicfunctions) for details. Additional restrictions are placed on functions which will be used as SavedModel signatures, `save(obj, signatures=...)`, since these must be callable from the C++ loader API.


### Specifying signatures

`tf.saved_model.save` will take an optional argument specifying which function/methods will be recorded as "signatures" in the SavedModel (allowing them to be used when serving for example). These functions must have input signatures specified, either when the `tf.function` is created or by calling `get_concrete_function`. SignatureDefs and corresponding function call ops will be generated for each signature function in the SavedModel.


```
class Net(tf.Module):

  @tf.function(input_signature=tf.TensorSpec([None, 5], tf.float32))
  def infer(self, x):
    return x

net = Net()
tf.saved_model.save(
    net, "/tmp/serve1", 
    # One SignatureDef added with the default serving signature key.
    signatures=net.infer)

# Or if multiple signatures should be exported:
signature_functions = {
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    net.infer,
    "other_signature_key": ...)
}
tf.saved_model.save(net, "/tmp/serve2", signature_functions)
```


Input signatures need not be specified when using `tf.function` decorators, which would generally require them to be static for the whole Python program. Instead, a signature may be specified at export time. If no corresponding trace exists in the function cache, a new trace will be created.


```
class Net2(tf.Module):

  @tf.function
  def infer(self, labels, training, x1, x2):
    if training:
      return labels, x1, x2
    else:
      return labels, x1 + 1., x2 + 1.

net = Net2()
tf.saved_model.save(
    obj=net, export_dir="/tmp/serve1", 
    signatures=net.infer.get_concrete_function(
        tf.TensorSpec(None, tf.int64),
        training=False,
        x1=tf.TensorSpec([None, None, 3], tf.float32),
        x2=tf.TensorSpec(None, tf.float64)))
# Then a serving request could specify:
#   {"labels": [0, 1], "x1": [[[1., 2., 3.]]], "x2": 0.}
# and get a response like
#   {"output_1": [0, 1], "output_2": [[[2., 3., 4.]]], "output_3": 1.}
# The "training" argument is important for picking which trace to serve with, 
# but does not affect the signature.
```


Non-`Tensor` arguments to functions used to generate signatures are fine (e.g. `training=False`). The `Tensor` arguments are the only ones serving APIs will know about, and these may not be nested inside lists, tuples, or dictionaries (since there is no way to specify nested `Tensor` arguments when serving). Arguments are identified by name when importing a `SavedModel` for serving, so there is no ambiguity even if non-`Tensor` Python arguments are interspersed with `Tensor` arguments.

SavedModel SignatureDef outputs require each output to have a name. If a single dictionary is returned from the traced function, the string keys will be used to name outputs. Otherwise signature functions must return a flat list of `Tensor`s, and the outputs will be numbered ("output_1", "output_2", ...). Flattening outputs would be trivial before numbering them, but serving APIs would have no way to reconstruct the structure.

Any return structure other than a dictionary with string keys or a flat sequence of Tensors from a function used to generate a signature will raise an exception on export. Note that this limitation applies only to serving functions; functions attached to objects but not specified as signatures may have other output patterns.

#### Protos in serving signatures, re-exporting

The above examples have only numeric `Tensor`s in their exported signatures. If we now decide that `tf.Example` protos are a better way to pass data to our model, we can add parsing functionality by re-exporting.


```
net = tf.saved_model.load("/tmp/serve1")  # From the previous example

@tf.function(input_signature=tf.TensorSpec([None], tf.string))
def _parse(serialized):
  parsed = tf.parse_example(
      serialized,
      features={"labels": tf.FixedLenFeature(shape=[10], dtype=tf.int64),
                "x1": tf.FixedLenFeature(shape=[10, 3], dtype=tf.float32),
                "x2": tf.FixedLenFeature(shape=[10], dtype=tf.float64)})
  return net.infer(**parsed)

net.infer_from_proto = _parse
tf.saved_model.save(obj=net, export_dir="/tmp/serve_proto", 
                    signatures=net.infer_from_proto)
```


Of course re-exporting is not a necessary step; adding a proto parsing head before the first export would be the common case.

#### Implicit default serving signature

When unambiguous and not specified with an explicit `signatures=` argument, a default serving signature may be inferred automatically. For example if the object passed to `tf.saved_model.save` has only one `@tf.function` attached and that function has a signature. Symbolic `tf.keras.Model` objects, but not subclassed `Model`s in general, similarly have signatures and so may provide an automatic default serving signature.

#### Imported representation of signatures

The `.signatures` attribute will be reserved on objects which are saved and restored. On `tf.saved_model.load` it will contain an immutable mapping from signature keys (e.g. `"serving_default"`) to the functions used to implement each signature. This supports introspection, allowing users to import a SavedModel and run exactly the computation a serving API would run.

On `tf.saved_model.save`, signatures in the exported SavedModel come from the first available of the following sources:



1.  The `tf.saved_model.save(..., signatures=...)` argument
1.  The `.signatures` attribute of the exported object
1.  A set of heuristics to determine a default signature, for example exporting a functional Keras model or searching for `@tf.function`-decorated methods with a signature. This may fail, in which case no signatures will be exported.

For example:


```
class Net(tf.Module):

  @tf.function
  def infer(self, x):
    return x

net = Net()
net(tf.constant(1.))
tf.saved_model.save(
    net, "/tmp/serve",
    signatures=net.infer.get_concrete_function(tf.TensorSpec(None, tf.float32)))
```



```
loaded = tf.saved_model.load("/tmp/serve")
loaded.signatures["serving_default"](x=tf.constant([[1.]])) 
#   -> {"output_0": <tf.Tensor [[1.]], dtype=float32>}
tf.saved_model.save(loaded, "/tmp/serve1")  # Contains the same signature
```


Attempting to save an object with a `.signatures` attribute containing something other than an immutable signature mapping (for example created by `tf.saved_model.load`) will raise an exception. This prevents accidentally ignored signatures when the attribute is modified and the argument passed. However, internal APIs may make use of the attribute to provide a user-overridable default.

### Format and API compatibility

The Python APIs proposed in this document will be targeted at a TensorFlow 2.x environment. They will not be tested with `Graph` and `Session`, and so will not be usable from TensorFlow 1.x without eager execution enabled.

SavedModels exported from TensorFlow 1.x APIs [will be importable using the proposed APIs](#importing-existing-savedmodels). SavedModel signatures will be available as callable Python functions. This includes the functionally-equivalent `Estimator.export_saved_model`, which will still be available in TensorFlow 2.x.

SavedModels exported from the proposed APIs will be importable using TensorFlow 1.x APIs, including TensorFlow Serving and the C++ loader API. The available computation will be the exported signatures, `tf.saved_model.save(..., signatures=...)`.


#### Importing existing SavedModels

Using `tf.saved_model.load` on a SavedModel exported from a TensorFlow 1.x API will import each SignatureDef as an individual  `tf.compat.v1.wrap_function` object. This will follow the [same style as for signatures exported using tf.saved_model.save](#imported-representation-of-signatures), with a `.signatures` attribute of the root object containing a mapping from signature keys to `wrap_function` objects. Another attribute will contain variables.

If multiple MetaGraphs exist in the SavedModel, the `tf.saved_model.load(..., tags=...)` argument must be specified and must match exactly one MetaGraph. Only one MetaGraph will be loaded per call to `tf.saved_model.load`.

Loading for a MetaGraph will follow the existing procedure for the C++ and Python loader APIs, a checkpoint restore followed by the main op running with asset paths "fed". This procedure will be wrapped in its own `wrap_function` object and executed when `tf.saved_model.load` runs.

Not all existing SavedModels will be loadable to start. Some known tricky issues:



*   Reference variables (as opposed to resource variables) do not exist in 2.x, so SavedModels using these will require rewriting
*   Control flow using collections (while_loop/cond) will need some graph rewriting to import correctly


### Resources and assets

State (a variable, table, etc.) is represented in TensorFlow 2.x using a resource-dtype eager tensor. Such state is uniquely associated with a Python object (e.g. a `tf.Variable`), and deletion/collection of the Python object triggers deletion of the resource (DestroyResourceOp). Functions reference state through special "capture" inputs, with a resource-dtype placeholder in the function body which is fed the eager resource tensor on each function execution, giving the function a temporary reference to the resource. This is true even if a resource is created while tracing the function, in which case the resource handle is lifted out into the eager context before being captured.

Managing resources involves two operations: the creation of the resource-dtype tensor (e.g. `VarHandleOp`) and its initialization (e.g. `AssignVariableOp`). Both of these operations may be executed eagerly, but an exported SavedModel needs to include the operations themselves. A `TrackableResource` type will associate resource-dtype eager tensors with functions to create and initialize them. On export, resource tensors will be collected through object dependencies and matched to the captured inputs of exported functions.

Objects which reference external files that should be included in the SavedModel will indicate these asset paths by subclassing `TrackableAsset`, serving the same purpose as the v1 assets collection. Paths referenced this way will be copied into the SavedModel's assets directory on export.

`TrackableResource` and `TrackableAsset` may be used together, creating a resource which is initialized from an asset.


```
class TrackableAsset(CheckpointableBase):
  
  def __init__(self, asset_path):
    # This variable will be initialized using the absolute path to a resource 
    # on SavedModel restore. It will not be checkpointed.
    self._asset_path = tf.Variable(asset_path, dtype=tf.string)

  @property
  def asset_path(self):
    return self._asset_path

class TextFileInitializer(TrackableAsset):

  def __init__(self, asset_path):
    # Lets object-based saving track asset paths. This value is recorded in an
    # AssetFileDef in the SavedModel.
    TrackableAsset.__init__(self, asset_path)

  def initialize(self, table):
    gen_lookup_ops.initialize_table_from_text_file_v2(
        table.resource_handle,
        self.asset_path,
        ...)

class Table(TrackableResource):

  def __init__(self, initializer):
    self._initializer = initializer
    self._track_checkpointable(initializer, name="_initializer")

  @tf.function(input_signature=())
  def create_resource(self):
    Return gen_lookup_ops.hash_table_v2(...)

  @tf.function(input_signature=())
  def initialize(self):
    self._initializer.initialize(self)  # May capture an asset variable
```


In the SavedModel protocol buffer, AssetFileDefs will have a restore function taking the full asset path which assigns to the asset path variable. A serving API expecting a 1.x-style SavedModel will feed values for the AssetFileDef Tensors and run the referenced function call op, initializing the variable. The 2.x SavedModel import API will run the function directly. Asset variables will then be captured inputs to `TrackableResource` initializer functions.

`TrackableAsset` and `TrackableResource` objects will be recreated by the Python SavedModel import routine to make reexport possible while preserving asset paths.


### Devices

Functions will be traced outside of any device scope, and we will rely on the placement of the `PartitionedCallOp` for a "default" device. So no special treatment is needed to switch devices between export and import: just call the imported function in a device scope.

Device placements specified within the function body will be hard-coded in the SavedModel, and aside from library code needing to place things on the CPU, we should discourage `tf.device` within graph functions so devices aren't hard-coded for export.

This means that the Python implementation of polymorphic functions (`tf.function`) should not specialize a function's trace based on the device stack where it is called. Instead, it should look up the graph function to call without regard to device placement, tracing outside a device scope if a new graph function must be created. Then the function call itself will be within the enclosing device scope.

This does not protect users who use device-specific operations (cuDNN) or layouts which are only supported on one type of device. Such SavedModels may only be usable on one type of device.


#### Distribution strategy integration

A user should eventually be able to export a single-machine computation and import the SavedModel under a `DistributionStrategy` scope. An initial implementation will simply hard-code device placements when a distribution strategy is active, meaning that the `DistributionStrategy` used on export will be the only usable configuration on import.

Options for allowing single-device models to be imported with a `DistributionStrategy` include recording and saving attribute accesses for variable objects (`assign_add`, `assign`, `read`, etc.) and rewiring the graph on import, or supporting templated functions which can be specialized to access variables in a certain way. Solving this will be crucial to support sharing SavedModels for reuse (see [Sharing](#sharing) above).


### Custom revived types, stateful Python attributes

By default, imported objects will have unique types inheriting from `Checkpointable`. Objects of these types will have `tf.function` callables in their attributes, along with attributes for variables and other checkpointable dependencies.

In some cases, Python values are important parts of an object's API. For example the `tf.keras.backend.learning_phase() `global is a Python integer which affects the behavior of `tf.keras.Layer` methods. Such Python values must already be part of a polymorphic function's cache key for correct tracing regardless of export/import. This will be implemented by a "`tf.function`-compatible" method which explicitly takes all of its inputs as arguments and returns its outputs (e.g. taking a learning phase and returning regularization losses). There will be a way to register a custom base class for a revived type which has arbitrary Python attributes and convenience wrappers for the `tf.function`-compatible TensorFlow methods. The registration will be keyed to a unique string which must be the same at import time as it was at export time.

Registrations for revived types will initially be considered implementation details used to support saving and restoring TensorFlow types, but may eventually be exposed as public APIs.


```
class HasPython(tf.Module):

  @tf.function
  def do(self, x, learning_phase):
    if learning_phase == 0:
      return x
    else:
      return x + 1.

  def __call__(self, x):
    # Python methods which call TF methods are fine, but need a custom revived
    # type.
    return self.do(x, learning_phase=tf.keras.backend.learning_phase())

has_python = HasPython()
tf.saved_model.save(has_python, "/tmp/haspython")
```



```
imported = tf.saved_model.load("/tmp/haspython")
tf.keras.backend.set_learning_phase(1)
imported(1.)  # 2.
```



### Optimizers

For the same reasons slot variables are special cased in `tf.train.Checkpoint`, optimizers will require some special-casing when restored from a SavedModel.

Restored optimizers will be generic `Optimizer` instances with their behavior defined by the SavedModel, and have their slot variables restored and mapped from the right recreated variable objects (this will be the main `Optimizer`-specific special casing). Non-slot variables will be handled as for any other objects. The functionality in `_prepare`, `_resource_apply_dense`, `_resource_apply_sparse_duplicate_indices`, ..., `_finish` will all be traced, with the restored `Optimizer` using the implementations from the SavedModel.

Restored `Optimizer`s will not be limited to optimizing variables in the imported model. The exported signatures will allow any gradient shape, which should be no problem for the ops used to implement core optimizers. Tracing a slot variable lookup doesn't make much sense from the perspective of a TensorFlow graph (or would limit the `Optimizer` to working with variables which existed at export time), so some refactoring may be required to create pure functions which take (primary, slots, gradient) rather than taking (primary, gradient) and looking up slot variables. Then the `RevivedOptimizer` Python type would be responsible for looking up the correct slot variables.


### Initialization graphs

On import, objects are restored and variables set to their checkpointed values. However, the imported types will be usable, exposing initialization graphs.


```
class Net(tf.Module):

  def __init__(self, units):
    self.units = units
    self.var = None
    self.built = False

  def build(self, x):
    self.var = tf.Variable(x * tf.ones(self.units))

  @tf.function
  def do(self, x):
    if not self.built:
      self.build(x)
    return x + self.var

net = Net(5)
net.do(1.)
net.var.assign([1., 2., 3., 4., 5.])
tf.saved_model.save(net, "/tmp/net")
```



```
imported_net = tf.saved_model.load("/tmp/net")
assert list(imported_net.var.numpy()) == [1., 2., 3., 4., 5.]
net_from_class = type(imported_net)()  # No Tensor constructor arguments
# net_from_class.var is uninitialized
net_from_class.do(2.)
assert list(net_from_class.var.numpy()) == [2., 2., 2., 2., 2.]
```


Constructing a new object from a revived type will also construct new objects for any dependencies. To be usable, the pre-export object associated with this type must not have had a transitive dependency on any function or method unless it also had transitive dependencies on all that function's referenced variables. So for example an object referencing a checkpointable list of functions which reference its variables may be constructed, but the list itself may not be constructed on its own.

Unless `__init__` is decorated, revived objects will not take constructor arguments. Constructing a new object from a revived type creates uninitialized variables of the same shape and dtype as the revived object with that type, and calling the method which created a variable (before export) initializes it. Variable initialization will be automatic and idempotent, as implemented in [tf.function](https://github.com/tensorflow/tensorflow/blob/4a5126674c9c3086a2c38c78126d0e190cb93a61/tensorflow/python/eager/def_function.py#L504).

A `@tf.function`-decorated `__init__` before export requires corresponding tensors be passed to the constructor of the revived type. Dependencies are constructed in an uninitialized state even if they have tensor arguments to their `__init__` methods, with the understanding that initialization will be included in the trace of the method of the parent object which created the depended-on object before export (i.e. using a traced constructor to create a depended-on object outside of a traced method will result in uninitialized variables).


## Prioritization



1.  (Done) Basic export to SavedModel for serving. Requires function(s) to be specified for signatures. Marked as experimental to avoid creating a format that won't import correctly into Python once that's implemented.
1.  Import of existing SavedModels (v1 compatibility)
1.  Python object re-import
    1.  Serialized representation for polymorphic functions/methods
    1.  Collect and save functions and methods attached to all objects (not just those specified explicitly as signatures)
    1.  A function-based SaverDef to allow a sessionless restore
    1.  Import generic objects back into Python with functions/methods and variables.
    1.  Mark as not experimental. Export no longer requires signatures.
1.  Decorate call methods (and loss lambdas, etc.) of Layers so that the TensorFlow parts of their functionality can be referenced individually when deserialized.
1.  Python wrapper support for imported objects, used to make Keras Layers and Models import nicely.
1.  Distribution strategies integration: allow imported models to be run on multiple devices (without requiring them to have been saved with a distribution strategy set).
1.  Assets, any necessary changes to immutable tables to work with the tweaked asset handling.
1.  Import wrappers for other Python types (e.g. Optimizers)
1.  Make imported types work for creating new objects using saved initialization graphs


## Detailed Design


### Restore procedure

At a high level, every time an `Operation` or `Tensor` is referenced directly in the SavedModel format, it will be helpful to add a function name for use when loading the SavedModel into a sessionless TF 2.x environment. The existing op/tensor references will then reference function call ops in the MetaGraph, telling existing SavedModel loading infrastructure how to call the functions.

The loading procedure for session-based APIs will be the same as it is today, relying on ops in the SavedModel's MetaGraph. Loading into a 2.x-style context without any sessions uses the following procedure:



1.  Python objects are created from the CheckpointableObjectGraph proto with the dependency structure they had before export. Every created object inherits from CheckpointableBase, but many will have more specific types.
    1.  Objects with custom types registered will be re-created using those types
    1.  Variables are recreated as `tf.Variable` objects. This will use the VariableDef as a serialization format for variable attributes, but unlike the existing `tf.Variable.from_proto` it will always create a new resource handle rather than looking up an operation in a Graph. They will be uninitialized to start.
    1.  Resources with no more specific type will be revived as `TrackableResource` objects, which ensures that the information required to re-create and initialize their resources is preserved for re-export.
    1.  Objects which inherited from `TrackableAsset` before serialization will be revived as that type.
    1.  All other objects will be generic Checkpointable objects.
1.  `TrackableAsset` objects have string variables created for them, initialized with the absolute path of the corresponding asset.
1.  Functions for resource handle creation associated with `TrackableResource`s are imported from the SavedModel into the eager context and represented as `Function` objects.
1.   Resource handle creation runs for `TrackableResource` objects. Variables will have already created resource tensors. We create a map from resource tensors in the SavedModel to the newly created eager resource tensors.
1.  Remaining "concrete" functions from the SavedModel are imported into the eager context and wrapped as `Function` objects, with captured resources mapped to their corresponding eager resource tensors.
1.  `Function` objects are gathered into `PolymorphicFunction` objects and assigned to object attributes
1.  Variables are restored from the checkpoint by running the restore function referenced in the SaverDef (through the imported `Function` object)
1.  `TrackableResource` objects have their initializer functions run, which includes for example initializing tables from assets


### Serialization formats

Several protocol buffers will require backwards-compatible additions to support loading without a `Graph`/`Session`.


#### CheckpointableObjectGraph

The existing [CheckpointableObjectGraph](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/protobuf/checkpointable_object_graph.proto) will be augmented with pointers to the SavedModel components necessary to recreate objects. For example objects representing variables will identify their VariableDef, `TrackableAsset`s will identify their AssetFileDef, and `TrackableResource`s will identify handle creation and initialization functions. 

`PolymorphicFunction`s will be nodes in this graph (but without any children), which allows multiple objects to reference the same function.


#### PolymorphicFunctions

Each `PolymorphicFunction` is a list of signatures (indicating input and output formatting) each with a corresponding "concrete" FunctionDef. This format will build on existing Hub work which serializes Python function signatures.

There will be no general pickling, so only a limited set of types will be supported in the signatures of serialized functions. This support will include at least the basic types Boolean, string, integer, float, `None`, TensorShape, and dtype. Nests of container types may involve lists, tuples, dicts, and namedtuples.

For each concrete function in the `PolymorphicFunction`, any Python arguments will be serialized along with indicators specifying tensor inputs. This is necessary to allow selection between concrete functions when the restored `PolymorphicFunction` is called, e.g. `f(..., training=True)` vs. `f(..., training=False)`. Each argument to the function and its output may be an arbitrary nest of the supported Python types and tensors.


#### Save and restore functions (SaverDef)

SaverDef currently names a feed tensor which takes a checkpoint path and an operation to run to save and restore. These fields will continue to exist in the SaverDef and will be filled in by `tf.saved_model.save` so that 1.x-style loader APIs can restore variables from checkpoints.

Two fields will be added identifying save and restore functions (FunctionDefs in the GraphDef's FunctionDefLibrary), each taking scalar string tensors. Call operations for these functions will be referenced by the existing `restore_op_name` and `save_tensor_name` fields.

Each object in the CheckpointableObjectGraph will have save and restore functions for each SaveableObject they export, and these functions will be composed into the SaverDef's save and restore functions. The restored Python objects will re-export these save and restore functions so that loading and saving again is idempotent (and subsets of objects are re-saveable).


#### Variables (VariableDef)

VariableDefs will be used as-is to store variable attributes. The existing `tf.Variable.from_proto` restore logic will not be used to re-create variables when loading into a 2.x context. They will instead be created with new eager resource handles in an uninitialized state, then restored from the checkpoint.


#### Assets (AssetFileDef)

`TrackableAsset` objects will have corresponding AssetFileDef protos in the SavedModel. When loading using a 1.x-style API, the fed filename tensor will be used to assign to the `TrackableAsset`'s variable. The AssetFileDef proto will not change.

## Questions / Discussion Topics

### The object & types of this format are python specific

Are there concerns on the concepts, types used on this format being python specific?

What about trying to load these representations on other languages?

### Imported representations of `signatures=`

How, once a user has specified a `signatures=` argument on export, should that argument be represented on re-import (if at all). One idea is to put them in a special `signatures` attribute of the imported object, another is to use the signature keys (e.g. `serving_default`) as attributes on the imported object directly. There seems to be some consensus that being able to access these signatures on import is important.

### Does tf.saved_model.load() work in graph mode?

Probably not. The compatibility section now makes this explicit.

### Return value of load() for SavedModels with multiple MetaGraphs

TensorFlow 1.x SavedModel load APIs have a tag-based selection system for choosing which MetaGraphs to load. Should `load()` take arguments to replicate this behavior, even if they’re not relevant to `save()`? Or is returning a list and lazily loading MetaGraphs sufficient?

Is returning an unwrapped object in the single-MetaGraph case but a list in the multi-MetaGraph case too surprising? If so, do we need a separate API?

## Design Review Notes

Things that changed since doc was sent out

*   Import representation of signatures: attach to root object? Decided to attach to .signatures attribute (see [Imported representation of signatures](#imported-representation-of-signatures))

Two big issues to discuss



*   Functions in MetaGraph  
    *   e.g. should train/eval/serving be in the same metagraph? What if train graph has a custom op that can't be loaded into the servo?
    *   Is it possible to import a subset of the graph?
        *   If all attributes are imported when the SavedModel is loaded, then this is impossible
        *   Proposal:
            *   already in current design=single recursive export
            *   allow two options during import: entire object vs set of signatures 
        *   Is it possible to filter at load time? yeah, will need to be implemented
    *   Proposal: Export entire object graph -and- v1 SavedModel as separate fields
        *   Two MetaGraphs, shared FunctionDef
        *   Pros:
            *   Allows compatibility with V1 tools (ease transition)
            *   Conceptually consistent for users/library devs: one metagraph=restore, other=serving
        *   Cons:
            *   Duplicate information
            *   Bigger file (there's a 2gb limit?)
            *   Additional complexity for hypothetical situation
        *   Decision: Extend metagraph, v1 will only look at some fields
*   Compatibility story
    *   What happens when a V2 loader runs into a V1 SavedModel?
        *   Proposal: Always return an object that represents the single metagraph (default case)
            *   object has .signatures, .variables attributes
            *   User must specify tags if there are multiple metagraphs (Error out if not specified)
    *   Load V2 into Estimator/Graph mode
        *   Make sure collections are loaded correctly
        *   Implementable, low priority
    *   Load V2 into Keras V2 layer
        *   Should be able to be done, low priority

Distribution Strategy + SavedModels



*   Day 1: Functions are not device hard-coded, not DistributionStrategy friendly
*   Options:
    *   Export Distributed Graph with devices hardcoded
    *   Load inside DistributionStrategy scope
        *   If the variables are created and the forward pass is run in the scope, then the model will be distributed
        *   Look out for bugs in BatchNormalization
        *   Requires variable annotations in SavedModel
            *   Aggregation mode
            *   Sync mode

