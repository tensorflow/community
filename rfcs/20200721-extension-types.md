# Extension Types

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Authors**   | Edward Loper (edloper@google.com) |
| **Sponsor**   | Alex Passos (apassos@google.com)                     |
| **Updated**   | 2020-07-21                                           |

## Objective

This RFC proposes a protocol that can be used to define **_user-defined
object-oriented Python types_** that are supported by TensorFlow APIs.

## Motivation

Object oriented types can make systems more readable, modular, maintainable.
However, most TensorFlow APIs do not currently support user-defined Python
types.  This includes both high-level APIs (such as `Keras`, `tf.function`,
`tf.SavedModel`) and lower-level APIs (such as `tf.while_loop` and `tf.add`).  

This RFC proposes a set of protocols that will allow TensorFlow APIs to handle
user-defined Python types.  A version of this interface is already used
internally to implement several core TensorFlow data types, including
`tf.SparseTensor`, `tf.RaggedTensor`, `tf.data.Dataset`, and
`tf.StructuredTensor`.

At a high level, types supported by this interface can be divided into two broad
categories:

* **_General data structures_**.
  These types are handled by "generic" APIs whose behavior does not depend on
  the value of each object (such as `tf.function`, `SavedModel`, and
  `tf.while_loop`).

* **_Tensor-like types_**, which specialize or extend tf.Tensor.
  Values of these types have a `rank`, a `shape`, and usually a `dtype`.  In
  addition to the "generic" APIs, these types can be handled by Tensor-specific
  APIs (such as `tf.stack`, `tf.add`, and `tf.reduce_mean`).

Examples of user-defined types that could defined or extended with this protocol
include:

**General data structures:**

* `tfp.Distribution`: Encodes a statistical distribution.
* `TensorDigraph`: Encodes the set of nodes and edges in a directed graph.
* `DimensionAlignment`: Encodes a correspondence between two related dimensions
  (e.g., between a `word` dimension and a `speaker` dimension).

**Tensor-like types:**

* `CSRSparseTensor`: A sparsely-encoded tensor that uses the Compressed Sparse
  Row encoding.
* `MaskedTensor`: Pairs a Tensor with a corresponding boolean mask, indicating
  which values are valid, and automatically updates the mask as appropriate
  when used with TensorFlow ops (such as `tf.add` or `tf.reduce_sum`).
* `LabeledTensor`: Pairs a Tensor with a list of axis names, which can be used
  for error detection and reporting, for specifying axes by name, and for
  broadcasting.

## User Benefit

### Object-Oriented TensorFlow

This proposal brings the benefits of Object-Oriented Programming to TensorFlow
users, allowing them to define modular encapsulated data structures that
interoperate with TensorFlow APIs.  This allows TensorFlow models to be defined
at a higher level of abstraction.

### Development outside the main TensorFlow repository

Prior to this proposal, the only way to develop such data structures (e.g.
`SparseTensor`) was to develop them inside the main TensorFlow code base.  This
introduced significant barriers to rapid development, including slow release
cycles, strong backwards compatibility constraints, and centralized API
approval.  By allowing such data structures to be developed outside the main
TensorFlow code base, we hope to make it much easier to experiment with new
types and designs.  If general-purpose types are developed that become
sufficiently mature, we may consider bringing them into the main TensorFlow code
base.

### TensorFlow APIs supported by user-defined types

User-defined types that implement the interface proposed by this RFC will be
supported by the following APIs:

* **Keras**: User-defined types can be used as inputs and outputs for Keras
  `Models` and `Layers`.
* **tf.data.Dataset**: User-defined types can be included in `Datasets`, and
  returned by dataset `Iterators`.
* **Tensorflow hub**: User-defined types can be used as inputs and outputs for
  `tf.hub` modules.
* **SavedModel**: User-defined types can be used as inputs and outputs for
  `SavedModels`.
* **tf.function**: User-defined types can be used as arguments and return
  values for functions wrapped with the `@tf.function` decorator.
* **While Loops**: User-defined types can be used as loop variables in 
  `tf.while_loop`, and can be used as arguments and return values for the
  while-loop's body.
* **Conditionals**: User-defined types can be conditionally selected using
  `tf.cond` and `tf.case`.
* **py_function**: User-defined types can be used as arguments and return
  values for the function defined by `tf.py_function`.
* **Tensor ops**: User-defined types can optionally be supported by most ops
  that accept Tensor inputs (e.g., `tf.matmul`, `tf.gather`, and
  `tf.reduce_sum`).
* **Distribution Strategy**: User-defined types can be used as per-replica
  values.

## Background

This RFC unites three internal TensorFlow interfaces that have been used to help
define core TensorFlow data types (composite tensors, type specs, and the
dispatch registry), and updates those interfaces to be simpler and more robust:

* **`CompositeTensor`** is a base class for types whose data is backed by one or
  more tensors.
* **`TypeSpec`** is a base class for storing type information and static
  metadata for a value.
* **The Dispatch Registry** allows TensorFlow ops (such as tf.add) to run
  different functions depending on their arguments' types.

In the design proposed by this RFC, the `CompositeTensor` base class is replaced
by a `tf.ExtensionType` protocol; and the dispatch registry is replaced by a
`tf.DispatchableType `protocol.  The internal implementation of type-based
dispatch is also refactored to increase robustness.  For further details about
the current design, and how the design proposed by this RFC differs from it, see
the appendix "Changes from Current (Internal-Only) Design".

## Design Proposal: Public APIs

TensorFlow extension types are defined using two protocols:

* **The `tf.ExtensionType` protocol** allows extension types to be used with
  "generic" TensorFlow APIs whose behavior does not depend on the value of each
  object.

* **The `tf.DispatchableType` protocol** allows extension types to override the
  default behavior for TensorFlow ops when they are called with an extension
  type value.

Classes that implement the `tf.ExtensionType` protocol are sometimes also called
"composite tensors."

Note: We are also considering using registries or base classes rather than
protocols; see the section on "Registry vs Protocol vs Base class" for a
discussion of the pros and cons.

### The tf.ExtensionType Protocol

Classes that implement the `tf.ExtensionType` protocol can be used with
"generic" APIs whose behavior does not depend on the value of each object (such
as `tf.function`, `SavedModel`, and `tf.while_loop`).  In order to implement
this protocol, a class's values must be immutable and decomposable into two
parts:

* A collection of Tensors, which encodes the value's dynamic data -- i.e., data
  that may change for different executions of a graph.

* An instance of a TypeSpec subclass, which encodes the value's static data --
  i.e., data that is the same for all executions of a graph.  (Each extension type
  implements its own TypeSpec subclass.)

As an example, consider `tf.RaggedTensor`, which adds ragged dimensions to a
`flat_values` `Tensor` by using `row_partition` tensors to divide it into
sublists.  Its **_dynamic data_** consists of the `flat_values` tensor and the
list of `row_partition` tensors (one for each ragged dimension).  Its **_static
data_**, which consists of the dtype and static shape for `flat_values`, the
number of ragged dimensions, and the dtype used to encode the `row_partition`
tensors, is stored using an instance of `tf.RaggedTensorSpec`.

As another example, consider a hypothetical `LabeledDigraph` class, which
encodes a directed graph with data on both nodes and edges.  Its **_dynamic
data_** could consist of: (1) a pair of a string-keyed dictionary of node
tensors with shape `[num_nodes, ...]`; (2) a string-keyed dictionary of edge
tensors with shape `[num_edges, ...]`; and (3) a pair of integer tensors
specifying the source and destination node index for each edge.  Its **_static
data_**, which would include information about the dtypes and static shapes of
all the node and edge label tensors, and would be stored in a
`LabeledDigraphSpec` class.

The work of decomposing values into parts and reconstructing values from those
parts is handled by the extension type's `TypeSpec` subclass.  Thus, the
`tf.ExtensionType` protocol just requires that we provide a `TypeSpec` for each
value:

```python
class ExtensionType(Protocol):
  """Protocol for defining TensorFlow extension types.

  TensorFlow extension types must be immutable, and their values must be
  decomposable into two parts:

  * A collection of Tensors, which encodes the value's dynamic data
    (i.e., data that may change for different executions of a graph).
  * An instance of `TypeSpec`, which encodes the value's static data
    (i.e., data that is the same for all executions of a graph).

  The `TypeSpec` is returned by `self.__tf_type_spec__()`; and the collection
   of tensors is returned by `self.__tf_type_spec__().to_components(self)`.
  """
  def __tf_type_spec__(self): TypeSpec
    """The `TypeSpec` describing the type for this value."""
    raise NotImplementedError
```

Note: `tf.ExtensionType` is a Python `Protocol`, so it does not need to be
added as an explicit base class.  See [PEP
544](https://www.python.org/dev/peps/pep-0544/) for details.

#### tf.TypeSpec

Each extension type defines its own subclass of `TypeSpec`, which has four jobs:

1. Storing static (non-tensor) data for values.
2. Serializing the TypeSpec itself.
3. Decomposing values into tensors and reconstructing values from tensors.
4. Checking for type compatibility.

The methods and properties that perform these four jobs are summarized here, and
described in the sections below:

```python
class TypeSpec(object):
  # Job 1: Store static data (constructor & properties defined in subclass)

  # Job 2: serialize the TypeSpec
  def serialize(self): ...
  def deserialize(cls, serialization): …

  # Job 3: Decompose and reconstruct values
  def to_components(self, value): ...
  def from_components(self, components): ...
  def component_specs(self): ...
  def value_type(self): …

  # Job 4: Equality and compatibility
  def __eq__(self, other): ...
  def __hash__(self): ...
  def is_compatible_with(self, spec_or_value): ...
  def most_specific_compatible_type(self, other): ...
```

##### TypeSpec Job 1: Storing Static Value Data

The first responsibility of a `TypeSpec` subclass is to store any static (non-tensor) data associated with a value.  A few examples will help demonstrate the type of data that is included in `TypeSpec`s:

* `tf.SparseTensorSpec` includes the dtype and static shape of a sparse tensor.
* `tf.RaggedTensorSpec` includes the dtype and static shape of a ragged tensor,
  along with the number of ragged dimensions and the dtype used to encode row
  partitions.
* For a hypothetical `LabeledTensor` extension type that pairs a `values` Tensor
  with a list of axis names, `LabeledTensorSpec` would include the axis names.
  It would also include the `dtype` and static shape of the `values` tensor.
* For a hypothetical `MaskedTensor` extension type that pairs a `values` Tensor
  with a boolean `mask`, `MaskedTensorSpec` would include the `shape` and
  `dtype` of the `values` tensor.  It does not need to include the `shape` of
  the mask tensor (since it should match the shape of the `values` tensor) or
  the `dtype` of the mask tensor (since it should always be `tf.bool`).

This static data is generally passed to the constructor, and stored as read-only
properties.  At a minimum, the static metadata contained in a `TypeSpec` must be
sufficient to determine the `dtypes` of any tensor components.  But as can be
seen in the examples above, it can be useful to include additional information
as well.

##### TypeSpec Job 2: Serializing the TypeSpec itself

The second responsibility of a `TypeSpec` subclass is to serialize `TypeSpec`
values into a nested structure containing a limited set of Python types (and
deserialize `TypeSpec` values from those nested structures).  This ensures that
`TypeSpec`s can be transmitted between processes and stored on disk (e.g., in
`SavedModels`).  In particular, `TypeSpec`s are serialized as part of
`SavedModel`s using `tensorflow.StructuredValue` protocol buffers.

```python
  @abstractmethod
  def serialize(self):
    """Returns a nested tuple containing the state of this TypeSpec.

    The serialization may contain the following value types: boolean, integer,
    string, float, None, TensorSpec, tf.TensorShape, tf.DType, np.ndarray,
    TypeSpec, and nested tuples, namedtuples, dicts, and OrderedDicts of any of the
    above.

    This method is used to provide default definitions for: equality testing
    (__eq__, __ne__), hashing (__hash__), pickling (__reduce__), string
    representation (__repr__), `most_specific_compatible_type`, 
    `is_compatible_with` and protobuf serialization (e.g. TensorInfo and 
    StructuredValue).

    Subclassing:
      Subclasses must override this method.  If this method does not return a
      tuple of values that can be used as arguments to the class's constructor,
      then `self.deserialize` must also be overridden.
    """

  @abstractclassmethod
  def deserialize(cls, serialization):
    """Reconstructs a TypeSpec from a value returned by serialize().

    Subclassing:
      If not overridden by a subclass, this method will return cls(*serialization).
    """
```

Typically, `serialize` will just return the constructor arguments that would be
used to reconstruct the `TypeSpec`.  For example, `tf.SparseTensorSpec(shape,
dtype).serialize()` returns the tuple `(shape, dtype)`; and
`tf.RaggedTensorSpec(shape, dtype, ragged_rank, row_splits_dtype).serialize()`
returns the tuple `(shape, dtype, ragged_rank, row_splits_dtype)`.

As a convenience, the serialization is also used to provide default
implementations for several other methods (described below).

##### TypeSpec Job 3: Decomposing and reconstructing values

The third responsibility of `TypeSpec` subclasses is decomposing values into
tensors and reconstructing values from tensors.  This is what allows "generic"
TensorFlow APIs to handle extension types.  `TypeSpec` defines two abstract
methods (`to_components` and `from_components`) for decomposing and
reconstructing values into **_components_**, which can be any nested structure
(as defined by `tf.nest`) whose leaf values are `tf.Tensors` or
`tf.ExtensionTypes`.  For example, `tf.SparseTensorSpec.to_components(st)`
returns a tuple of the three tensors `(st.indices, st.values, st.dense_shape)`
that encode the sparse data.

```python
  @abstractmethod
  def to_components(self, value):
    """Encodes `value` as a nested structure.

    Args:
      value: A value compatible with this TypeSpec.
       (Caller is responsible for ensuring compatibility.)

    Returns:
      A nested structure (as defined by tf.nest) which can be used to reconstruct
      value.  Leaf values must be tf.Tensors or types that implement
      __tf_type_spec__.  Must be compatible with self.component_specs.

    Subclassing:
      Subclasses must override this method.
      This method may not call any TensorFlow ops.
    """

  @abstractmethod
  def from_components(self, components):
    """Reconstructs a value from a nested structure.

    Args:
      components: A nested structure (as defined by tf.nest).  Leaf values must
        be `tf.Tensors` or `tf.ExtensionTypes`.
        Must be compatible with self.component_specs.
        (Caller is responsible for ensuring compatibility.)

    Returns:
      A value compatible with this TypeSpec.

    Subclassing:
      Subclasses must override this method.
      This method may not call any TensorFlow ops.
    """
```

Note: the restriction that `to_components` and `from_components` may not call
any TensorFlow ops comes from the fact that these methods are used in contexts
(such as control-flow) where adding new ops to the graph would be problematic.

`TypeSpec` subclasses also need to define the `value_type` and `component_specs`
properties, which provide information about the expected input and output types
for `to_components` and `from_components`.  For example,
`tf.SparseTensorSpec.value_type` returns `tf.SparseTensor`; and
`tf.SparseTensorSpec.component_specs` returns a tuple of three `tf.TensorSpecs`
describing each component of the sparse tensor (`indices`, `values`, and
`dense_shape`).

```python
  @abstractproperty
  def component_specs(self):
    """TypeSpecs for this type's components.

    Returns:
      A nested structure describing the component encodings that are returned by
      this TypeSpec's to_components method.  In particular, for a TypeSpec spec
      and a compatible value value, the following must not raise an exception:

      nest.map_structure(lambda t, c: assert t.is_compatible_with(c),
                         spec.component_specs, spec.to_components(value))

    Subclassing:
      Subclasses must override this property.
    """

  @abstractproperty
  def value_type(self):
    """The Python type for values that are compatible with this TypeSpec.

    Subclassing:
      Subclasses must override this property.
    """
```

##### TypeSpec Job 4: Equality and Compatibility.

The final responsibility of `TypeSpec` subclasses is checking equality and
compatibility between `TypeSpecs`.  Strict value-based equality is implemented
with `__eq__`:

```python
  def __eq__(self, other):
    """Returns True if `self` and `other` describe the same type.

    Subclassing:
      If not overridden by a subclass, the default behavior is to return true if
      self.serialize() is equal to other.serialize(), where TensorShapes are
      considered equal if their rank and dimensions all match exactly.
    """

  def __hash__(self):
    """Returns a hash value for `self`.

    Subclassing:
      If not overridden by a subclass, the default behavior is to hash a
      transformed copy of self.serialize(), where dictionaries are replaced
      by sorted (key, value) tuples.
    """
```

But there are some circumstances where we don't wish to impose strict equality
requirements for `TypeSpecs`.  For example, it should be possible to pass a
value with shape `[8, 3]` into a `tf.function` that expects a value with shape
`[None, 3]`, even though those shapes are not strictly equal.  To handle these
cases, `TypeSpec` defines the `is_compatible_with` method, which checks whether
two `TypeSpecs` (or a `TypeSpec` and a value) are compatible:

```python
 def is_compatible_with(self, spec_or_value):
    """Returns true if `spec_or_value` is compatible with this TypeSpec:

    * `spec.is_compatible_with(value)` is true if `value` belongs to the
      type described by `spec`.
    * `spec1.is_compatible_with(spec2)` is true if there are any values
      that belong to both `spec1` and `spec2`.

    `spec1.is_compatible_with(spec2)` must return False if `spec1.value_type !=
    spec2.value_type` or `spec1.component_specs != spec2.component_specs`.

    spec1.is_compatible_with(spec2) must equal spec2.is_compatible_with(spec1).

    Examples:

    >>> spec1 = TensorSpec([3], tf.float32)
    >>> spec1.is_compatible_with(TensorSpec([None], tf.float32))
    True
    >>> spec1.is_compatible_with(TensorSpec([4], tf.float32))  # shape mismatch
    False
    >>> spec1.is_compatible_with(TensorSpec([3], tf.int32))  # dtype mismatch
    False

    Args:
      spec_or_value: The TypeSpec or value to check.

    Returns:
      True if `self` is compatible with `spec_or_value`.

    Subclassing:
      If not overridden by subclasses, the default behavior is to convert
      spec_or_value to a TypeSpec (if it isn't already); and then to consider
      two TypeSpecs compatible if they have the same type, and the values
      returned by serialize are compatible (where tf.TensorShape, tf.TensorSpec,
      and tf.DType are checked for compatibility using their is_compatible_with
      method; and all other types are considered compatible if they are equal).
    """
```

Additionally, there are cases where we wish to combine two values that might be
incompatible, as long as there is some TypeSpec that is compatible with both.
For example, consider the expression `tf.cond(c, lambda: x, lambda: y)`, where
`x.__tf_type_spec__.shape=[8, 3]` and `y.__tf_type_spec__.shape=[8, 5]`.  Even
though these `TypeSpecs` are incompatible, we can return a value `r` whose
`TypeSpec` is compatible with both (`r.__tf_type_spec__.shape=[8, None]`).
These cases are handled by `TypeSpec.most_specific_compatible_type`:

```python
  def most_specific_compatible_type(self, other):
    """Returns the most specific `TypeSpec` compatible with `self` and `other`.

    Args:
      other: A TypeSpec.

    Returns:
      A `TypeSpec`; or `None` if no `TypeSpec` is compatible with both `self`
      and `other`.

    Subclassing:
      If not overridden by a subclass, the default behavior is to return None if
      self and other have different Python types, or if their type serializations 
      differ by anything other than TensorShapes.  Otherwise, the two type
      serializations are combined (using `most_specific_compatible_shape` to
      combine TensorShapes), and the result is used to construct and return a
      new TypeSpec.
    """
```

Notes:

* `spec1.is_compatible_with(spec2)` and
  `spec1.most_specific_compatible_type(spec2)` should generally return False
  if `type(spec1) != type(spec2)`.

* `TypeSpec.is_compatible_with` is used to check if two `TypeSpecs` are
  compatible.  E.g., `tf.function` can re-use a traced graph if the `TypeSpecs`
  of the arguments it is called with are compatible with the `TypeSpecs` that
  were used to trace the graph.

* `TypeSpec.most_specific_compatible_type` is used to merge two `TypeSpec`s or
  values.  E.g., in `tf.cond(pred, lambda: v1, lambda: v2)`, the `TypeSpec`
  used to reassemble the result is `spec1.most_specific_compatible_type(spec2)`
  (where `spec1=v1.__tf_type_spec__` and `spec2=v2.__tf_type_spec__`).


#### tf.nest support for tf.ExtensionType

The functions in `tf.nest` provide support for automatically unpacking and
repacking TensorFlow extension types (also known as composite tensors).  In
particular, most functions in the `tf.nest` package take an optional argument
`expand_composites`.  This argument indicates that composite tensors should be
treated as nested structures, and expanded into their component `Tensors`; and
similarly, that `TypeSpecs` should be treated as nested structures, and expanded
into their component `TensorSpecs`.  In particular:

**tf.nest.flatten:**

* `tf.nest.flatten(composite_tensor, expand_composites=True)` returns a flat
  list of the `tf.Tensor`s components from `composite_tensor`.
* `tf.nest.flatten(type_spec, expand_composites=True)` returns a flat list of
  `tf.TensorSpec`s describing the tensor components for `type_spec`.

**tf.nest.pack_sequence_as:**

* `tf.nest.pack_sequence_as(type_spec, tensor_list, expand_composites=True)`
  uses `type_spec.from_components` to reconstruct a composite tensor from its
  components.  Note that the new value's dynamic (tensor) data will come from
  `tensor_list`, but static (non-tensor) data will come from `type_spec`.

* `tf.nest.pack_sequence_as(composite_tensor, tensor_list, expand_composites=True)`
  uses `composite_tensor.__tf_type_spec__().from_components` to reconstruct the
  CompositeTensor from components.

Note: When using `tf.nest.pack_sequence_as` with composite tensors, the
`flat_sequence` argument must be a list of `Tensor`; it may not be a list of
TensorSpec`.

**tf.nest.assert_same_structure:**

* If `x` and `y` are both composite tensors or `TypeSpec`s, then
  `tf.nest.assert_same_structure(x, y, expand_composites=True)` raises an
  exception if there is no `TypeSpec` compatible with both `x` and `y` (as
  determined by calling `TypeSpec.most_specific_compatible_type`).

**tf.nest.map_structure:**

* `tf.nest.map_structure(func, composite_tensor, expand_composites=True)`
  transforms `composite_tensor` by flattening it into its component tensors,
  applying `func` to transform each component tensor, and then repacking those
  transformed tensors into a composite tensor with the original type.

The following example uses `nest.flatten` with `expand_composites=True` to
convert a nested structure containing composite tensors to a list of
`tf.Tensors`; applies a function `f` to transform each tensor; and then uses
`nest.pack_sequence_as` with `expand_composites=True` to reassemble the results
back into the original structure.

```python
>>> rt = RaggedTensor(values=v1, row_splits=r)
>>> st = SparseTensor(indices=i, values=v2, dense_shape=d)
>>> structure = {'a': rt, 'b': st}
>>> flat = nest.flatten(structure, expand_composites=True)
[v1, r, i, v2, d]
>>> mapped = [f(t) for t in flat]
>>> nest.pack_sequence_as(structure, mapped)
{'a': RaggedTensor(f(v1), f(r)), 'b': SparseTensor(f(i), f(v2), f(d))}
```

#### TypeSpec Registry

In order to be used with `SavedModel`s, extension types must register their
`TypeSpec`s using `tf.register_type_spec`.

```python
def register_type_spec(type_spec_subclass, name=None):
  """Registers a globally unique name for a `TypeSpec`.

  Args:
    type_spec_subclass: A concrete subclass of `TypeSpec`.
    name: The name of the type spec.  Must be globally unique.
      Defaults to `type_spec_subclass.__name__`.

  Raises:
    ValueError: If a different `TypeSpec` has already been registered with the
      same name; or if `type_spec_subclass` has already been registered with a
      different name.
  """
```

#### tf.StackableTypeSpec

`tf.StackableTypeSpec` is an abstract subclass of `tf.TypeSpec` that is used to
define extension types that support stacking and unstacking values.  But unlike
the `tf.stack` and `tf.unstack` operations, the number of values to be
(un)stacked does not need to be statically known.  Extension types that extend
`StackableTypeSpecs` can be used with TensorFlow APIs that require stacking and
unstacking an arbitrary number of values, such as `tf.data.Dataset.batch`,
`tf.data.Datset.unbatch`, and `tf.map_fn`.  For example, datasets containing
`RaggedTensor` can be batched or unbatched because `RaggedTensorSpec` is a
`StackableTypeSpec`:

```python
>>> rt = tf.ragged.constant([[1, 2], [], [3], [4, 5, 6], [7], [8, 9]])
>>> ds = tf.data.Dataset.from_tensor_slices(rt)
>>> for x in ds.batch(3):
...   print(x)
<tf.RaggedTensor [[1, 2], [], [3]]>
<tf.RaggedTensor [[4, 5, 6], [7], [8, 9]]>
```

The `tf.StackableTypeSpec` class has two jobs (in addition to the four jobs
defined by the `TypeSpec` base class):

* "Boxing" values into a `tf.Tensor` that can be stacked/unstacked (and
  "unboxing" them).
* Building the `TypeSpec` describing a stacked/unstacked value.

Stacking, unstacking, or concatenating boxed tensors must be equivalent to
stacking, unstacking, or concatenating the corresponding unboxed values.  I.e.,
if `values=[v<sub>1</sub>, v<sub>2</sub>, …, v<sub>N</sub>]` is a list of values
that have the same `type_spec`, then boxing those values, stacking the boxed
tensors, and unboxing the result is equivalent to stacking the values:

```python
boxed_tensors = [type_spec.to_boxed_tensor(v) for v in values]
stacked_tensor = tf.stack(boxed_tensors, axis=0)
unboxed_stacked_value = type_spec.stacked(N).from_boxed_tensor(stacked_tensor)
assert unboxed_stacked_value == tf.stack(values, axis=0)
```

Going in the other direction, if `v` is a single value whose `TypeSpec` is
`type_spec` and whose `rank>0`, then boxing that value, unstacking the boxed
tensor, and unboxing the result is equivalent to unstacking the value:

```python
boxed_tensor = type_spec.to_boxed_tensor(v, minimum_rank=1)
unstacked_tensors = tf.unstack(boxed_tensor, axis=0, num=N)
unboxed_unstacked_values = [type_spec.unstacked().from_boxed_tensor(t)
                            for t in unstacked_tensors]
assert unboxed_unstacked_values == tf.unstack(boxed_tensor, axis0, num=N)
```

In some cases, it can be convenient to use a collection of "parallel" boxed
tensors to encode a value.  To support that use case, the boxing method may
return a list of tensors, which must be stacked or unstacked in parallel.  I.e.,
stacking, unstacking, or concatenating values must be equivalent to stacking,
unstacking, or concatenating each of the corresponding tensors from the boxed
encoding.

`StackableTypeSpec` defines the methods `to_boxed_tensor` and
`from_boxed_tensor` for boxing and unboxing values:

```python
class StackableTypeSpec(TypeSpec):

  @abstractmethod
  def to_boxed_tensor(self, value, minimum_rank=0):
    """Encodes `value` as a stackable Tensor.

    Args:
      value: A value compatible with this TypeSpec.
        (Caller is responsible for ensuring compatibility.)
      minimum_rank: The minimum rank for the returned tensor(s).  This can
        be used to ensure that the boxed tensor(s) can be unstacked this number
        of times.

    Return:
      A `Tensor` (or list of `Tensors`) that encodes `value`.  Stacking, 
      unstacking, or concatenating boxed tensors must be equivalent to stacking, 
      unstacking, or concatenating the corresponding unboxed values.

      The returned tensor must have rank greater than or equal to `minimum_rank`.

      If `to_boxed_tensor` returns a list of `Tensors`, then they should be
      treated as parallel tensors, and corresponding values should be combined.
      I.e., stacking, unstacking, or concatenating values must be equivalent to
      stacking, unstacking, or concatenating each of the corresponding tensors
      from the boxed encoding.  If a list of `Tensors` is returned, they must all
      have the same shape up to axis `minimum_rank`.
    """

  @abstractmethod
  def from_boxed_tensor(self, boxed_tensor):
  """Decodes `value` from a stackable Tensor.

  Args:
    boxed_tensor: A `Tensor` (or list of `Tensors`) that was returned by 
      `to_boxed_tensor`; or a `Tensor` (or list of `Tensors`) that was formed
      by stacking, unstacking, and concatenating the values returned by
      `to_boxed_tensor`.

    Returns:
      A value compatible with this TypeSpec.
  """

  @abstractmethod
  def boxed_tensor_spec(self, minimum_rank=0):
  """Returns a TensorSpec (or list of TensorSpecs) for the boxed tensor encoding.

  Args:
    minimum_rank: The minimum rank for the returned TensorSpecs.

  Returns:
    A `TensorSpec` (or list of `TensorSpecs`) that is compatible with 
    `self.to_boxed_tensor(v, minimum_rank)` for any value `v` that is
    compatible with this `TypeSpec`.
  """

  @abstractmethod
  def stacked(self, num):
    """Returns a TypeSpec representing stacked objects with this TypeSpec.

    Args:
      num: An `int` indicating the number of objects that are stacked together,
        or `None` if the number of objects is not known.
    """

  @abstractmethod
  def unstacked(self):
    """Returns a TypeSpec representing a single unstacked element in this TypeSpec.
    """
```


Note: The `to_boxed_tensor` and `from_boxed_tensor` methods are typically
implemented by defining new c++ Kernels that encodes values using tensors with
`dtype=tf.variant`.  The gradient for `to_boxed_tensor` typically calls
`from_boxed_tensor`, and vice versa.

Note: one key difference between the "boxed encoding" and the "component
encoding" is that `to_boxed_tensor` and `from_boxed_tensor` may (and often do)
add operations to the graph, while `to_components` and `from_components` may
not.


##### Motivation for tf.StackableTypeSpec

As mentioned above, the `StackableTypeSpec` class allows extension types to be
handled by TensorFlow APIs that require stacking and unstacking an arbitrary
number of values, such as `tf.data.Dataset.batch`, `tf.data.Datset.unbatch`, and
`tf.map_fn`.  However, it's not immediately obvious why we can't use "simpler"
solutions instead.  This section explains why those simpler solutions won't
work.

**Why can't we just use tf.stack and tf.unstack?**

> `tf.stack` and `tf.unstack` require that the number of values being stacked
> (or unstacked) be statically known.  However, the APIs listed above are
> often used in contexts where the number of values to stack or unstack is not
> known ahead of time.

**Why can we just use control flow with indexing and concatenation?**

> It would be possible to implement the APIs listed above using a `while_loop`
> that uses indexing (`value[i]`) to unstack values (one at a time), and
> `tf.concat` to concatenate them back together (one at a time).  However,
> indexing individual elements is inefficient for some types (such as
> `tf.SparseTensor`); and concatenating values back together with `N-1` calls
> to `tf.concat` is inefficient for most types.  We decided that the poor
> performance that these operations would have if implemented with indexing
> and concatenation is unacceptable.


### The tf.DispatchableType Protocol

Extension types that are "tensor-like" (i.e., which specialize or extend
`tf.Tensor`) can use the `tf.DispatchableType` protocol to specialize the
behavior of TensorFlow ops when they are called with extension type values:


```python
class DispatchableType(Protocol):
  """Protocol for defining TensorFlow extension types that support dispatch.

  When a `DispatchableType` is passed to a TensorFlow op argument that supports
  dispatch, the `DispatchableType`'s `__tf_dispatch__` method will be used to
  execute the op (unless `__tf_dispatch__` returns `NotImplemented`).

  If the `__tf_dispatch_types__` class variable is set, then `__tf_dispatch__`
  will only be called if all arguments that expect Tensor values have types
  in the specified list.  (In most cases, this avoids the need to check argument
  types and return `NotImplemented` when unsupported types are found.)
  """

  @classmethod
  def __tf_dispatch__(cls, op, args, kwargs):
    """Returns a value for `op(*args, **kwargs)`, or `NotImplemented`.

    Args:
      op: A TensorFlow function that supports operation dispatch.
      args: The positional arguments from the original call.
      kwargs: The keyword arguments from the original call.

    Returns:
      The result of applying `op` to the specified arguments, or `NotImplemented`
      if this dispatch handler does not support the specified arguments.
    """

  __tf_dispatch_types__: ClassVar[Optional[Tuple[type, ...]]] = None

```


Note: `tf.DispatchableType` is a Python `Protocol`, so it does not need to be
added as an explicit base class.  See [PEP
544](https://www.python.org/dev/peps/pep-0544/) for details.

A tensorflow operation that **_supports dispatch_** will check whether its
arguments implement the `DispatchableType` protocol; and if so, then it will use
`__tf_dispatch__` to execute the op (unless `__tf_dispatch__` returns
`NotImplemented`).

Dispatch will be supported by most public TensorFlow operations that have
`tf.Tensor` or `Sequence[tf.Tensor]` arguments.  But only arguments that expect
`tf.Tensor` or `Sequence[tf.Tensor]` are checked for dispatch.  In particular,
note that:

* Arguments that expect non-`Tensor` values are not checked for dispatch.  For
  example, the `keepdims` argument to `tf.math.reduce_sum` expects a python
  boolean (not a `Tensor`), so it is not checked.

* Arguments that expect functions or predicates are not checked for dispatch.
  For example, the return values of the `true_fn` and `false_fn` arguments to
  `tf.cond` are not checked for dispatch.  (But they are handled generically if
  the arguments implement the `tf.ExtensionType` protocol.)

* Arguments that expect arbitrary nested structures (as defined by `tf.nest`)
  that may include tensors are generally not checked for dispatch.  For example,
  the `loop_vars` argument to `tf.while_loop` is not checked.


#### Which operations to override

Dispatchable types may choose which operations to override, and only need to
override the operations that make sense for that type.  For example:

* `tf.StructuredTensor` (which can conceptually be thought of as a tensor of
  dictionary-like "structures") supports array manipulation operations (such as
  `tf.concat`, `tf.tile`, `tf.slice`, and `tf.gather`); but not mathematical
  operations (such as `tf.add` or `tf.reduce_sum`).

* `tf.RaggedTensor` does not support the operations `tf.shape` and `tf.reshape`,
  since ragged shapes can not be encoded as a vector of dimension sizes. 

TensorFlow defines a large number of operations, which makes it difficult to
define overrides for all of them.  In order to simplify the task of overriding
TensorFlow operations, we will provide a collection of functions that give
information about the semantic properties of an operation.  For example:

* `tf.dispatch.is_unary_elementwise_op(op)`: Returns true if `op` applies an
  independent transformation to each element of its first argument. Examples
  include: `tf.math.abs`, `tf.math.log`, `tf.strings.length`.

* `tf.dispatch.is_binary_elementwise_op(op)`: Returns true if `op` applies an
  independent transformation to the corresponding elements of its first two
  arguments.  Examples include: `tf.math.add`, `tf.math.equal`.  Note that these
  operations generally support broadcasting between their first two arguments.

* `tf.disptach.is_reduction_op(op)`: Returns true if `op` combines the values of
  its first argument along an axis (or set of axes) specified by the `axis`
  argument.  Examples include: `tf.math.reduce_sum`, `tf.strings.reduce_join`.

#### Argument canonicalization

To simplify the work that needs to be done by dispatch handlers, the `args` and
`kwargs` arguments are canonicalized by moving any
[`POSITIONAL_OR_KEYWORD`](https://docs.python.org/3/library/inspect.html#inspect.Parameter)
arguments to `args`, even if the original caller used a keyword
argument to pass them.  E.g., this ensures that the first argument to a unary
elementwise op will always be `args[0]` (and will not be in `kwargs`).

#### Precedence

If multiple arguments to a TensorFlow op implement the `tf.DispatchableType`
protocol, then we need to decide which one to call first.  We will use the
following rules (which are [consistent with Numpy’s `__array_function__`
protocol](https://numpy.org/neps/nep-0018-array-function-protocol.html#trying-
array-function-methods-until-the-right-one-works)):

* Subclasses are used before superclasses, regardless of position.  I.e., if
  two arguments `x` and `y` both implement `DispatchableType` (with different
  methods), and `issubclass(x, y)`, then `type(x).__tf_dispatch__` method should
  be called instead of `type(y).__tf_dispatch__`, even if `y` occurs first in
  the argument list.

* Otherwise, values are used left-to-right.  I.e., earlier arguments are used
  before later arguments; and for sequence-valued arguments, values are used in
  the order they appear in the sequence.

* If all `__tf_dispatch__` methods return `NotImplemented`, then the original op
  is called (which will typically lead to a `TypeError` unless the extension
  type is convertible to a tensor).

### Example: SimpleMaskedTensor

To make the ExtensionType and DispatchableType protocols more concrete, we will
illustrate how they could be used to create a class that pairs a Tensor with a
corresponding boolean mask, indicating which values are valid.  We begin by
defining the `SimpleMaskedTensor` class itself.  Note that we make `value` and
`mask` read-only properties, to ensure that `SimpleMaskedTensor` is immutable:

```python
class SimpleMaskedTensor(object):
  """Class that pairs a `value` tensor with a corresponding boolean `mask`."""

  def __init__(self, value, mask):
    value.shape.assert_is_compatible_with(mask.shape)
    self._value = value
    self._mask = mask

  value = property(lambda self: self._value)
  mask = property(lambda self: self._mask)

  # The shape & dtype of the masked tensor are the shape & dtype of its values.
  shape = property(lambda self: self._value.shape)
  dtype = property(lambda self: self._value.dtype)

  # Implement the tf.ExtensionType protocol.
  def __tf_type_spec__(self):
    return SimpleMaskedTensorSpec(self.shape, self.dtype)
```

Next, we define `SimpleMaskedTensorSpec`.  The following table summarizes how `SimpleMaskedTensorSpec` handles each of its four jobs:

Job                         | SimpleMaskedTensorSpec
--------------------------- | -----------------------------------------------
Storing non-tensor metadata | Stores the shape and value dtype for the masked tensor.
Serializing the TypeSpec    | Serializes the shape and dtype as a tuple.
Decomposing values          | Decomposes the masked tensor into a (value, mask) tuple.
Checking compatibility      | Considers two MaskedTensors compatible if their dtypes and shapes are compatible.

The complete code for `SimpleMaskedTensorSpec` is shown below:

```python
class SimpleMaskedTensorSpec(tf.TypeSpec):
  """Type specification for a `SimpleMaskedTensor`."""

  def __init__(self, shape: tf.TensorShape, dtype: tf.dType):
    """Creates a new `SimpleMaskedTensorSpec`.

    Args:
      shape: The shape of the `SimpleMaskedTensor`.
      dtype: The dtype of the `SimpleMaskedTensor`'s values.
    """
    self._shape = shape
    self._dtype = dtype

  shape = property(lambda self: self._shape)
  dtype = property(lambda self: self._dtype)
  value_type = property(lambda: SimpleMaskedTensor)

  def to_components(self, masked_tensor):
    return (masked_tensor.value, masked_tensor.mask)

  def from_components(self, components):
    return SimpleMaskedTensor(*components)

  def component_specs(self):
    return (tf.TensorSpec(self._shape, self._dtype),
            tf.TensorSpec(self._shape, tf.bool))

  def serialize(self):
    return (self._shape, self._dtype)
```

Note: `SimpleMaskedTensorSpec` uses the default implementations for several
`TypeSpec` methods, such as `is_compatible`, which are defined based on
`serialize` and `deserialize`.

At this point, `SimpleMaskedTensor` can be used with "generic" TensorFlow APIs,
such as `tf.function`, `SavedModel`, and `tf.while_loop`.  But since
`SimpleMaskedTensor` is tensor-like, it makes sense for it to implement the
`tf.DispatchableType` protocol as well.  We can do so by adding a
`__tf_dispatch__` method.  For simplicity, we will only show support for unary
and binary elementwise operations and a handful of other operations in this
example.


```python
class SimpleMaskedTensor(object):
  # [...definition continued from above...]

  @classmethod
  def __tf_dispatch__(cls, op, args, kwargs):
    if tf.dispatch.is_unary_elementwise_op(op):
      return self._unary_elementwise_dispatch(op, args, kwargs)
    elif tf.dispatch.is_binary_elementwise_op(op):
      return self._binary_elementwise_dispatch_op, args, kwargs)
    else:
      dispatch_handler = SimpleMaskedTensor.__dispatchers.get(op, None)
      if dispatch_hander is not None:
        return dispatch_handler(args, kwargs)
      else:
        return NotImplemented

  # We support ops that take tf.Tensor and SimpleMaskedTensor arguments.  We
  # don't support any other dispatchable argument types (such as tf.RaggedTensor).
  __tf_dispatch_types__ = (tf.Tensor, SimpleMaskedTensor)

  __dispatchers = {}  # dict mapping operation to handler function.

  @classmethod
  def _unary_elementwise_dispatch(op, args, kwargs):
    args = list(args)  # Make a copy so we can make modifications.
    first_arg = args.pop(0)
    if not isinstance(first_arg, SimpleMaskedTensor):
      return NotImplemented
    transformed_values = op(first_arg.values, *args, **kwargs)
    return SimpleMaskedTensor(transformed_values, first_arg.mask)

  @classmethod
  def _binary_elementwise_dispatch(op, args, kwargs):
    args = list(args)  # Make a copy so we can make modifications.

    # Extract values & masks from the first two args.  Allow Tensors to be
    # combined with SimpleMaskedTensors.
    values = []
    masks = []
    for arg in args[:2]:
      if isinstance(arg, tf.Tensor):
        values.append(arg)
      elif isinstance(arg, SimpleMaskedTensor):
        values.append(arg.values)
        masks.append(arg.mask)
      else:
        return NotImplemented

    transformed_values = op(*values, *args[2:], **kwargs)
    if len(masks) == 1:
      combined_mask = masks[0]
    else:
      combined_mask = tf.math.logical_and(*masks)
    return SimpleMaskedTensor(transformed_values, combined_mask)

def masked_tensor_shape(input, out_type=tf.int32, name=None):
  return tf.shape(input.values)

def masked_tensor_tile(input, multiples, name=None):
  with tf.name_scope(name):
    return SimpleMaskedTensor(tf.tile(input.values, multiples),
                              tf.tile(input.mask, multiples))

def masked_tensor_reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
  with tf.name_scope(name):
    return SimpleMaskedTensor(
        tf.reduce_sum(input_tensor.values, axis, keepdims),
        tf.reduce_all(input_tensor.mask, axis, keepdims))

SimpleMaskedTensor.__dispatchers.extend({
  tf.shape: masked_tensor_shape,
  tf.tile: masked_tensor_tile,
  tf.reduce_sum: masked_tensor_reduce_sum,
})
```

## Future Work

This section describes extensions capabilities that we are not including in the
initial release for TF Extension Types, but that we plan to add in the future.


### Automatic type casting

Under the current design, extension type values can only be combined if they
have identical `value_type`s and `component_spec`s.  This can prevent seamless
interoperation between types.  For example, the following expression is not
supported under the current design:

```python
tf.cond(pred, lambda: dense_tensor, lambda: ragged_tensor)  # not supported
```

One solution to this problem would be to add support for automatic type-casting
of `TypeSpec` values.  In particular, we could extend `TypeSpec` with the
following methods:

```python
  def cast(self, value):
    """Returns a value that is equivalent to `value` and compatible with `self`."""

  def castable_type(self, spec):
    """Returns a TypeSpec that values of `self` and spec can be cast to."""
```

For example, `RaggedTensorSpec(...).cast(dense_tensor)` would return `RaggedTensor.from_dense(dense_tensor)`.

In addition to improving seamless interoperation between types in APIs that
combine values, the automatic type casting mechanism might also be useful for
supporting backwards compatibility.  In particular, this would make it possible
for a `TypeSpec` to change the component encoding for a value, as long as the
`TypeSpec` itself records a version number, and implements a `cast` method that
can convert the old encoding to the new encoding (or vice versa).

### ExtensionTypes and TypeSpecs in c++

Under the current design, extension types exist only in Python.  As a result,
the TensorFlow c/c++ APIs and APIs such as TensorFlow serving do not support
extension types.  In order to extend extension type support to c++, we are
considering defining corresponding `ExtensionType` and `TypeSpec<T>` abstract
base classes in c++ (where the template parameter `T` is a subclass of
`ExtensionType`).


## Open Questions


### Which Class should Decompose & Reconstruct Values?

In the current design, the only job of an `ExtensionType` is to return a
`TypeSpec`; but, as described above, the `TypeSpec` class has four different
jobs it must perform:

1.  Storing static (non-tensor) data for values.
2.  Serializing the TypeSpec itself.
3.  Decomposing values into tensors and reconstructing values from tensors.
4.  Checking for type compatibility.

In an alternative design, we could move job (3) from `TypeSpec` to
`ExtensionType`.  I.e., the `ExtensionType` would be responsible for decomposing
and reconstructing values.  In particular, this would require removing the
`to_components`, `from_components`, and `component_specs` methods from
`TypeSpec`, and adding them to `ExtensionType` as:


```python
class ExtensionType(Protocol):
  def __tf_to_components__(self):
    """Encodes `self` as a nested structure."""

  @classmethod
  def __tf_from_components__(cls, components, type_spec):
    """Reconstructs a value from a nested structure and a TypeSpec."""

  @classmethod
  def __tf_component_specs__(cls, type_spec):
    """Returns the TypeSpecs for the components a value with a given TypeSpec."""
```

(With this heavier-weight definition, we might also want to change
`ExtensionType` from a `Protocol` to an abstract base class.)

Advantages of moving the responsibility for decomposing & reconstructing values
from `TypeSpec` to `ExtensionType` include:

* Reduces the number of jobs that `TypeSpec` is expected to perform.
  `TypeSpec` becomes closer to a pure-data class (though it still needs to
  understand type compatibility.)

* Eliminates a circular dependency between `ExtensonType` and `TypeSpec`
  (assuming that a registry is added to track the `ExtensionType` for each
  `TypeSpec` -- the link from `TypeSpec` to `ExtensionType` is required to
  implement `tf.nest.pack_structure_as(type_spec, list_of_tensors)`).

* Makes it easier to define extension types with "hidden implementations" --
  i.e., where the component tensors are not meant to be publicly accessible.

But advantages of the currently proposed design include:

* All the code needed to turn a type into an extension type is kept together in
  one place (rather than being split across two types: the extension type itself
  and the TypeSpec).  This is important because the logic behind this code can
  be fairly tightly tied.  For example, the definitions for `is_compatible_with`
  and `most_specific_compatible_type` must be kept consistent with the component
  decomposition.

* Makes it possible to non-invasively turn types into extension types (though
  see below for a discussion about whether this is a good idea or not).

### Should it be possible to define extension types non-invasively?

Allowing types to be non-invasively declared as TensorFlow extensions has both
pros and cons:

* **Pro:** TensorFlow APIs can be used with a type, even if that type was not
  designed to be used with TensorFlow.  (However, that type must satisfy the
  requirements of extension types: namely it must be immutable; it must store
  all dynamic data in tensors; and all static data must be serializable.)

* **Con:** Multiple users might declare different (incompatible) TypeSpecs for
  the same type, especially for common types.  In most cases, this will result
  in an error (which may not be very actionable if the incompatible TypeSpecs
  are registered by libraries that a user is using).  But in some cases, this
  could result in silently incorrect behavior -- e.g., if one TypeSpec is
  registered for a type when a model is saved to a SavedModel, but a different
  TypeSpec is registered for that type when a model is loaded via tf.hub.

### Registry vs Protocol vs Base Class

The current RFC proposes that the API for defining TensorFlow extension types be
implemented using protocols.  Two alternative APIs that we considered are:

* **Base classes**: Define an abstract class that must be included as a base
  class for TensorFlow extension types.  We could have a single base class with
  an abstract `type_spec` property and an abstract `dispatch` method; or we
  could have separate "mix-in" base classes for each one.  These base classes
  might or might not have metaclasses.

* **Registration**: Provide functions which can be used to register the
  `TypeSpec` and the dispatch handler for an extension type.

The following table summarizes some of the pros and cons of each approach:

                                              | Protocol   | Subclass | Registry
--------------------------------------------- | ---------- | -------- | -------
Works well with type annotations                    | Y    |    Y     |    N
Extension type doesn't need to depend on TensorFlow | Y    |    N     |    N
Extension types can be added non-invasively ①      | N    |    N     |    Y
Provides a centralized list of all extension types  | N    |    Y ②   |    Y

① As noted above, allowing extension types to be added non-invasively has both
pros and cons.

② Requires defining a metaclass.


### TypeSpec.name

`TensorSpec` is currently the only `TypeSpec` that defines a `name` attribute.
To my knowledge, the main use for this attribute is to override the default
argument names for concrete functions when they are called with the flat-
arguments calling convention.  The question has been raised as to whether all
`TypeSpec`s should have a `name` attribute.  My initial feeling is that they
should not, because the core concept for `TypeSpec` is a type specification, and
type specifications are generally not named.  However, feedback on this question
is welcome, especially when grounded in concrete use cases.  (Does
`TensorSpec.name` get used for other things than concrete signature argument
naming?)


### Should user-defined types be supported as inputs/outputs when computing gradients?

User-defined types are not currently supported as the inputs or outputs for
gradients.  I.e., gradients need to be computed with respect to individual
component tensors.  We could potentially add support for using extension types
with gradients directly, though further work might be needed to define the exact
semantics.


## Appendix: Changes from Current (Internal-Only) Design

### Changes to `CompositeTensor`


* The CompositeTensor base class is replaced with an `ExtensionType` protocol.

* `CompositeTensor._type_spec` is renamed to `ExtensionType.__tf_type_spec__`,
  and is changed from an abstractproperty to an abstractmethod.

* The `CompositeTensor._consumers` method is dropped -- any clients that need
  the consumers of components can use `tf.nest` to flatten it to a list of
  tensors, and check the consumers of those tensors.

* The `CompositeTensor._shape_invariant_to_type_spec` method is dropped.  This
  was used for backwards compatibility.

### Changes to `TypeSpec`

* Several private methods are made public (e.g. `_to_components`).

* `most_specific_compatible_type(t1, t2) now returns None (rather than raising
  an exception) if there is no type compatible with both `t1` and `t2`.

* The `BatchableTypeSpec` subclass is renamed to `StackableTypeSpec`, and method
  names are renamed accordingly:
    *   `_to_tensor_list` and `_to_batched_tensor_list` → `to_boxed_tensor`
        * A new `minimum_rank` parameter is used to indicate the desired rank
          for the boxed tensor.
        * `to_boxed_tensor` may optionally return a single tensor (instead of a
          list of tensors).  We expect this to be the common case.
    *   `_from_tensor_list` → (removed)
    *   `_from_compatible_tensor_list` → `from_boxed_tensor`
    *   `_flat_tensor_specs` → `boxed_tensor_spec`
    *   `\_to\_batched\_tensor\_list

* Added a registry for `TypeSpec`s.  In the current (internal) design, all
  extension types are listed explicitly in
  `tensorflow/core/protobuf/struct.proto`.  That proto will be extended to allow
  TypSpecs to be encoded using their registered name.

### Changes to `OpDispatcher`

* The `OpDispatcher` class is replaced with the `DispatchableType` protocol.
* Added functions (such as `tf.dispatch.is_unary_elementwise_op(op)`) that can
  be used to check semantic properties of operations.
* The dispatch-handling implementation will be changed from a reactive
  exception-based mechanism to a proactive protocol-checking mechanism.


