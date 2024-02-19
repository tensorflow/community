# RFC: Tensor Indexed Updates

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [433](https://github.com/tensorflow/community/pull/433) |
| **Author(s)** | C. Antonio Sanchez (Google, cantonios@) |
| **Sponsor**   | Alan Liu (Google)            |
| **Updated**   | 2022-10-20                                           |

## Objective

Introduce a way to set tensor values using a JAX or NumPy-like syntax:
```python
# Immutable Tensors can be copied with updates following a JAX-like syntax:
y = y.at[:, :, 2].set(x)

# tf.Variable can either use JAX-like syntax (copy) or NumPy syntax (in-place):
v = v.at[:, :, 2].set(x)  # Updated copy.
v[:, :, 2] = x            # In-place.
```
The principal goal is to unify many of the existing tensor update methods into a
more friendly and familiar syntax.

Non-goal: introduce mutable tensors.  For performance and compatibility reasons,
tensors will remain immutable, only allowing modified _copies_ to be returned.

## Motivation

Updating a slice of an existing tensor ideally would be as simple as
```python
tensor[:, :, 2] = values
```
Many TensorFlow users expect this to work out of the box - it
does in [NumPy](https://numpy.org/doc/stable/user/basics.indexing.html#assigning-values-to-indexed-arrays)
and [PyTorch](https://pytorch.org/cppdocs/notes/tensor_indexing.html#setter), so
why not TensorFlow?  The inability to easily update tensor values by index has
been a major pain point (e.g. tensorflow/tensorflow#33131,
tensorflow/tensorflow#51582, tensorflow/tensorflow#55812,
tensorflow/tensorflow#56381), resulting in calls to abandon TensorFlow entirely
in favor of PyTorch.

Why isn't it so simple to add such indexed updates?  TensorFlow tensors are
immutable.  This early design choice simplifies the mental model, and allows
many important performance optimizations in graph building, multithreading, and
cross-device support.  So, we need another mechanism.  It turns out we can
already accomplish most of these update operations by building upon existing
functionality within TensorFlow.  We simply need to expose them via a
friendlier API.

## Existing Update Mechanisms

The majority of the functionality required to support index updates actually
already exists in TensorFlow, it is just cumbersome to use and hidden across
multiple functions.

To extract items from a tensor:
- `tf.gather`: gathers slices along a single axis according to a tensor of indices.
- `tf.gather_nd`: gathers slices according to a tensor of indices.
- `tf.slice`: extracts a contiguous slice form a tensor.
- `tf.strided_slice`: extracts a strided slice of a tensor.
- `tf.Tensor.__getitem__`: numpy-like indexing that supports strided slices and
   masks (internally uses the above).

To replace or modify items in a tensor according to a set of indices:
- `tf.where`: multiplexes between two tensors based on a mask.
- `tf.tensor_scatter_nd_update`: copies a tensor, replacing values at a set of
   indices (similar functions exist for `add`, `multiply`, etc.)
- `tf.raw_ops.TensorStridedSliceUpdate`: internal op for replacing a strided
  slice with values.

The previous functions can be used in combinations to arbitrarily update a
tensor.  Consider the following illustrative examples.

**Setting a tensor slice to zero**:
```python
#========================================================================
# Numpy:
#========================================================================
x[:,:,2] = 0

#========================================================================
# TensorFlow:
#========================================================================
# Construct a set of indices to access tensor[:,:,2].
x_shape = tf.shape(x)
dim0 = tf.range(x_shape[0])
dim1 = tf.range(x_shape[1])
dim0, dim1 = tf.meshgrid(dim0, dim1, indexing='ij')
dim0, dim1 = tf.reshape(dim0, [-1]), tf.reshape(dim1, [-1])
dim_shape = tf.shape(dim0)
dim2 = tf.broadcast_to(tf.constant(2, dtype=dim0.dtype), dim_shape)
indices = tf.transpose(tf.stack([dim0, dim1, dim2]))

# Create a set of values with the same number of elements in indices, since
# tensor_scatter_* currently doesn't support broadcasting.
updates = tf.broadcast_to(tf.constant(0, dtype=tensor.dtype),dim_shape)

# Scatter updates into x.
x = tf.tensor_scatter_nd_update(x, indices, updates)
```

**Setting negative values to zero:**
```python
#========================================================================
# Numpy:
#========================================================================
x[x < 0] = 0

#========================================================================
# TensorFlow via tf.tensor_scatter_nd_update:
#========================================================================
# Extract index tensor from boolean mask.
indices = tf.where(x < 0)
# Build values tensor.
dim_size = tf.shape(indices)[0:1]
values = tf.broadcast_to(tf.constant(0, dtype=x.dtype), dim_size)
# Scatter updates into x.
x = tf.tensor_scatter_nd_update(x, indices, values)

#========================================================================
# TensorFlow via multiplexing tf.where:
#========================================================================
# NOTE: this is only equivalent to NumPy for simple cases where the "updates"
# is a scalar.  Otherwise, the RHS to `x[mask] = RHS` needs to match the shape
# of indices returned via `tf.where(mask)`, which is not the same shape as the
# mask itself.
x = tf.where(x < 0, 0, x)
```

As you can see, although it still is technically possible to perform many of the
same operations with using TensorFlow's existing functionality, it's quite
cumbersome.

## Proposed Syntax

Rather than invent a new API syntax, we will rely on the successful rollout of
indexed updates in JAX - which also has the immutable tensor restriction - via
[`.at[]`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).


```python
#========================================================================
# Proposed TensorFlow Syntax:
#========================================================================
# Setting a tensor slice to zero:
x = x.at[:,:,2].set(0)

# Setting negative values to zero:
x = x.at[x < 0].set(0)
```

For `tf.Variable`, we have already had _some_ support for in-place slice updates
via the `.assign()` method:
```python
# Existing tf.Variable in-place slice assignment:
v[:, :, 2].assign(0)
```
For consistency with NumPy, we propose to introduce direct assignment as well:
```python
# Proposed tf.Variable in-place slice assignment:
v[:, :, 2] = 0
```
This would modify the `tf.Variable` in-place.  For consistency with the proposed
`Tensor` API, we would also allow
```python
# Proposed tf.Variable copy-and-update assignment:
v = v.at[:, :, 2].set(0)
```
Which would create a modified copy of the original variable.


## User Benefit

Users will finally be able to "update" tensors by index using a friendly
NumPy-like slice syntax, while also keeping `Tensor`s immutable.  This is part
of a larger direction to introduce more Numpy-like behaviors in TensorFlow to
improve compatibility and the user experience.

## Design Proposal

We are proposing the addition of a new python API only - no new ops need to be
added, though some may be updated to support broadcasting to take full
advantage of efficient updates.  For consistency with NumPy and JAX, this API
should handle the following types of indexing:

- Basic indexing
  - single element: `x[1, 2, 2]`
  - slicing/striding: `x[:, 1:2:4, 2]`
- Advanced indexing
  - integer tensor indexing: `x[ [[1, 2], [3, 4], [5, 6]] ]`
  - boolean array indexing: `x[x < 0]`

We are _not_ currently targetting NumPy's [_combined_ basic and
advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing), since this is not currently
supported by our `Tensor.__getitem__` method, and this has not been identified
as a common use-case.

We will use the JAX `.at[]` syntax to access the elements to update.
Specifically, we will:
- Add a `.at` helper property to `tf.Tensor` and related
  types to enable this indexing behavior.
- For `tf.Variable`, overload `tf.Variable.__setitem__` to allow slice
  assignment directly.

Under the hood, these will call into existing tensorflow methods and kernels to
perform the copy-and-update operation.  See [Detailed Design](#detailed-design)
for more details about the implementation.

Along with `set(values)`, for consistency with JAX we will also provide:
- `add(values)`
- `multiply(values)`
- `divide(values)`
- `power(values)`
- `min(values)`
- `max(values)`
- `apply(ufunc)`
- `get()`

The initial implementation for these will simply use a get-modify-set pattern.
If use-cases are discovered where we need performant implementations, new ops
and kernels can be added in an ad-hoc fashion.  Note that performance can also
be achieved by taking advantage of XLA and jit-compilation via
`tf.function(jit_compile=true)` to fuse the operations internally.

### Alternatives Considered

#### Copy-And-Update Function

The main alternative considered was to introduce a new function:
```python
tf.copy_and_update(input_tensor, index_expression, updates)
```
The main difficulty is in constructing the `index_expression`.  NumPy _does_
already have a convenience function for this: [`np.index_exp`](https://numpy.org/doc/stable/reference/generated/numpy.s_.html),
so the overall syntax might look like:
```
x = tf.copy_and_update(x, np.index_exp[:, :, 2], 0)
```

- Advantages
  - More explicit that the tensor is being copied.
  - More consistent with exising TensorFlow free-function syntax.

- Disadvantages
  - More verbose.
  - `np.index_exp` is not very widely known or used.
  - New untested API.
  - Lack of consistency with JAX.
  
In discussions with the JAX team, they initially started with a syntax like
this, but eventually switched to the current `.at` syntax due to user feedback.
In particular, users found `index_exp` unintuitive, which led to syntax errors
when trying to use integer/tuple indices directly.  The `.at` approach was found
more readable and succinct.

#### Augment Existing Tensor Scatter Methods

We do already have a suite of ops that copy-and-modify the tensor:
`tf.tensor_scatter_nd_*`.  These accept single-index and integer-tensor
indexing. We could add new functionality to them to allow striding and boolean
masks.  However, there are a few drawbacks:
- The scatter/gather operations currently have a very specific meaning,
  and have a 1-1 correspondence with dedicated ops/kernels
  (e.g. `TensorScatterUpdate`).  If we change the meaning this will no longer
  apply, and gather/scatter would act more like dispatchers.
- The shape of index arguments to gather/scatter are inconsistent with the Numpy
  convention: one is the transpose of the other.  For example,
  ```python
  x = np.reshape(np.arange(6), [2, 3])
  indices = tuple([[0, 1], [0, 2]])
  
  y = x[indices]                               # [0, 5]
  z1 = tf.gather_nd(x, indices)                # [1, 2]
  z2 = tf.gather_nd(x, tf.transpose(indices))  # [0, 5]
  ```
  
We would therefore either need to break consistency with NumPy, or break
backward compatibility with existing usages - neither of which is desirable.

#### Mutable Tensors or tf.Variables

Making existing tensors mutable is not possible nor desirable due to
TensorFlow's current design.  We considered introducing a `MutableTensor` object
that could modify and track an underlying tensor object.

However, this would be very similar to `tf.Variable`, which does already allow
slice assignments via the `.assign()` member function.  Instead we propose to
allow direct assignment via `tf.Variable.__setitem__` to update a variable
in-place.  Technically `tf.Variable` could then be used as a proxy to update a
`tf.Tensor`:
```python
var_x = tf.Variable(x)
var_x[:,:,2] = 0             # Modify the Variable's underlying contents.
x = var_x.read_value()       # Extract the modified contents.
```
Forcing this to be the main update syntax for regular tensors seems a bit
clunky and heavy-handed, since we don't need all that `tf.Variable` provides.
We also cannot freely create `tf.Variable`s within `tf.function`, which would
limit usage.  The `.at[]` syntax provides a more direct option.
  
### Performance Implications

No new ops are to be added (at least not initially).  Instead, we propose
creating a simplified API around existing ones.  Performance for existing ops
will be unaffected.

Such a copy-and-update method can lead to poor practices: users may overuse and
abuse the function by setting many individual elements separately, leading to
excessive copying.  The hope is that by emphasizing that the method copies the
entire tensor in the docs, we can minimize this.  This concern is currently only
theoretical - we have not found such abuse via `tf.scatter_nd_update`.

For compound update methods such as `.at[].add()`, current implementations will
likely be slow since we do not have dedicated ops/kernels for sliced versions -
we will essentially need to perform
```
tmp = x[indices]
tmp = tmp + updates
x = x.at[indices].set(tmp)
```
to get-modify-set the values.  Additional ops/kernels can be added later as
performance requirements are uncovered.  These ops should be benchmarked
individually.  Alternatively, we can rely on XLA to fuse these operations within
a `tf.function(jit_compile=True)`, which is what JAX relies on.

### Dependencies

The proposed changes do not add any dependencies to TensorFlow.  Other projects
may choose to use the new API.

Some planned work around improving overall consistency with NumPy may be
simplified by these indexed updates.

### Engineering Impact

_Engineering Impact_: Changes to binary size should be minimal, since this is
mainly a python-only change.  This should not affect start up or run times.

_Maintenance_: the TensorFlow API team will maintain this code.  It will exist
under `array_ops.py` and `core.py`, and can be tested on its own.

### Platforms and Environments

_Platforms_: since this change does not involve new ops, it will be supported by
any platforms that support the current `scatter`/`where` ops.  All
transformation tools should be unaffected.

_Execution environments (CloudML, Cloud TPUs)_: depending on frequency of usage,
this may raise the priority of other existing API gaps for XLA.  For example, 
the XLA version of `TensorStridedSliceUpdate` op does not currently support
non-unit strides nor broadcasting.  It should otherwise work with Cloud TPUs.

### Best Practices

Due to copying of tensor values, the best practice will be to minimize the
number of calls to the proposed API (i.e. rather than setting single values
one-by-one, one should set an entire slice at once via a stacked tensor).
These best practices will be shared in a blog post, as well as in the
documentation for the new method(s).

We may choose to deprecate existing set/get methods such as
`tf.tensor_scatter_nd_update` in favor of the new API, since it is more general.
In this case, a note can be made in the documentation for the deprecated methods
with instructions on how to update.

### Tutorials and Examples

The update methods are meant as a drop-in replacement for numpy `__setitem__`.

```python
#======================================================================
# Basic indexing
#======================================================================

# Single index.
x = x.at[1, 2, 2].set(-1)

# Slicing and striding.
x = x.at[1, 1:4:2, 2].set(-1)

#======================================================================
# Advanced indexing
#======================================================================

# Integer array indexing.
indices = [[0, 0, 1], [0, 4, 3], [3, 3, 2]]
updates = [3, 4, 5]
x = x.at[indices].set(updates)

# Boolean array indexing.
x = x.at[x < 0.5].set(0.5)
```

### Compatibility

Since this change introduces a new API and does not add any ops, all backwards
and forwards compatibility requirements will continue to be met.

Similarly, all existing functionality should continue to work with other parts
of the TensorFlow ecosystem (TFLite, GPU/TPU, SavedModel), as long as the ops
themselves worked in the first place.

### Third Parties

This does not affect 3rd party partners, unless they wish to use the new API.

### User Impact

The proposed change will address the major pain point brought up by users
(see [Motivation](#motivation)).  We will announce the feature via a blog post,
and link to it on all the associated bugs.

There are no special requirements for rolling out the feature.  The Python API
can be added without changing existing models.

## Detailed Design

To introduce `.at`, we can use a Python `property` to allow it to treat
`Tensor.at` as a function call.  This function returns a helper object that can
be used with general indexing to update an underlying tensor:
```python
class TensorIndexUpdateHelper:
  """Helper class that allows copying/modifying a tensor at a set of indices."""
  def __init__(self, tensor: tf.Tensor, index_exp: Any):
    self.tensor = tensor
    self.index_exp = index_exp

  def set(self, values: tf.Tensor) -> tf.Tensor:
    """Replaces values in a tensor, returning the modified copy."""
    # Internal implementation that performs the copy-and-update operation.
    return tf.internal.copy_and_update(self.tensor, self.index_exp, values)


class TensorIndexUpdateHelperFactory:
  """Creates `TensorIndexUpdateHelper`s via __getitem__."""
  def __init__(self, tensor: tf.Tensor):
    self.tensor = tensor

  def __getitem__(self, index_exp: Any) -> TensorIndexUpdateHelper:
    """Creates a helper via Tensor.at[index_exp]"""
    return TensorIndexUpdateHelper(self.tensor, index_exp)


class Tensor:
  # ...
  def create_index_update_helper(self) -> TensorIndexUpdateHelperFactory:
    return TensorIndexUpdateHelperFactory(self)

  # Allows Tensor.at to be treated as a function call, returning a
  # `TensorIndexUpdateHelperFactory`.
  at = property(create_index_update_helper)
```

Through the indirection above, we have
```python
#   x.at                     # Returns a TensorIndexUpdateHelperFactory.
#   x.at[:,:,2]              # Returns a TensorIndexUpdateHelper.
y = x.at[:,:,2].set(values)  # Sets values in tensor at the specified indices.
```

For the `tf.internal.copy_and_update` functionality, we will modify the logic
already in [`array_ops._slice_helper`](https://github.com/tensorflow/tensorflow/blob/3efa4cb01c79e8aa46ca2e80af1a8f85bbc6435d/tensorflow/python/ops/array_ops.py#L915)
to extract the relevant slice information and call into the appropriate kernels
to replace values in the slice.  A preliminary version of this is already
available in the `tf.numpy` package via [`np_array_ops.__slicer_helper`](https://github.com/tensorflow/tensorflow/blob/3efa4cb01c79e8aa46ca2e80af1a8f85bbc6435d/tensorflow/python/ops/numpy_ops/np_array_ops.py#L1509).  The two helpers will be unified and properly tested
using NumPy's [slice testing](https://github.com/numpy/numpy/blob/c679644b0bc074ca27d9cd26b2aea7cbf4c6f60c/numpy/array_api/tests/test_array_object.py#L25).

## Questions and Discussion Topics

- Does the `.at[]` syntax seem appealing to the TF community?
- Should we consider deprecating existing gather/scatter methods in favor of
  the general API?
