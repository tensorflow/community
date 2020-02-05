# TensorCord Variant Object

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Eugene Brevdo (ebrevdo@google.com) |
| **Sponsor**   | Alex Passos (apassos@google.com)                 |
| **Updated**   | 2019-06-05

## An update on `TensorCord Variant Object` RFC.

After some internal discussion, we have decided to merge this RFC into another
planned update to how TensorFlow runtime represents and handles strings.  When
that updated proposal is available, it will include a section on transparently
representing rope-like objects (without having to use `Variant` for this
behavior).

## Objective

This document proposes a new Variant object called TensorCord.  A TensorCord
contains a list of `absl::string_view` and release functors / tensor references; and
can be used to store references to other Tensors or other memory locations.  The
destructor of a TensorCord reduces the count on its referenced releasers; once
the reference count of a releaser reaches 0, it is executed.

## Motivation

A number of ops within TensorFlow could use a rope-like object that
is propagated through the graph.  Examples are:

* `tf.data` input pipelines that perform a lot of string munging (substrings and
  concats, especially); these incur a lot of copy overhead.
* `tf.contrib.rpc` and `tf.io.*_proto` (prev. `tf.contrib.proto`) ops that
  handle strings and messages and submessages.  For example, in proto
  encode/decode ops, encoded submessages are contiguous substrings of serialized
  messages.  Decoding is therefore done by copying substrings on decode, and
  encoding is performed by concatenating substrings on encode.
* When serializing numerical tensors, there is currently no equivalent to
  `tf.io.decode_raw`.  `tf.io.encode_raw` would make sense, but it would incur
  the overhead of copying the tensor data into a new string.  A more efficient
  approach is to create a TensorCord pointed at the old tensor.
* Strings coming in from network I/O are copied out of protos and into
  tensors, which also incurs a copy overhead.

## User Benefit

Faster input data pipelines and handling of strings and views for users (in a
transparent manner).

## Design Proposal

The TensorCord object itself can be implemented using RefCounted objects with a
constructor that takes an `absl::string_view` and either a releaser callback or
a pointer to a `Tensor` or `RefCounted`.

Below is an example of use:

```c++
auto t = strings.flat<string>();
// old way via copy:
Tensor copy = tensor::DeepCopy(strings);

// new way: a referencing view:
Tensor view(DT_VARIANT, {strings.NumElements()});
auto t_view = view.flat<Variant>();
for (int i = 0; i < num_elem; ++i) {
  t_view(i) = TensorCord(t(i), &strings);
}
```

## Alternatives Considered

A new tensor type `DT_CORD`, which is a tensor of arrays of
`absl::string_view` objects, and additionally has a releaser that runs on its
unref. This implementation seems to be faster but much more invasive from an API
standpoint; e.g. it adds a `CHECK` to the `Tensor::Tensor(dtype, shape)`
constructor so users don't accidentally create a `DT_CORD` tensor without
a releaser.

| Alternatives | TensorCord DT_VARIANT | DT_CORD |
:------------- |:--------------------- |:------- |
| Separate releasers per element | Yes | No |
| Overhead | Higher (each Tensor element keeps a reference; Variant & RunVariantDtor overhead is more costly) | Lower (can have onereleaser per tensor) |
| Intrusiveness | Lower (use DT_VARIANT) | Higher (add new TF type) |
| Flexibility | High (elements can point to locations backed by different owners) | Lower (all elements must be backed by data whose lifetime depends a shared set of releasers)

## Detailed Design

### Public C++ API

The TensorCord object constructor and Append methods accept a string_view and
either a Tensor pointer or releaser callback.  Its underlying string views can
be iterated over or a string can be constructed via explicit cast:

```c++
class TensorCord {
public:
  typedef void (*Releaser)(void*);

  // At final destruction, releaser will be called as releaser(memory).
  // To create a releaser for a std::function that captures objects, use:
  //
  //  template <typename T>
  //  TensorCord::Releaser CreateThunkFor(const T& fn) {
  //    return [](void* ptr) { (*static_cast<T*>(ptr))(); };
  // }
  //
  // auto fn = [&]() { ... };
  // auto releaser = CreateThunkFor(fn);
  // auto tc = TensorCord(view, releaser, &fn);
  //
  // Remember that in this case, fn needs to outlast the TensorCord.
  //
  // Creates a TensorCord from `view`, with memory releaser `releaser` and releaser
  // arg `memory`.
  explicit TensorCord(absl::string_view view, Releaser releaser,
                                  void* memory = nullptr);

  // Creates a TensorCord from `view`, with memory backed by `tensor`.  If it `view`
  // is small enough, no reference is created on `tensor`; instead the memory is
  // stored inline.
  explicit TensorCord(absl::string_view view, Tensor* tensor);
  explicit TensorCord(absl::string_view view, RefCounted* ref_counted);

  void Append(const TensorCord& other);
  void Append(absl::string_view view, CordRep::Releaser releaser,
                        void* memory = nullptr);
  void Append(absl::string_view view, Tensor* tensor);
  void Append(absl::string_view view, RefCounted* ref_counted);

  size_t size() const;
  bool empty() const;

  explicit operator string() const;

  // Usage example:
  //  for (absl::string_view s : cord.Chunks()) { ... }
  ChunkRange Chunks() const;
  ChunkIterator chunk_begin() const;
  ChunkIterator chunk_end() const;

  // Copy and move constructor, copy and move assignment operators.
  // And all the associated Variant-stored object methods
  // (Encode, Decode, DebugString, etc).
};
```

### Ops and Op extensions supporting TensorCord:

The following ops would be extended to support TensorCord:
* basic string ops (join, concat, reduce_join, as_string)
* `tf.contrib.rpc`
* `tf.io.*proto`
* Example parsing ops (`Parse{Single,}{Sequence,}Example`)

A new op would be added to create views into dense tensors as TensorCord
objects.

### Python API

We create a new TensorCord python object:

```python
class TensorCord(composite_tensor.CompositeTensor):

  def __init__(self, variant):
    self._variant = variant
    self._as_string = None
    if not in eager or tf.function mode:
      self._as_string = tf.strings.as_string(variant)

  def as_string(self):
    if self._as_string is None:
      self._as_string = tf.strings.as_string(self._variant)
    return self._as_string

  @property
  def variant(self):
    return self._variant

  def _to_components(self):
    return (self.variant,)

  @classmethod
  def _from_components(cls, components, metadata):
    variant, = components
    return cls(variant=variant)

  @property
  def _is_graph_tensor(self):
    return getattr(self._variant, "graph", None) is not None

  # also properties/methods like name, op, dtype, graph, shape, numpy, etc.
```

Additionally we add a conversion object for `convert_to_tensor(cord, dtype)` to
return `cord.as_string()` when `dtype=string` and return the variant otherwise;
and a similar conversion for `session.run()`:

```python
def _tensor_cord_to_tensor(value, dtype=None, name=None, as_ref=False):
  if as_ref:
    raise ValueError
  if dtype == dtypes.string:
    return value.as_string()
  elif dtype in (None, dtypes.variant):
    return value.variant
  else:
    raise ValueError("Don't know how to convert TensorCord to dtype {}".format(dtype))

ops.register_tensor_conversion_function(TensorCord, _tensor_cord_to_tensor)

def _tensor_cord_session_fetch(tensor_cord):
  return ([tensor_cord.as_string()], lambda val: val[0])

session.register_session_run_conversion_functions(
  TensorCord,
  fetch_function=_tensor_cord_session_fetch)
```

## Performance Implications

**NOTE** The statement below requires an upcoming PR that allows inlining small
values inside a `Variant` object.

TL;DR: Creating a TensorCord view of full strings of a `DT_STRING` tensor is
1-1.25x more expensive than a direct copy of `DT_STRING` unless the string lengths
are approximately 128 bytes each.  Once string lengths on average are >128
bytes, the TensorCord approach is more performant.

We are able to match or exceed `DT_STRING` performance by using a specialized
implementation of TensorCord and modifications to the `Variant` class:

* TensorCord performs optional inlining and selective referencing of backing
  Tensors (usually for strings < 32 bytes in size).  This requires a specialized
  constructor that knows about `Tensor` objects.
  
* The Variant object is modified to inline its underlying data if the stored
  value is <= 48 bytes in size (leaving 16 bytes for alignment + additional
  stored Variant data).  This reduces the amount of overhead and indirection in
  storing small values like TensorCord inside Variant and greatly reduces the
  cost of `DT_VARIANT` tensor destruction.  It keeps the Variant object <= 64
  bytes, which is the per-element aligned size inside `Tensor`
  buffers.

## Questions and Discussion Topics

* Are there other good rope-like alternatives?
* Any python API considerations?
* Do we need additional Python code to create TensorCord objects from python
  strings?
* Considerations for TF-Lite if we extend a number of string processing ops to
  using `DT_VARIANT` inputs.
