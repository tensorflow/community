#  tensorflow/api-owners review practices

## Overview

This is an attempt to gather commonly discussed topics when doing API
reviews. It’ll hopefully be a useful resource to both API owners and people
proposing API changes.  TF API Owners team (@tensorflow/api-owners) 
meet twice weekly to discuss changes. 
We try to get to PRs on the next meeting, 
but we don’t always make it all the way through. If your change is particularly 
urgent, please ping the PR to notify us of any urgency.

Please note that also if the TF API team is public Github public teams are 
[only visible](https://docs.github.com/en/github/setting-up-and-managing-organizations-and-teams/about-teams#team-visibility) 
and mentionable by TF organization members.

## Process

We only look at changes which have already been approved by other reviewers. If
there are major outstanding comments, we will wait with API review until those
are resolved. If there are questions for API owners, explicitly raise this in
the comments to get an answer.


## High level points

### Python Versions

TensorFlow supports a range of Python versions and changes need to be
compatible with all of them. This means that language features not available in
any of [TensorFlow's supported versions](https://www.tensorflow.org/install) cannot be used.

We regularly reconsider the range of supported versions based on the number of
affected users (estimated via pip downloads by Python version), and the default
installed version of Python on important platforms (Linux distributions, OSX,
...), but these are large, breaking changes to the ecosystem which are not
triggered by reviews of individual features.

### Backward and forward compatibility

We avoid backwards-incompatible API changes. We also avoid
backwards-incompatible behavior changes, such as restricting the set of valid
inputs to a function or extending the set of valid outputs of a function. Adding
support for previously not supported behavior is okay, as are changes to
explicitly experimental APIs (see section below). When needing to provide a new
or different behavior, we strongly prefer a new version of the API over breaking
backwards compatibility. Note that we are free to deprecate APIs; we just cannot
break code which relies on their documented behavior. We need to worry about
backward compatibility both of our python APIs and of the serialized GraphDefs,
and in general breaking serialized GraphDefs is worse than breaking the python
APIs.

Forward compatibility is more subtle: we should avoid changing the graph
produced by currently correct python code without a three weeks notice. This
comes up most frequently when adding new ops, but also applies to non-obvious
things such as the graph emitted by gradients or pfor.

### TensorFlow 1.x and 2.x

When adding new API end points, we use the `@tf_export` decorator. For *new* endpoints,
expose them to the v2 API only: `@tf_export("some_name", v1=[])`. This prevents
unnecessarily diverging the `compat.v1` namespace in future releases from what is
published in TensorFlow 1.15.

### Docstrings 

TF APIs should have comprehensive documentation in the form of docstrings. If at
all possible these docstrings should have runnable examples, and these examples
should form a doctest so they stay correct. The examples should demonstrate an
end-to-end user workflow, such that it’s clear how to generate the necessary
inputs for the API and what to do with the outputs. The docstring should be
understandable by someone who is not familiar with TF. See the [guide to writing
TF docstrings](https://www.tensorflow.org/community/contribute/docs_ref) for
more information.

Our documentation generator for classes only sees methods, so prefer defining
members as properties instead of assigning them in `__init__`.

Docstrings should only refer to other public TF API symbols (i.e. do not refer
to other symbols defined in the same file as a function which is just now being
made public) and should refer to public API symbols by their full exported name.

### Common names

Prefer keepdims over keep_dims. Prefer axis over dim. Data types are called
dtype. name is a common last argument of ops but backward compatibility mandates
that new arguments are added after the last existing argument, even if that
results in name not being the last argument.

We generally prefer spelling things out over using abbreviations except when
abbreviations are more standard than spelling things out (i.e. don’t spell out
linalg or svd). When in doubt we ask domain experts or use web search to see
what spelling is most common.

If possible we prefer to name things in a similar way to numpy (e.g., we would
not pick einsum as a name, but numpy and others before it have, and that
precedent is very strong).

We prefer experimental namespaces (i.e. tf.data.experimental.foobar) over
experimental-prefixed names (i.e. tf.data.experimental_foobar) except when
adding an experimental class method, or an experimental argument. Experimental
endpoints should be deprecated in a minor release before they can be removed in
the next. We would like new experimental symbols to be things which will
eventually end up in core TF as opposed to things we expect will be phased out
with no clear replacement. The best expectation to have for an experimental
endpoint is that the “experimental” will simply be removed. If you don’t believe
that’ll work, it should probably not be added in its current form.

### Style

Generally, follow Google style.

Avoid redundancy. Do not write arguments of the form `function(...,
enable_feature=False, feature_config=None)` if you can also write `function(...,
feature_config=None)`, where implicitly, `enable_feature = feature_config is not
None`.

Try to embed well with the ambient language. Think about how your API interacts
with language idioms (e.g., in Python: can it be hashed, i.e., used as a dict
key? Is it iterable? Is it a mapping? Can it be equality compared?
Ordered?). Think about how your API interacts with other pieces of the Python
ecosystem as well— is there an analogue in Numpy or PyTorch that we should
consider aligning with?

Use language-native constructs wherever you can. In Python, a tuple should be a
tuple. The bar for custom configuration objects is relatively high, a dict or
namedtuple goes a long way.

In particular, do not expose protobufs directly as part of an API. You can use
protobufs for serialization or to encode network traffic. Protobufs should
always be an implementation detail, and never visible on the API surface. Use
language native constructs (dicts or classes for Python, structs for C/C++) if
you need configuration objects.

Avoid global (or any non-local) state as much as possible (this includes Python
'with' scopes). If you need global context, consider whether it can be
thread-local. The TF API is supposed to be thread-safe. Avoid stateful operation
(mutability) if you can. Both features make it hard to reason about code, and
make composability harder to achieve.

We prefer strings ("auto", "never", etc) over enums (tf.namespace.AUTO,
etc). Strings are easier to type, and forces us to document all possible values
and their semantics in the docstrings of all places which accept the string, as
opposed to only in the enum definition, which is a little friendlier.

### Orthogonality and integration with the existing APIs 

Is the new API implementable in terms of existing APIs? If so, we might want to
consider pointing users to using the existing APIs. Does the new API add enough
value over a combination of existing APIs? Does the API solve only a specific
problem (that’s usually a sign combinations of existing APIs would be
preferred)?

If not, are existing APIs implementable in terms of the new API? If this is
simple, we might want to steer users towards the new and away from the old API
(possibly, old APIs should be deprecated along with introducing the new API).

If neither is the case, it might be possible that there is a more general API
which makes both the existing API and the new API easy to express. We try to
keep global consistency of our API in mind when reviewing new changes.

How will this API work together with others? Does it do something slightly
differently than others? Does it expect inputs which match what other parts of
TensorFlow produce? Does its output match what other parts of TensorFlow can
consume?

Does it do things the same way other similar pieces in TensorFlow do it? E.g.,
if a common pattern to achieve a behavior is an extra argument, don't use a
function decorator to achieve the same in a different area of the API.

Two wrongs don’t make a right. That is, if a bad API already exists in TF, that
does not give license to new APIs to be bad in the same way. Improvement must be
balanced with consistency, however, and sometimes it’s okay to carry small
imperfections into new APIs for the sake of consistency with old APIs.

### Optional arguments with default values

Many APIs have optional arguments that assign a default value (for example, `activation = None`
in [`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)).

Our recommendation is to use `None` as the default value for _any optional arguments
that may be adjusted or changed over time_, and have the implementation be responsible
for handling the value, as opposed to using a default value that directly represents 
the behavior (e.g. `num_threads = 10`).  The latter prevents the implementation from
distinguishing between the caller not setting the argument vs. the caller setting the
argument to the default value, which may be needed when the default behavior is changing.

If the optional argument is _backwards incompatible to change_, however, its default should
reflect the actual default value when possible.

#### Documented types

Arguments and return values to public APIs must be either public types, or
inherit from a public type. This ensures that the argument types and return 
value types are documented and gives users clear guidance on what can be 
passed to a public API, and what can they do with the returned values. 
If it is not desirable for the user to construct these types on their own, 
one can choose to expose superclass with no constructor, but adequate docstrings.

### Does it belong in TF at all?

As TF evolves there’s a tendency to put everything inside of it, with costs
compounding over the long term. If there is a reasonable home for a new API
outside core TF (say in addons, io, TFP, or other projects entirely) that can be
strongly preferrable. If new code can be released as independent libraries, it
should be. This is especially true for APIs that are actively evolving; core TF
imposes many restrictions, so it’s far better to trial new APIs outside of the
core library.

## Adding new ops 

Adding new ops to TF should be done with care. We generally prefer not adding
new ops if possible, but performance, hardware compatibility, and other concerns
often do require new ops.

When adding new ops, look for:

 - closure under automatic differentiation (i.e. we avoid ops which are
   differentiable but not twice-differentiable, or which are technically
   differentiable but not marked as such)
 - performant kernels (it’s better not to have an op than to have an op with a
   suboptimal kernel; we need to make sure kernel experts have reviewed the
   code)
 - broadcasting (all numerical ops should broadcast using numpy rules)
 - does support for this op have to be added to pfor/vectorized_map?
 - dtype support (in general all numerical ops should support the common
   integer, floating point, and complex dtypes, if they all make sense; we need
   to watch out for int32 on GPUs though)
 - device support (cuda kernels should be implemented if possible, and similarly
   a tf/xla bridge entry should be added if it makes sense)
 - attributes versus inputs (anything which can be an input to an operation
   should be an input, and attributes should only be used to parametrize the
   operation in ways that affect the output dtypes or sometimes shapes)
 - state management (is the op stateful? Can it instead be made stateless and
   rely on optimizations like memory reuse for performance? Can it be made to
   keep its state using one of the existing mechanisms like variables? If not,
   its state should be encapsulated using resource handles if at all possible)
 - we generally don’t like ops which are supported only on a single device (be
   it CPU, GPU, XLA, TPU, etc) and prefer to have at least a plan for writing
   device-agnostic code
 - should the python layer for this operation support raggedtensor/sparsetensor?

## Experimental APIs

Experimental APIs are APIs which have the word 'experimental' somewhere in their
name; for example `tf.experimental.foo`, or `tf.foo.experimental.Bar`, or
`tf.foo(experimental_bar=True)` or `tf.Foo().experimental_bar()`. We generally
prefer experimental namespaces when possible, so prefer
`tf.foo.experimental.Bar` over `tf.foo.ExperimentalBar`.

Experimental APIs are APIs intended to be added to TensorFlow as-is, but which
we reserve the right to change in backwards-incompatible ways if we have
to. This is different from apis in `tensorflow/addons`, many of which are not
necessarily intended to be added to core TF as they might have a more narrow use
case initially (if APIs in `tensorflow/addons` do become widely useful they can
"graduate" to core, either using experimental or not).

No temporary APIs should be added to experimental (i.e. "we just need this until
certain bugfix or certain new feature becomes available" is not a valid reason
to add an API with experimental in the name.) 

No API with known deficiencies should be added to experimental. Experimental
APIs should, to the best of our knowledge, not be expected to change in a known
way (no argument with a known bad name, etc). Experimental can, however, be used
for APIs which are a work-in-progress: it's fine to add experimental methods to
a base class even if those methods are only implemented on some subclasses as
long as we expect all classes to eventually implement those.

The same amount of due diligence required for a real API is required for an
experimental API: this means tests, benchmarks, documentation, end-to-end
examples, etc

Experimental APIs are not a license to break users. This means:
 1. we do not remove experimental APIs which are widely used without an effort
    to help migrate users away
 2. experimental APIs are not removed without warning and don't have
    backwards-incompatible changes made to them without warning (the warning can be
    a deprecation on version 2.x and removal on 2.x+1, but plain removal on 2.x
    with no notice on 2.x-1 is not ok)

Small changes which are mentioned in relnotes and have obvious fixes might be
made (for example if adding a new argument to a long argument list and we
believe there are few pass-by-position users we might allow the new argument to
be added to the middle and not the end of the parameter list).

Large backwards-incompatible changes to experimental APIs still require an
`experimental_foo_v2` or similar backwards-compatible evolution plan to avoid
breaking users of the existing experimental API.

No API endpoint should stay in experimental forever. If a particular
experimental API hasn't had major changes in two minor releases we should remove
the experimental annotation from the API name or delete it. If we do want to
delete it we need to have a deprecation plan that can migrate all users to some
other API endpoint or composition of existing APIs. In rare cases experimental
APIs can continue to be iterated on after many releases (see TPUStrategy); this
only applies for fairly large API surfaces.

When removing the experimental annotation we should, if at all possible, allow
escape routes to not break existing code. This means toplevel symbols
`tf.experimental.foo` and methods like `tf.Class.experimental_foo` should get a
deprecation warning on 2.x before deletion on 2.x+1; we should use the
doc_controls decorators to not pollute API docs with deprecated "graduated"
experimental APIs. For experimental function arguments we should consider
catching `**kwargs` to raise the proper warnings for at least one version (note
though that `**kwargs` is generally discouraged from our APIs; we prefer
explicitly named keyword arguments if at all possible).
