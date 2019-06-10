# Standardizing composite ops in tensorflow.

Status        | Proposed
:------------ | :------------------------------------
**Author(s)** | Mark Sandler (sandler@google.com)
**Sponsor**   | Alexandre Passos (apassos@google.com)
**Updated**   | 2019-06-10

## Objective

The objective is to create a simple API that enables adding new composite ops in
a way that they can be automatically and robustly processed by downstream
tooling. The developers should be able to iterate on their implementation and
potentially add them as standard tf_op, without breaking existing tools (which
might not even be aware of their existence).

Why? Composite ops often provide building blocks for building complex models.
However, and this is especially true for embedded and specialized hardware,
these ops when implemented naively, become unusable due to architectural
(hardware) details. It is thus preferable for the downstream tools to be able to
extract such composite ops and for instance provide specialized implementation.

### Goals:

*   Create a standard way for tensorflow community to implement re-usable ops
    that can be efficiently processed by Tensorflow core tooling (such as
    TOCO/MLIR) as well as third party tooling (e.g. conversion to 3-rd party
    engines).
*   Maximize backward and forward compatibility of such composite ops, while
    enabling implementation changes later on.
*   Leave the path open for maintainer to switch to standard tf_op later on.
*   Bonus: enable *serialized* models to benefit from more efficient
    composite-ops implementations as underlying platform changes.

### Non Goals

*   Operation fusion or any graph optimizations. This should be handled by tensorflow
    compilers. Though part of this proposal might simplify detecting desirable
    transformations by MLIR and XLA frameworks.
*   Discussion what should live in "core" tensorflow (e.g. `tf.xxx.my_cool_op` )
*   Ultra complex functions (e.g. trained models) that are unlikely to get
    specialized implementations in hardware.

## Motivation

Historically, tensorflow API contained two types of operations:
“core” operators implemented in CUDA/C++, and composite ops that 
are implemented as a subgraph containing other tensorflow
operations. Here we will refer to "core" operators that have native
implementation as `tf_op`. As ops mature and gain adoption, efficiency often
dictates replacing composite op with their native implementation.

Some examples of ops that were composite at some point (or still are today):

*   Composite non-linearities (such as swish, tanh, and sigmoid);
*   many flavors of convolutions (such as atrous convolutions (expressible via
    transpose/batch_to_depth and regular convolutions), depthwise-convolution
    with depth multiplier);
*   Normalization methods (e.g. BatchNorm, Instance Norm, etc… ), some unusual
    flavors of convolutional padding, etc;
*   Advanced numeric functions (e.g. matrix exponentiation);
*   Combinatorial algorithms (e.g bipartite matching).

Many of these ops have since became standard tensorflow ops with efficient
native implementations.

It is important to note that many of the ops above precede or appeared during
the early days of Tensorflow, when compatibility with downstream tooling wasn't
that much of a concern. Though, for instance switch from non-fused batchnorm to
fused one, caused some disruption in early tensorflow processing tools. Some of which are still 
reflected in the [comments](https://github.com/tensorflow/tensorflow/blob/84c5a4551e2e71d854932cb389c359db11cfa2b1/tensorflow/python/ops/nn_impl.py#L1241).

Adding new operations today is much more complex due to existence of a large
eco-system of processing tools. Even within core
tensorflow there are multiple teams targeting dozens of hardware architectures
with various priorities so adding new operations (or even changing the implementation
of existing composite ops) becomes a bad case of chicken-and-egg problem.

Why is it important?

Today, new ML operations and composite blocks emerge regularly. They promise
improved functionality, efficiency and often both. These ops can often be
represented as
[simple](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py#L531),
and
[not-so-simple](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py#L220)
tensorflow subgraphs. On the ther hand, to get the full utility a custom
implementation on various platforms is necessary even when composite implementation
in tensorflow fully sufficient. For example using un-optimized
activations in mobile applications can increase latency on mobile devices by more than 100%. 
This presents a conundrum: should implementers of such operations create new atomic op and
provide efficient implementations for their target platform and break everyone
else? Or should they provide tensorflow based graph implementation and then rely
on graph pattern recognition to extract and match in the tooling of their
choice.

Who is affected? Tensorflow users, software and hardware vendors.

### Existing prior art.

Tensorflow Lite have developed [tf.lite.OpHint](https://www.tensorflow.org/api_docs/python/tf/lite/OpHint),
which solves a very similar problem. However it is positioned as tf.lite
specific extension, which doesn't provide a public api for graph consumers
(other than TOCO) to extract the hints, limiting the potential for broader
adoption by other tooling and limiting its usefulness to the users.

`tf.lite.OpHint` also adds wrong direction to the flow
of dependencies from tensorflow to lite, should core tensorflow api choose to 
use to annotate composite ops.

Pattern matching is another approach that is currently used. For example TfLite
has pattern matching code that tries to detect various patterns that can then be
converted into specialized ops, such as
[prelu](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_prelu.cc)
and
[lstm](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_lstm.cc)
and others files in tf.lite directory.

## User Benefit

The main beneficiaries would be:

a) ML community that get a clean way of defining of composite ops without having
to commit to particular implementation (or whether to build new `tf_op`)

b) Tool maintainers that wouldn't need to write complicated and brittle graph
extraction code whose sole purpose is to reverse engineer tensorflow
implementations. For example here are some tf-lite transformations that identifies
composite ops like [prelu], [lstm], [l2 pooling], etc. Which essentially
requires updates to lite whenever tensorflow modifies those implementations.

_Blog post announcement_: "How to add new functions to tensorflow"

_Subtitle_: "So you don't regret it later".

## Design Proposal

Tensorflow already provides a mechanism to define functions inside a graph
called `tf.function`. In short, `tf.function` allows to define a subgraph with
dedicated inputs and outputs, that then is substituted as a custom node into
graphs. The gist of our proposal is to add a few standardized attributes to
signal to the processing/executing library that this function implements a
“standard” function. The consumer library then can choose to process it
differently should they choose to. The important part is that we define a
standard set of attributes that users and tooling can rely on.

As a start, we propose to add “implements” and “reference” as two attributes
defining a “standard” function.

For example:

```
@function.library_function(
    implements=“google.matmul_low_rank_matrix”,
    reference=“arxiv.org/…”)
def matmul_lowrank(x, y):
   “Multiplies two low rank matrices. ”
   # Basic implementation goes here.
```

Note we use a wrapper named library_function, that under-the-hood will create a
standard tensorflow function with standard attributes set.

Arguments:

* `implements`:  we propose using namespaces with the top-level namespace
being company name, with a few reserved namespaces indicating built-in ops, 
if so desired.  This attributes indicates what function this subgraph implements,
and provides a hint for downstream parsers that they can replace this with 
their own implementation should they so choose. 

* `reference`:  The goal for this attribute is to provide a stable reference for the function
implementation details. This could be a reference to non-tensorflow library (e.g. numpy.linalg.special_matmul)
or arxiv reference. The downstream tooling can use this value as an opaque cache to verify that 
it implements the right flavor of the function. 


This brings us the following advantages:

1) *Backward and fork compatibility*: the tooling by default will,
ignore `implements` and just rely on function implementation available in the
graph, thus ensuring that it “just works”. Further, if users have their
proprietary implementations they can still ship models that would transparently
work in open source tensorflow.

2) *Easy detection*: Tools can easily detect the functions attribute and substitute with
its own more efficient implementation if available.

3) *Simplified implementations on custom hardware*: The downstream tooling can
use provided reference implementation as a source of ground truth (that can run
on custom hardware out of the box!) when developing custom implementation on
their hardware.

4) *Reduced implementation dependencies*: Operation maintainer can change the
implementation without the fear of breaking the downstream tools pattern
recognition (since there isn’t any.)

5) *Forward compatibility*: Operation maintainer can even add an atomic
implementation without breaking existing tools. No overhead in keeping the basic
implementation available.

6) *Simpler automatic conversion* As tensorflow moves forward it is conceivable
that its computation description language (today: graph def, tomorrow:MLIR) 
will be fully  be separated from the underlying tensorflow kernel implementations.
This kind of standard attributes allow for more  automatic conversions
should the need arise.

7) Does not change op developer workflow and can be introduced incrementally to
existing code.

### More on forward compatibility

Suppose later, the maintainer of the op decides that certain composite op
deserves its own atomic op due to its usefulness. If we have standardized
attributes like the one above, the XLA compiler can check if op with the name
that this function “implements” is available and substitute it at runtime. The
significant advantage is that even old serialized models will be able to take
advantage of improved implementations. As the op gains more adoption and its
support becomes more widespread the previous composite definition can be
deprecated or retired. However such an approach allows both implementations to
coexist for unlimited period of time and the same graph can be both: efficient
on the platforms that support it, and “not-broken” on the platforms that don’t.

### Alternatives considered

#### Alternative 0: Do nothing

Everyone just continues with adhoc implementations.

##### Pros:

*   No need for RFC or new public API

##### Cons:

*   Downstream tools are stuck between (a) trying to implement ops, even if the
    efficiency is not yet a concern. In which case this just redos
    implementation in the downstream tooling using its own language. Or (b)
    trying to extract certain graph patterns that can be optimizing and
    following the moving target. Once such pattern is extracted there is a
    pressure on op-maintainers to freeze his implementation to avoid breaking
    the patterns used by the downstream tools. Or (c) Tring to roll out their
    own extension such as `OpHint` for tflite.

*   New ops that could push the industry forward however are stymied due to lack
    efficient implementation, software vendors are not interested in
    implementation until the ops become popular enough, creating a vicious
    cycle.

#### Alternative 1: Recommend each new op to be implemented as a new atomic node in graphdef.

#### Pros:

*   Simplified graph processing since the operation is just a single Node in the
    graph.
*   Clean graph def.
*   Tooling (once supports new ops) stays forward compatible. The underlying
    implementation is invisible.

#### Cons:

Introduces backward incompatible changes that could be avoided. Every time an
operation is added, the versions of Tensorflow that would have been capable of
processing the original graph (with core ops), will no longer be able to read
the graph def that contains the new to process the new version, even though the
only thing that *has* changed is the tensorflow added a faster implementation of
that Op.

For example: consider matrix exponentiation. It is implemented as a fairly
complex tensorflow graph that uses just regular matrix multiplication and other
standard ops. However, one can easily imagine that this implementation could be
highly optimized if done as a single expm node, however if we replace that, it
will break old tools, which would either need to effectively re-implement
tensorflow original implementation.

Requires custom version of tensorflow to add new ops outside of tensorflow which
makes it out of reach for most users and introduces incompatibilities within the
ecosystem, effectively forcing large users to be stuck at old versions of tensorflow.

#### Alternative 2: Use name scopes for delineation of the ops

Pros:

*   Simple and intuitive graph-def
*   Backward compatible - no new operations are added.

Cons:

*   very brittle for the tools to optimize such ops. If they depend on scope
    names it can easily cause conflicts with models that are doing something
    else or accidental renames can cause scopes to become invisible for the
    graph processing.
*   If tools depend on graph pattern matching, this makes it hard to change
    implementations later on.

*   Tooling is not forward compatible.

## Questions and Discussion Topics

1.  We should provide namespacing for the operation names early on to avoid
    future collisions with function names. One option is to adopt java-style
    "org.opname", basically using the developing organization as disambiguating
    factor. Another alternative is to use semantic namespaces e.g. `linalg`.  Yet
    third, is to use org.semantic.opname.
    
    Note: since many of these functions will live in user codebase we obviously
    can't and shouldn't police them, however having a recommended style guide
    will simplify things later on.
    
    Should there be a `graduation` process? Where ops eventually move to a higher tier.
    E.g. dedicated top-tier op like tensorflow.opname?
    If so, we might also consider having another attribute 'aliases' to allow
    history of names for backward compatibility.

2.  If we go org/name, do we need centralized registry of accepted "org" names?

3.  Do we need `reference` field? The isdea is that points to the source of
    authoritative ground truth implementation (likely not tensorflow
    based), which this op is faithfully trying to reproduce? Should this
    reference be structured? (For example it could point to existing
    scipy/numpy/pillow library, or it could be pointing to a paper?). Should we
    make this optional or get rid of it altogeher and delegate this to
    documentation?
