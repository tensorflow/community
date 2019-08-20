# Standardizing composite ops in tensorflow to support efficient inference.

Status        | Accepted
:------------ | :------------------------------------
**Author(s)** | Mark Sandler (sandler@google.com)
**Sponsor**   | Alexandre Passos (apassos@google.com)
**Updated**   | 2019-06-10

## Objective

The goal of this proposal is to create a simple API that allows
adding new composite ops in a way that they can be automatically and robustly 
processed by downstream _inference_ tooling. The developers (of the composite op) should be able to
iterate on their implementation or even replace them with standard tensorflow
op, without breaking any existing tools.

Why? Composite ops often provide building blocks for building complex models.
However, and this is especially true for embedded and specialized hardware,
these ops when implemented naively, become unusable due to architectural
(hardware) details. It is thus preferable for the downstream tools to be able to
extract such composite ops and for instance provide specialized implementation.

The current proposal concentrates on supporting inference-only transformations
and optimizations. However we leave the door open for follow-up gradient
optimization support. See appendix for a few possible ways forward.

### Goals:

*   Create a standard way for tensorflow community to implement re-usable ops
    that can be efficiently processed by Tensorflow core tooling (such as
    TOCO/MLIR), grappler, as well as third party tooling, such as conversion to
    3-rd party engines (e.g. TFLite).
*   Maximize backward and forward compatibility of such composite ops, while
    allowing changes to implementation including switch to native tensorflow op. 
*   Provide for *future* support of composite op gradient extraction and
    optimization by downstream tools.
*   Bonus: enable *serialized* models to benefit from more efficient
    composite-ops implementations as underlying platform changes.

### Non Goals

*   Operation fusion or any graph optimizations. This should be handled by
    tensorflow compilers. Though part of this proposal might simplify detecting
    desirable transformations by MLIR and XLA frameworks.
*   Discussion what should live in "core" tensorflow (e.g. `tf.xxx.my_cool_op` )
*   Ultra complex functions (e.g. trained models) that are unlikely to get
    specialized implementations in hardware.
*   Immediate support for processing gradients (and forward inference in the presense of gradients)
    for composite ops. 

## Motivation

Historically, tensorflow API contained two types of operations: “core” operators
implemented in CUDA/C++, and composite ops that are implemented as a subgraph
containing other tensorflow operations. Here we will refer to "core" operators
that have native implementation as `tf_op`. As ops mature and gain adoption,
efficiency often dictates replacing composite op with their native
implementation.

Some examples of ops that were composite at some point (or still are today):

*   Composite non-linearities (such as swish, tanh, and sigmoid);
*   many flavors of convolutions (such as atrous convolutions (expressible via
    transpose/batch_to_depth and regular convolutions), depthwise-convolution
    with depth multiplier);
*   Normalization methods (e.g. BatchNorm, Instance Norm, etc… ), some unusual
    flavors of convolutional padding, etc;
*   Advanced numeric functions (e.g. matrix exponentiation);
*   Combinatorial algorithms (e.g bipartite matching and nms)
*   Specialized losses CTC loss, RNN layer steps
*   tf.einsum

Many of these ops have since became standard tensorflow ops with efficient
native implementations.

It is important to note that many of the ops above precede or appeared during
the early days of Tensorflow, when compatibility with downstream tooling wasn't
that much of a concern. Nevertheless, for instance the switch from non-fused batchnorm to
fused one, caused some disruption in early tensorflow processing tools. Some of
which are still reflected in the
[comments](https://github.com/tensorflow/tensorflow/blob/84c5a4551e2e71d854932cb389c359db11cfa2b1/tensorflow/python/ops/nn_impl.py#L1241).

Adding new operations today is much more complex due to existence of a large
eco-system of processing tools. Even within core tensorflow there are multiple
teams targeting dozens of hardware architectures with various priorities so
adding new operations (or even changing the implementation of existing composite
ops) becomes a bad case of chicken-and-egg problem.

Why is it important?

Today, new ML operations and composite blocks emerge regularly. They promise
improved functionality, efficiency and often both. These ops can often be
represented as
[simple](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_impl.py#L531),
and
[not-so-simple](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py#L220)
tensorflow subgraphs. On the other hand, to get the full utility a custom
implementation on various platforms is necessary even if composite
implementation in tensorflow is  sufficiently performant. For example using un-optimized
activations in mobile applications can increase latency on mobile devices by
more than 100%. This presents a conundrum: should implementers of such
operations create new atomic op and provide efficient implementations for their
target platform and break everyone else? Or should they provide tensorflow based
graph implementation and then rely on graph pattern recognition to extract and
match in the tooling of their choice.

Who is affected? Tensorflow users, software and hardware vendors.

### Why no gradients?

Adding gradient support would dramatically widen the scope of this proposal. See
appendix for details on why it is complicated. We also have outlined several
possible options to add gradient support on top of this proposal, depending
on the needs.

### Existing prior art.

Tensorflow Lite have developed
[tf.lite.OpHint](https://www.tensorflow.org/api_docs/python/tf/lite/OpHint),
which solves a very similar problem. However it is positioned as tf.lite
specific extension, which doesn't provide a public api for graph consumers
(other than TOCO) to extract the hints, limiting the potential for broader
adoption by other tooling and limiting its usefulness to the users.

`tf.lite.OpHint` also adds wrong direction to the flow of dependencies from
tensorflow to lite, should core tensorflow api choose to use to annotate
composite ops.

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
implementations. For example here are some tf-lite transformations that
identifies composite ops like
[prelu](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_prelu.cc),
[lstm](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_lstm.cc),
[l2 pooling](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/graph_transformations/identify_l2_pool.cc),
etc. Which essentially requires updates to lite whenever tensorflow modifies
those implementations.

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

As a start, we propose to add “implements” as a new attribute
defining a “standard” function.

For example:

```
@function.function(
    implements=“google.matmul_low_rank_matrix”)
def matmul_lowrank(x, y):
   “Multiplies two low rank matrices. ”
   # Basic implementation goes here.
```

Note we use a wrapper named function, that under-the-hood will create a standard
tensorflow function with standard attributes set.

Arguments:

*   `implements`: we propose using namespaces with the top-level namespace being
    company name, with a few reserved namespaces indicating built-in ops, if so
    desired. This attributes indicates what function this subgraph implements,
    and provides a hint for downstream parsers that they can replace this with
    their own implementation should they so choose.

This brings us the following advantages:

1) *Backward and fork compatibility*: the tooling by default will, ignore
`implements` and just rely on function implementation available in the graph,
thus ensuring that it “just works”. Further, if users have their proprietary
implementations they can still ship models that would transparently work in open
source tensorflow.

2) *Easy detection*: Tools can easily detect the functions attribute and
substitute with its own more efficient implementation if available.

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
that its computation description language (today: graph def, tomorrow: MLIR)
will be fully be separated from the underlying tensorflow kernel
implementations. This kind of standard attributes allow for more automatic
conversions should the need arise.

7) Does not change op developer workflow and can be introduced incrementally to
existing code.

### Behind the scenes changes

When a new tf.function is created, behind the scenes tensorflow creates up-to 3
functions that are stored in the graph. These functions are: 1) `Inference`:
x->y, a straightforward function that given composite op implements. 2) `Forward
function`: same as inference but includes extra outputs that are needed for
backprop 3) `Backprop function`: takes as an input the dL/dY, and all the side
outputs of `Forward function` and produces dL/dx

In the current implementation the annotations will only be added to `inference`
function, but not to `Forward` or `Backprop` functions.

### More on forward compatibility

Suppose later, the maintainer of the op decides that certain composite op
deserves its own atomic op due to its usefulness. If we have standardized
attributes like the one above, the TensorFlow compiler can check if op with the
name that this function “implements” is available and substitute it at runtime.
The significant advantage is that even old serialized models will be able to
take advantage of improved implementations. As the op gains more adoption and
its support becomes more widespread the previous composite definition can be
deprecated or retired. However such an approach allows both implementations to
coexist for unlimited period of time and the same graph can be both: efficient
on the platforms that support it, and “not-broken” on the platforms that don’t.

### Alternatives considered

#### Alternative 0: Do nothing

Everyone just continues with ad-hoc implementations.

##### Pros:

*   No need for RFC or new public API

##### Cons:

  Downstream tools are stuck between
  
    a) trying to implement ops, even if the
    efficiency is not yet a concern. In which case this just redo's
    implementation in the downstream tooling using its own language. Or 
    
    b) trying to extract certain graph patterns that can be optimizing and
    following the moving target. Once such pattern is extracted there is a
    pressure on op-maintainers to freeze his implementation to avoid breaking
    the patterns used by the downstream tools. Or 
    
    c) Trying to roll out their  own extension such as `OpHint` for tflite.

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
ecosystem, effectively forcing large users to be stuck at old versions of
tensorflow.

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


## Appendix: Future support for optimizing gradient functions

This proposal is concerned with optimiziging inference pass of composite ops . The
motivation today is that the downstream tooling today rarely if ever deals with gradients, and when
it does, it can rely on the provided implementation. However eventually
this is likely to change, and we would like to keep the door open for extending
this composite op framework to support optimization of both forward
and backward passes. In this appendix we provide several options of how this
could be potentially supported in the future, for the reference purposes.

### Background: why is it hard

Suppose we have implemented function `my_function(x) =
very(complex(implementation(x)))`. Now, if some downstream library would like to
support optimized implementation of `my_function`, all it needs to do is to
replace the tensorflow implementation with its own implementation. However, if
at any point we need to compute gradient of `my_function(x)`, then to avoid
recomputation, the default implementation of gradient would need all
intermediate values produced by tensorflow implementation. In this case this
would be values of `implementation(x)`, `complex(implementation(x))`, etc.

This is problematic for two reasons:

1) Downstream have dependence on tensorflow's implementation of the composite
ops, and for instance can't just choose arbitrary implementation 2) If
tensorflow implementation changes so needs downstream's.

In this appendix we outline two possible paths forward to resolve these issues.

### Option 1: Stabilize Gradient Signature

Option 1 basically revolves about allowing the composite_op to provide
explicit list of side outputs that could possibly be required for the efficient
gradient computation. The `tf.function` machinery would then validate that the
signature actually matches the implementation and respect the order
provided in the signature. The downstream tooling  would need to compute the side-output
when providing its implementation.

This option means that we would not be able to significantly change the
implementation in the future. For instance, if it is discovered that
`my_function(x)` can be computed as `simpler(transformation(x))` we won't be
able to change the implementation it without changing the signature.

Note, for a non-trivial subset of fast functions this gradient signature could
be empty. In fact, *any* gradient could be computed without any side outputs, by
recomputing the function internally. Thus, side outputs only become important
when the function involve non-trivial compute.

Thus, this option might be acceptable in the following  cases, where there are no
standard outputs, or if side-outputs are unlikely to ever change.

### Option 2: Allow dynamic substitution of side outputs. 

Consider the `inference` and `forward` functions. The former has signature
`x->y`, the latter is `x->y, s`. Either y or s can be a list of multiple tensors. 
Comparing the signature, tooling can select the split point to separate inference_part 
and side output for backward part

> Assumption: side_outputs of forward_XXX are only ever used as inputs to
> inference_backward_XXX, if they are used for any other purposes, then the
> downstream can’t replace the underneath XXX until those uses are eliminated.
> (Possibly via graph transformations). This makes sense because the graph
> depends on implementation detail of tf.function implementation, thus
> tf.function shouldn’t be optimized away.```

Suppose the tooling have an efficient implementation of the gradient that needs
its own side outputs let them be t1 ... t_k. Then it can replace all three
functions with the re-implementations with the following signatures.

Function  | Original signature      | New Signature
--------- | :---------------------: | -----------------------:
Inference | x -> y                  | x -> y
Forward   | x->y, s1,..., sk        | x-> y, t1, ..., tl
Backward  | dl/dy, s1,..., sk -> dx | dl/dy, t1, ..., tk -> dx

Important: Implementing this today would be fairly complex in case of a nested
functions, because the gradient of the outer function, requires side-outputs of
all inner functions, thus not-only the signature of the function that we change
changes, but also the signature of all functions that _call_ this function. Thus
the tooling will need to do a non-trivial whole-graph transformation to update
the signatures of **all** functions that call the optimized function. However,
it desn't seem to be insurmountable and possibly fairly straightforward with
MLIR.

Another,  cleaner option would be to wrap all the side output into a single blob which
contains multiple tensors which the implementation can then replace with its
own. There are no such structure in tensorflow today, but it might be in the
future. We should use this, if it becomes available. In this case this would
essentially make a forward and backward to have stable signature.

## Questions and Answers

This list reflects questions raised and the consensus discussed during
Tensorflow Review process.

1.  We should provide namespacing for the operation names early on to avoid
    future collisions with function names. One option is to adopt java-style
    "org.opname", basically using the developing organization to disambiguate. Another alternative is to use semantic namespaces e.g. `glinalg`. Yet
    third, is to use org.semantic.opname.

    Note: since many of these functions will live in user codebase we obviously
    can't and shouldn't police them, however having a recommended style guide
    will simplify things later on.

    Should there be a `graduation` process? Where ops eventually move to a
    higher tier. E.g. dedicated top-tier op like tensorflow.opname? If so, we
    might also consider having another attribute 'aliases' to allow history of
    names for backward compatibility.

> The consensus appears to be to follow org.semantic.opname

1.  If we go org/name, do we need centralized registry of accepted "org" names?

> No

1.  Do we need `reference` field? The idea is that points to the source of
    authoritative ground truth implementation (likely not tensorflow based),
    which this op is faithfully trying to reproduce? Should this reference be
    structured? (For example it could point to existing scipy/numpy/pillow
    library, or it could be pointing to a paper?). Should we make this optional
    or get rid of it altogether and delegate this to documentation?

> The consensus seems that this field is not needed.

1.  Name collisions - crash, no crash, etc.

> The consensus appears to be no-crash is the most natural outcome despite
> initial misgivings. The argument that tilted in this direction is that the
> definition that will be invoked is actually well defined by python code
> semantic, (e.g. user code would call say python_lib.something.my_conv_op,
> which declares (myai.my_conv_op) and user intent is clear. What the downstream
> tooling will do is going to be up-to downstream tooling as long as it follows
> the contract. If there are two implementations available and different parts
> of the code call different > ones, we might end up with two definitions in the
> same function, but every invocation is still well defined in the graph itself,
> and thus preferrable.

1.  tomhennigan: Aligning the definitions with MLIR

    I wonder if we should consider more metadata here. For example within a
    dialect MLIR op definitions [0] include a name [1], summary, description.
    Maybe we can align with them? Concretely I suggest considering something
    like:

@tf.function( op_def=tf.OpDef( dialect="google", name="mm_low_rank_matrix",
summary="..", description="..", )) def f(..): pass [0]
https://github.com/tensorflow/mlir/blob/master/g3doc/OpDefinitions.md#operation-definition
[1]
https://github.com/tensorflow/mlir/blob/master/g3doc/OpDefinitions.md#operation-name

> No, because MLIR dialects are not well aligned with semantic dialects that we
> are considering. E.g. MLIR dialects are following framework dialect (e.g. TPU,
> or tensorflow, etc...), instead of "this is a group of related ops".

1.  Should this belong to existing `tf.function` or a new alias is preferrable
    e.g. `tf.library_function`

> Use existing tf.function

1.  Should there be a specialization mechanism, where different block
    implementations that are more efficient on different target hardware. (Both
    for tensorflow, and downstream tooling).

> yes, eventually, but seem not critical to get it from get-go.

1.  ycling: Should function_def have dedicated fields for describing these, or
    just using attrs<string, AttrValue> attrs is a good option.

> attrs

1.  joker-eph: what about backward pass? If downstream tooling is looking to
    support back-prop efficiently, there will need to be more hooks to implement
    the gradient functions. In particular, the back-prop pass won't be able to
    use efficient forward implementation, because back-prop requires internal
    tensors (unless there is a matching efficient gradient implementation).

From joker-eph: "Right, replacing a function with the gradient computation seems
to me like a key use-case that we will want to support. Not having a solution
for this makes this proposal much less attractive."

> I think we run out of time on this one, but the proposal is that this is
> probalby will be left unspecified in this iteration. The potential path
> forward is to automatically wrap gradient into its own function that
> downstream tooling can identify and replace with its own implementation. If
> the tooling needs to support both forward and backward path, it seems that to
> benefit from this the tooling would need to provide both implementations (and
> do internal caching) or simply rely on default implementation.

1.  tomhennigan: Performance implication of wrapping too much stuff into
    tf.function: One thing to note is that there is an overhead to calling a
    @tf.function from eager mode (currently ~100us) and I suspect we will want
    to be careful about adding it around lots of TensorFlow's public API without
    careful consideration (e.g. if 100us of overhead would dominate the runtime
    of the composite op).

I did actually try this a while back (wrapping all functions in TF public API
with tf.function) and found it only made a performance improvement for
tf.reduce_logsumexp and tf.einsum (at least for the model I tested with).

> The consensus is that we will eventually make tf.function fast enough that it
> won't be an issue, and given that this is likely to have very gradual roll out
> we will have time to adapt.
