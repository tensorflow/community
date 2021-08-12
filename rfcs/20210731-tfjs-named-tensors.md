# A "Named-Dimensions" TypeScript Library for TensorFlowJS

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [395](https://github.com/tensorflow/community/pull/395) |
| **Author(s)** | Lucas Dixon (ldixon@google.com) |
| **Sponsor**   | Ping Yu (piyu@google.com)       |
| **Updated**   | 2021-08-12                      |

## Objective

Support the naming of dimensions in TensorFlow JS to provide an abstraction that
makes it easier for developers and researchers to write and read ML algorithms
(and support some automated optimisation):

* Make it harder to make subtle mistakes (have type-checking get intelligible
  type-error messages).
* Leveraging TypeScript IDE support to get better auto-completion.
* Make it easier to support writing an algorithm and then generalise it to
  support batches.
* Have code more closely resemble the pseudocode and algorithms in papers.
* Allow machine optimization of the ordering of indexes, transpositions and
  permutations.

## Motivation

ML programming is currently much like assembly programming: you have to remember
what each index in a tensor means, and refer to dimensions by index numbers, and
when they change due to the operations you perform, you have to remember the new
index position. Higher level abstractions could make this better and various
ideas have been proposed around naming the dimensions in a Tensor e.g.
http://nlp.seas.harvard.edu/NamedTensor

The impact is likely largest for people who want to write ML algorithms, and for
people using TFJS for education, e.g. to illustrate and visualize ML algorithms.
However, it may also improve the experience for others using TFJS by providing
better tool support, better auto-completion, and improved  optimization for
evaluation.

Here's an example TFJS implementation of single attention-head (For reference,
see the [Tensorflow text tutorial on
Transformers](https://www.tensorflow.org/text/tutorials/transformer) which
provides a for walkthrough, explanation, and python code):

```ts
const queryM: = ...
const keyM: = ...
const valueM: = ...
const xs: = ...

// keys.shape == [seqLen, inputRep]
const keys = tf.matMul(xs, keyM.read())

// queries.shape == [seqLen, inputRep]
const queries = tf.matMul(xs, queryM.read())

// attention.shape == [seqLen (query), seqLen (key)]
const attention = tf.softmax(
 tf.div(tf.matMul(queries, keys, false, true), inputRepSizeSqrt)) as tf.Tensor2D;

// values.shape == [seqLen, valueRepSize]
const values = tf.matMul(xs, valueM.read())

// attendedValues.shape == [seqLen (query), valueRepSize]
const attendedValues = tf.matMul(attention, values);
```

The comments tell you which tensor index position corresponds to which dimension, and the author has to keep track of them and add appropriate comments to understand the code. It's hard to see the errors, especially without comments. For example, the following is very similar, will not typically result in a runtime error, but will give the wrong result:

```ts
const queryM: = ...
const keyM: = ...
const valueM: = ...
const xs: = ...

const keys = tf.matMul(xs, keyM.read())
const queries = tf.matMul(xs, queryM.read())
const attention = tf.softmax(
 tf.div(tf.matMul(queries, keys, true, false), inputRepSizeSqrt)) as tf.Tensor2D;
const values = tf.matMul(xs, valueM.read(), true, false)
const attendedValues = tf.matMul(attention, values, false, true);
```

The proposed Named-Dimension library allows the same thing to be written as:

```ts
const queryM: GTensor<'inputRep', 'kqRep'> = ...
const keyM: GTensor<'inputRep', 'kqRep'> = ...
const valueM: GTensor<'inputRep', 'kqRep'> = ...
const xs: GTensor<'inputRep', 'seqLen'> = ...

const inputKeys = xs.inputRep.dot(keyM.inputRep).seqLen.rename('keySeqLen');
// Note: that rename returns a GTensor, and typechecking happens at edit time,
// so 'xs.inputRep.dot(keyM.kqRep)' would give a red underline and nice error :)
const inputQueries = xs.inputRep.dot(queryM.inputRep);
const attention = inputKeys.kqRep.dot(inputQueries.kqRep);
const values = xs.inputRep.dot(valueM.inputRep);
const attendedValues = values.seqLen.dot(attention.seqLen).keySeqLen.rename('seqLen');
```

This has the following advantages:
* It's easy to see what dimensions are being multiplied together: this is now
  explicit where it was previously implicit.
* TypeScript editors, like VSCode, support auto-completeion to select only the
  valid dimensions. Referencing an index that does not exist can no longer
  happen. e.g. auto-completion of `xs.` results in `inputRep` and `seqLen`. No
  need to remember which order they have either.
* The optimal ordering of indexes can be computed behind the scenes, so the user
  does not need to worry about transpositions and permutations, or how to make
  it optimal.

## User Benefit

A headline might be: "ML programming in JS just got much easier with Named Dimensions". Programming is easier in two import ways:

1. It's easier to write code (and harder to write subtle bugs). This is because
   there is helpful auto-completion, one doesn't need to remember the
   dimensions, and you can no longer accidentally multiply the wrong dimensions
   together (you can do this only via an explicit rename operation).
2. And It's easier to read code too: one doesn't need to remember and map the
   indexes to what they mean.
3. You don't need to work hard to find the optimal ordering and calling of
   permutation operations. This gets done for you behind the scenes.

This combines many of the benefits of eager and lazy evaluation.

## Design Proposal

The key idea is to introduce a Named-Dimension layer over the TFJS Tensor
object. This maintains a mapping between the indexes and their corresponding
names. We leverage TypeScript's ability to have a type that is a set of string
literals (within union operators `|` ). Using TypeScripts intersection (`&`) and
exclusion (`Exclude`) type operations, we can automatically infer types
appropriately. We can also use the type choice operator (`?`) and introduce some
unique error types to make error messages easy to read and act upon.

A `GTensor` (a "Graph Tensor", inspired by
[OpenGraphs](https://arxiv.org/abs/1011.4114)) is a tensor with a set of named
dimensions parameters (each dimension is an axis in the larger tensor). Each
named dimension parameter is just the literal string type of the name.

Each `GTensor` object has a `dim` property that gives access to the dimensions
as named properties. Each dimension is parameterized by both the `GTensor` it
comes from, as well as the name of the type it is a dimension of. To select
which dimension to perform an operation on, one simply uses the property of that
dimension.

```ts
const g1: GTensor<'inputTokens'|'tokenRep'> = ...;
const g2: GTensor<'tokenRep'|'queryRep'> = ...;
const g3 = g1.dim.tokenRep.dot(g2.dim.tokenRep);
// g3: GTensor<'inputTokens'|'queryRep'>
```

Type-checking ensures that dimension names match. i.e.

```ts
g1.dim.tokenRep.dot(g2.dim.foo)
             // Type Error: "tokenRep" is different from "foo".
```

Dimensions can be renamed also to provide a new dimension object with the
correct name, e.g. if `g2` didn't have an `tokenRep` dimension, but we wanted to
dot product with the `foo` dimension, we could do:

```ts
g1.dim.tokenRep.dot(g2.dim.foo.rename('tokenRep'));
```

By working at this more abstract level, you never need to worry about the axis position, you just reference it by name. Underneath this abstraction, we can now optimise the "layout" of the tensors (the order of the axis) and the various permutation operations.

The vision is that this also provides a higher level abstraction that can be used to efficiently compile to XLA, and thus provide a better high level abstraction for ML programming in TypeScript, with better tool support, making it easier for more people be able to explore and write ML algorithms, and remove a large part of the boring and frustrating challenges of making sure indexes align correctly. A side effect is that this also makes code much more readable (see the Attention Head implementation below).

We can make a `GTensor` using an initializer, or by naming the dimensions in a tf.Tensor (and potentially just providing a array or typed array directly):

```ts
// Making a GTensor with an initializer:
const g1 = gtensor.makeTruncNormal({ inputRep: 2, kqRep: 3 });
// gshape() gives you the dict that describes the dimension's sizes.
expect(g1.gshape()).toEqual({ inputRep: 2, kqRep: 3});

// Making a GTensor from a tensor by naming the dimensions:
const g2 = new gtensor.GTensor(tf.tensor(
      [ [ // 'example' dimension index 0
          [ // 'pos' dimension index 0: contains an array of repSize
            1, // 'repSize' dimension index 0
            2  // repSize index 1
          ],
          [3, 4], // pos index 1
          [5, 6], // pos index 2
        ],
        [ [1, 2], [3, 4], [5, 6], ], // example index 1
        [ [1, 2], [3, 4], [5, 6], ], // example index 2
        [ [1, 2], [3, 4], [5, 6], ], // example index 3
      ]
      ), ['example', 'pos', 'repSize']);
expect(g2.gshape()).toEqual({example: 4, pos: 3, repSize: 2});
```

TypeScript's type-inference for `g1` (e.g. in VSCode, mouse-over g1) is:

```ts
const g1: gtensor.GTensor<"inputRep" | "kqRep">
```

And for `g2` it is:

```ts
const g2: gtensor.GTensor<"example" | "pos" | "repSize">
```

We treat each dimension as a first class object (not just string-indexed members
of a dictionary): one can access the dimensions of a `GTensor` via the property
`dim`, and each dimension of a `GTensor` is a property itself. This allows for
nice auto-completion, in particular, for the above example, `g1.dim.`
auto-completes with the drop down menu with `inputRep` first and `kqRep` second.
Notice that the type was correctly inferred from the object given to the
constructor.

Because referencing by `.dim.` many times becomes tedious, we suggest working
with the named dimensions object directly. A `Dimension` knows the `GTensor` it
is part of and this is reflected in the Dimension's type which has two
parameters: one being the set of dimensions in the `GTensor`, and the other the
selected dimension.

This allows the key part of an attention head to be programmed as follows:

```ts
function attentionHeadFn(input: Dims<'seqLen'|'inputRep'>)
  : Dims<'seqLen'|'valueRep'> {
  const inputKeys = input.inputRep.dot(keyM.inputRep)
                          .seqLen.rename('keySeqLen');
  const inputQueries = input.inputRep.dot(queryM.inputRep);
  const attention = inputKeys.kqRep.dot(inputQueries.kqRep);
  const values = input.inputRep.dot(valueM.inputRep);
  const attendedValues = values.seqLen.dot(attention.seqLen)
                                .keySeqLen.rename('seqLen');
  return attendedValues;
}
```

Where `Dims` is the convenient type for the `dim` object of a `GTensor`; this is
what avoids having to write `.dim.` everywhere, and instead to be able to
directly treat dimensions as properties of an object.

Because batching is a key and often complex part of programming, we note that
this paradigm makes turning a non-batched computation into a batched one
somewhat "beautiful":

```ts
const batchedAttentionHeadFn = gtensor.liftFnOverDim(
  'batchSize', attentionHeadFn);
const batchedAttendedValues = batchedAttentionHeadFn(batchedInput);
```

One simply has to specify the name of the batch dimension.

### Alternatives Considered

#### Dimensionsets

In future work, we think we should generalise "Dimension" to being a
"Subdimensions" that can reference many axis names at the same time.

#### What about including the dimension size in the type?

We did experiment with including the size as an integer in the type, however, we
found that this makes the types much longer, more difficult, and the
type-handling code is far more complicated. Overall, because a name has a fixed
size, the only benefit we say to doing this is that it would catch rename
operations for incompatible names. Such a class of error is actually quite easy
to spot, and typically also caught by any tests on the first execution run.

### Performance Implications

* Do you expect any (speed / memory)? How will you confirm?

  * It's expected that the overall performance of code written using this API
    will be faster, depending on the complexity of the optimization of
    permutation operations. But because it's just a way to organize the
    computation, the final computation, or an equivalently manually optimised
    algorithm will not be different.

    For eager execution code, we expect a very slow down due to the extra
    book-keeping of named dimensions.

* There should be microbenchmarks. Are there?

  * For code run on the GPU there is no impact. For code run on CPU eagerly, we
    do not yet have microbenchmarks, but if this RFC proves to interesting to
    the community we will write some.

* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

  * There are some tests, this is still relatively early stage (a full
    implementation is not complete). But tests and benchmarks will be tracked
    with github issues.

### Dependencies

* Dependencies: does this proposal add any new dependencies to TensorFlow?

  * The proposed library depends on TensorFlow JS only.

### Engineering Impact

* Do you expect changes to binary size / startup time / build time / test times?

  * This is expected to start as a separate library, with no effect on existing
    TF code.

* Who will maintain this code? Is this code in its own buildable unit? Can this
  code be tested in its own? Is visibility suitably restricted to only a small
  API surface for others to use?

  * This can be tested on it's own, and the long term plan would be that it is
    maintained and developed by the TFJS team.

### Platforms and Environments

* Platforms: does this work on all platforms supported by TensorFlow? If not,
  why is that ok? Will it work on embedded/mobile? Does it impact automatic code
  generation or mobile stripping tooling? Will it work with transformation
  tools?

  * This is an extension of TFJS, and while it can work in pure JS, key
    advantages (e.g. auto-completion, and type-checking) will only apply to
    those working in TypeScript.

* Execution environments (Cloud services, accelerator hardware): what impact do
  you expect and how will you confirm?

  * No impact. In time one would imagine a compilation from NamedTensors to XLA.
    This would allow TFJS code to work directly on accelerators.

### Best Practices

* Does this proposal change best practices for some aspect of using/developing
  TensorFlow? How will these changes be communicated/enforced?

  Not in the short term. It might after further experimentation be a better way
  to write TFJS code (being more readable), and may in time become the standard
  for that (in the short term, NamedTensor code would look different to other
  tensorflow code, although it is we believe more readable).


### Tutorials and Examples

* If design changes existing API or creates new ones, the design owner should
  create end-to-end examples (ideally, a tutorial) which reflects how new
  feature will be used. Some things to consider related to the tutorial:
   - The minimum requirements for this are to consider how this would be used in
     a Keras-based workflow, as well as a non-Keras (low-level) workflow. If
     either isn’t applicable, explain why.

   See the [included exampe of implementing Transformers in TFJS]().

   - It should show the usage of the new feature in an end to end example (from
     data reading to serving, if applicable). Many new features have unexpected
     effects in parts far away from the place of change that can be found by
     running through an end-to-end example. TFX
     [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have
     historically been good in identifying such unexpected side-effects and are
     as such one recommended path for testing things end-to-end.

   We've like early feedback before implementing a lot more examples. But happy
   to do more of these if this proposal receives interest.

   - This should be written as if it is documentation of the new feature, i.e.,
     consumable by a user, not a TensorFlow developer.
   - The code does not need to work (since the feature is not implemented yet)
     but the expectation is that the code does work before the feature can be
     merged.

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?

Yes.

* How will this proposal interact with other parts of the TensorFlow Ecosystem?
   - How will it work with TFLite?
   - How will it work with distribution strategies?
   - How will it interact with tf.function?
   - Will this work on GPU/TPU?
   - How will it serialize to a SavedModel?
 For all these there is no change w.r.t. TFJS.

### User Impact

* What are the user-facing changes? How will this feature be rolled out?

  This would start as a new TFJS library. The long term vision is to provide a
  better way to write TFJS code that leverages modern IDEs and produces more
  readable, efficient, and less error prone code. Maybe, longer term, it will
  set a standard for other parts of the TF ecosystem. But we want to start with
  a smaller library and experiment with it. TFJS is the best part of the
  ecosystem for this experiment as we can leverage the flexible and strong
  TypeScript type system and modern IDE support.

## Detailed Design

If this idea seems promising, we'll work on sketching a detailed design in the
future. For now we provide an example interface for GTensor and Dimension, and a prototype implementation has also been written, see [`gtensor.ts`](20210731-tfjs-named-tensors/gtensor.ts) and [`20210731-tfjs-named-tensors/gtensor.spec.ts`](gtensor_spec.ts).

### GTensor Interface

```ts
interface GTensor<G extends DName> {
  dim!: Dims<G>;
  tensor: tf.Tensor;
  dimNames: G[];

  constructor(tensor: tf.Tensor, dimNames: G[]);

  gshape(): { [key in G]: number };

  // Rename a set of dimensions.
  public renaming<ReplacedNames extends G, NewNames extends DName>(
    renaming: { [Key in ReplacedNames]: NewNames }
  ): GTensor<Exclude<G, ReplacedNames>|NewNames>;

  // Rename a single dimension.
  public rename<T1 extends DName, T2 extends DName>(
    fromName: G extends T1 ? T1 : never,
    toName: T2
  ): GTensor<Exclude<G, T1>|T2>

  // Optionally, all class operation on Tensors can be here.
}
```

### Dimension Interface

```ts
// G is the set of all dimension names in the GTensor, while D is the specific
// name of this dimension.
interface Dimension<G extends DName, D extends G> {
  name: D;
  size: number;
  gtensor: GTensor<G>;
  index: number;

  constructor(e: DimensionData<G, D>);

  get dtype(): tf.DataType;

  // Operations on dimensions (basically anything that has an axis parameter)
  dot<D2 extends G2, G2 extends DName>(
    d2: DotCompatibleDimension<G,D,G2,D2>
  ): Dims<Exclude<G | G2, D>>;

  rename<T extends DName>(newName: T): Dims<Exclude<G, D> | T>;

  unstack(): Dims<Exclude<G, D>>[];

  // ...
}
```

## Questions and Discussion Topics

* Should GTensor contain a representation of the 'the model' that was used to create it? This could offer some interesting features...

  * If there was interest in have XLA compilation, then this might provide a
    competitive framework for writing larger scale models in TypeScript. This
    feels like it could be particularly interesting for the NodeJS ML community. GTensors could be placeholders for the models that compute them.

  * If there was a model stored behind the scenes, some global permutation
    optimization would be possible. We have not looked hard at the permutation
    optimization code, pointers or remarks on the difficulty of this and its
    potential impact would be particularly welcome. Right now the toy
    implementation does everything eagerly with `einsum`.

  * What's the relationship between the Layers API and GTensor?

    The Layers API is important for TFJS, so the relationship to `GTensors`
    should be elaborated and explored beyond the obvious: every `GTensor`
    contains a tensor internally, so Layers can be applied to it, and conversely
    the output of any layers operation can be turned into a `GTensor`.

    It is worth noting that the `Lifting` operation on `GTensors` provides a
    simpler and more direct alternative to `batch` functionality of Layers. So
    it may be that `GTensors` essentially replace the `Layers` API, but that
    would take some work to make `GTensor` equivalents for all the functionality
    in the Layers API.

  * Could/should GTensor's be thought of as an extended EINSUM notation? Can
    GTensor provide a DSL for easier to understand EINSUM operations?

* What should we do with rehape operations? One option is that reshaping is an
  operation like renaming, but selects a set of dimensions, and introduces a new
  set of dimensions e.g.

  ```ts
  reshape<GTensor<Names>>(
    inputNames: ChangeNames[],
    newShape: { [newName:NewNames]: number }
    ) -> GTensor<Exclude<Names,ChangeNames>|NewNames>
  ```

  Related to this, we should explore splitting and merging of dimensions. Ideas and proposals for this would be welcome.

* We think the typing code could be simplified with a small amount of support
  for the TypeScript team. We have not explored this but would welcome pointers
  or thoughts about this.

* There is a tradeoff with complexity of types that get checked vs
  understandability of type error messages. Currently type errors are very easy
  to understand. Mis-matched types result in a type-error that says what is
  mis-matches, and gives the missing/extra name. e.g. Using strings for named
  dimensions doesn't actually check the dimension's size, but gives a nice
  compromise between type-checking, readability, and clear type-error messages.
  But it may be possible to get the dimensions included also, but it'll make the
  types less readable. Note: the TFJS team did previous try to include the
  specify the specification of the size of each dimension in the TypeScript
  Tensor type (e.g. Tensor<Rank.4D, Dtype.Float32>), however, it resulted in
  very hard to comprehend type errors, and was removed.

  That attempt did not try to include the name of a dimension. Thus different
  dimensions with the same size, in the old effort, could still be confused
  easily (because remembering the meaning of dimension and which index it
  corresponds to is easy to forget).

  Moreover, in the earlier effort, the type system only had index positions and
  their size to create error messages with, thus having readable error messages
  was very challenging.

* There is a tradeoff between high level specification vs control of exactly
  what a computation gets performed. The proposal here takes away the low level
  control of dimension ordering, and the need to control the memory layout that
  way. The hypothesis is that, like in machine code, having to remember stack
  indexes isn't something people do well, and using names and having that
  automated will be a better compromise.

* Should we just make Tensor always be "named" (aka a GTensor)? There's a lot of
  code that depends on not having names. One option we could consider is that
  Tensor can optionally have names. And if they do, then named stuff "just
  works". This might introduce more complexity where having a separate simpler
  library might be easier to work with.

## Related Reading:

* Harvard NLP PyTorch Named Tensors:
 * Motivation: Tensors Considered Harmful (Harvard NLP online article)
 * Docs: https://pytorch.org/docs/stable/named_tensor.html
 * Code: https://github.com/harvardnlp/namedtensor/

* XArray: pandas inspired numerical library for python.
 * Docs: http://xarray.pydata.org/en/stable/

* Haskell:
 * Dex: https://google-research.github.io/dex-lang/prelude.html
 and https://github.com/google-research/dex-lang
 * Using Dependent Types: https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html

* NamedTensor for better ML notation
 * https://namedtensor.github.io/ and Figure 4 / Algo. 2 of https://arxiv.org/pdf/2006.16668.pdf
