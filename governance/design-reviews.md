# tf-design-reviews criteria

## Overview

The TensorFlow team has run internal and public design reviews for a while
now. This document tries to capture what type of questions get asked and
concerns get addressed in TF design reviews. It is intended to be used by design
authors as a way of spot checking whether a design review will be useful for
them and by design sponsors as a way of making sure a design proposal clears the
bar for review (ideally every topic in this document should be addressed by the
design proposal for it to be considered).

The main goal of tf-design-reviews is to socialize big changes to TF, document
them, and ensure all stakeholders get a chance to comment on planned
improvements before they’re implemented. Any time a change is made to TF that
will affect multiple aspects of its design or user interface, we should solicit
a design review. TF design reviews themselves are not binding: final approval
rests with whoever has the authority to approve the required code changes, and
the design review is a tool to get consensus and feedback on big changes before
actual approval.

By default TF design reviews should go through the open RFC process in the
tensorflow/community repository, but we will on rare occasions accept design
reviews of google-internal TF-related infrastructure which should be kept
private due to reasons beyond our control.

## General considerations

Every item in this section should be addressed by a TF design review. We do not
require a solution prior to review but we do want to see that the review author
has considered these issues. It is the design sponsor’s job to ensure that
review documents have thought through these issues.

### Performance
Performance is the core reason why most end users use TensorFlow at all; hand
writing code with the same level of performance is prohibitively expensive, and
any other similarly-performing or better-performing solution can also be
integrated in the ecosystem in principle. In that vein, all new designs to TF
should carefully consider their performance implications.

Performance in TF is multi-faceted: we need to worry about scaling from very
small devices (including microcontrollers) to very large devices (beyond TPU
pods); we need to worry about interactive usage (so the cost of making small
changes should be small) and about batch usage (where it’s ok to sacrifice some
startup time to improve steady-state performance); we care both about throughput
(maximizing accelerator utilization saves a lot of money) and latency (as TF is
used in all parts of Google’s software stack); we also care about performance on
many types of hardware.

Can a particular design proposal be implemented efficiently? Does it impose any
inherent limits on the performance in any of the scenarios above? How will it
interact with our other tools for performance (grappler, XLA, eigen, tf.data,
etc)?

### Scope

Does this proposal even belong in TF? As TF itself grows, it’s becoming
substantially more expensive to develop software inside TF itself than as a
separate TF-using project. In this light we need to evaluate whether it’s at all
possible to release a broadly new API or library as its own separate project in
the TF ecosystem.

Even separate projects in the TF ecosystem can benefit from TF’s devrel, blog,
twitter, etc for promotion. It might be possible to share resources dedicated to
CI or github triage, or share infrastructure around syncing to/from google3.

Ideally the only things being added to core TF at this point are things which,
if they are not in core TF, they dramatically limit the usefulness of core TF
itself. General protocols and APIs which different libraries in the TF ecosystem
can implement / accept are good examples of things which undoubtedly belong in
core TF. Ops and kernels used to need to be in core TF, but this is no longer
the case as other projects have sustainable releases of their own binary blobs
and the TF team is working to make it cheaper to release ops and kernels outside
core TF.

Note that we also encourage using the TF design review slot for reviewing
proposals which despite not living inside core TF are expected to be a part of
the broader TF ecosystem.

### Programmability / flexibility

TensorFlow is fundamentally a library to be programmed, and not a collection of
packaged black-box solutions. While it’s cheaper for any individual problem to
solve it with a simple one-line push-button packaged solution this tends to work
poorly in the long run, and lead to usability cliffs and undue API pressures.

For example, let’s think about what would happen if instead of providing tools
to build neural network layers, TF only provided a function that built an entire
network for you. At first we could have very impressively short code examples
(“switch from inception to resnet50 in one line of code!”), but over time users
whose use cases are not exactly covered by this API would either have to
reimplement substantial parts of it themselves or would (most likely) file bugs
asking for small extensions to the API (“can we make resnet52? resnet36?”). Over
time, these bottleneck APIs develop large parameter lists of mutually exclusive
parameters which amount to a poorly defined configuration language for how to
use them.

A key consideration when evaluating a TF design proposal is what would happen
for use cases that are slightly different from the use cases covered in the
proposal itself. The goal is not that the proposal should cover everything but,
rather, that it should be possible to easily reimplement parts of the proposal
using lower level APIs already in TF. If this is not the case then instead of
first implementing the end-to-end solution we need to discuss what low-level
APIs TF should have in such that this proposal could be easily implemented, and
only then reevaluate this proposal.

We also worry about proposals which are too device-specific (be it TPU-specific
or GPU-specific or CPU-specific). While many such things seem reasonable when
first proposed, they break down over time as the set of users for different
devices overlaps quite a bit.

### Integration

As TF has grown, it has sprouted an ecosystem of tools and libraries both
internal and external to TF. New entries to this ecosystem should, as much as
possible, coexist and peacefully cooperate with other entities in the TF
ecosystem. Failing that, new entries should cleanly replace existing
ones. Awkwardly coexisting is not an option we recommend.

The ecosystem includes both things currently inside TF such as Estimator, Keras,
tf.data, tf.distribute, tf.tpu, XLA, or tf.saved_model as well as things
developed outside TF, such as TF probability, vizier, TF serving, MLIR, TFRT,
Sonnet, among others. If existing integration points do not suffice, we should
consider developing new integration points (i.e. how the Sonnet team developed
tf.Module to integrate sonnet, which lives outside TF, with tf.train.Checkpoint,
tf.keras, tf.function, or tf.saved_model).

It is also important that new designs don’t break existing abstractions which TF
supports, such as eager execution, functions, control flow, gradients, or
automatic vectorization. In general, libraries which use simpler TF primitives
(like tf.data) are easier to integrate into the ecosystem than libraries which
try to rewrite TF programs (like TF transform v1). Similarly, we should prefer
proposals which rely on explicit APIs to accomplish things over proposals which
want to do post-hoc graph rewriting (or make converters, or exporters) as those
tend to integrate poorly with each other and tend to be hard to directly
program.

### Maintenance

As many proposals for TF improvements come from outside TF or from outside the
subteams in TF which currently maintain related functionality, TF design
proposals should be upfront about the maintenance story for new functionality
and code.

It is perfectly fine (and common) to punt maintenance on the TF team, but we
should — ahead of the design review — figure out who specifically in the TF team
is signing up to maintain this specific design.

### User groups

While TensorFlow cannot be everything for everyone we do try to cover a broad
swath of machine learning use cases, spanning the spectrum from research to
production, from small to large devices, and from commercial to educational
uses.

It is important for every proposal to TF to talk about which segments of our
user community’s needs are being addressed and for which ones this is expected
to be irrelevant. Specifically, consider stereotypical pure researcher in ML,
researcher applying ML to other fields, students learning ML, industry
professionals applying ML with little to no understanding, industry applied ML
developers, mobile developers, and others.

## Specific considerations

Some particular subsystems of TF have their own considerations which are often
relevant for TF design reviews. It is up to individual designs’ sponsors whether
any of these topics needs to be addressed in the document before review.

### Eager/graph mode

In TF1.x many libraries implicitly or explicitly assume graph-based
execution. As TF 2.0 has been released, eager execution is on by default. This
means that all new TF APIs should be usable from eager execution or from graphs,
and new design proposals should be implemented so they work with both.

In practice this means we cannot rely on per-graph global state, reference
dtypes, and graph pruning to ensure program correctness. Similarly it was
possible in some cases in TF1.x to treat a Tensor as a Promise. In TF2, however,
a Tensor is an already-computed value, and if you need a promise use instead a
function which can compute a tensor on-demand.

### Keras 

Keras has a special status existing both inside and outside TF. As such, changes
to Keras need to consider the impact on the entire Keras community. New APIs to
be added to Keras can comfortably live in tf-addons. Changes to core Keras APIs
need a review owners or sponsor from the Keras team before a TF-wide review.

Further, changes outside the scope of Keras should address how the change will
interact with Keras users, if at all. For example, if a new CompositeTensor is
proposed, will it be a plausible input to Keras layers? If so, how will support
be implemented?

### tf.data

tf.data is TensorFlow recommended API for data (pre)processing and any designs
that pertain to handling data should consider how they relate to tf.data. New
designs pertaining to handling data should provide useful functionality on top
of tf.data (such as the TFDS project or a library of common transformations for
tf.data.Dataset.map) as opposed to alternatives to tf.data.

In addition, new CompositeTensor subclasses should strongly consider
implementing the optional BatchableTypeSpec interface which is needed for
tf.data to be able to batch and unbatch instances of the subclass.

### SavedModel

SavedModel changes in particular need to be both forward and backwards
compatible, as SavedModel files will be written by and read from different TF
versions entirely. In general, removing things from the format is not OK but
adding things is possible if new additions are not required to correctly read
the model from older binaries.

### “Impossible” designs

There are many things which are possible to make work for specific cases but
impossible to make work in general. We should avoid proposing changes to TF that
look like they work in general but in practice each new use case needs to be
covered by manual work from the TF team.

### Distribution Strategies

tf.distribute is the recommended API for distributing computation over GPUs,
TPUs and multiple machines. It is important to consider the implications of a
new design wrt how it would work in a distributed setting. There may be explicit
changes required to ensure the new functionality works seamlessly with / without
tf.distribute.
