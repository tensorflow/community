# Sunsetting `tf.contrib`

![Tranquil Sunset](https://www.publicdomainpictures.net/pictures/30000/velka/sunset-15.jpg)

| Status        | *Accepted*                                           |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Martin Wicke (wicke@tensorflow.org)                  |
| **Sponsor**   | Edd Wilder-James (ewj@tensorflow.org)                |
| **Updated**   | 2019-04-09                                           |

## Objective

The `tf.contrib` module plays several important roles in the TensorFlow
ecosystem: It has made it easy for members of the community to contribute to
TensorFlow, and have their contributions tested and maintained. It is also used
as a staging ground to test early-stage and experimental features in TensorFlow.

However, as the community has grown, the lack of scalability of the current
approach for maintaining and supporting `tf.contrib` has become apparent. 

This RFC is a proposal to sunset the present `tf.contrib`, and replace its
important functions with more maintainable alternatives. Note that it also
affects some non-contrib code which is not part of the `tensorflow` module.

## Motivation

`tf.contrib` serves the following functions:

* Members of the community can submit code which is then distributed with the
standard TensorFlow package. Their code is reviewed by the TensorFlow team and
tested as part of TensorFlow's tests.
* Additions to TensorFlow can be tested without falling under the
[API stability guarantees](https://www.tensorflow.org/guide/version_compat)
imposed by semver. New features in TensorFlow is often staged in `tf.contrib`
before being "moved to core", where it falls under API stability guarantees.

`tf.contrib` is organized in "projects". Each project's code is located in a
subdirectory of [`tensorflow/contrib`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib).
There are currently 107 of such projects in `tf.contrib`. Each project is
nominally maintained and supported by its owners. However, adherence to and
enforcement of this rule has been weak, and most projects are effectively
maintained by the TensorFlow team. 

This leads to some issues with `tf.contrib`:

* The barrier to contribution is too high. Because all code contributed to
`tf.contrib` becomes part of the TensorFlow distribution, and because the
TensorFlow team effectively assumes maintenance responsibilities for it, it has
to be thoroughly reviewed. Even reviewing code for addition to contrib can take
a long time.
* `tf.contrib` is not actively maintained and the quality of its content is
highly variable. Many of the projects in `tf.contrib` are long deprecated,
though rarely formally. Many more have been abandoned by their original owners,
or have been left unsupported and unmaintained for long periods of time.
* Even where projects in `tf.contrib` are vigilantly maintained and
enthusiastically supported, issues with and PRs for these projects are all mixed
together in the rather unwieldy TensorFlow GitHub issue and PR trackers. No
matter how well supported, the TensorFlow team still carries the burden of issue
triage and PR management for all projects in `tf.contrib`.


## Design Proposal

This RFC argues that the `tf.contrib` has outlived its utility and should be
retired. Each project in contrib has one of three options for its future: move
to core; move to a separate repository; or delete. This decision will be made
with the project owners, a proposal for all projects is below, most of which
have been reviewed with the respective project owners.

### Moving to core

Some projects in `tf.contrib` have incubated successfully, and their content
should be moved to TensorFlow. Examples are `tf.contrib.lite`,
`tf.contrib.eager` or `tf.contrib.tpu`. Such a move will require an RFC, and is
reserved for projects which are mature, useful for a large fraction of users,
and which can be maintained by the TensorFlow team.

For parts of these projects for which the exact API remains unclear, symbols
should be prefixed with `experimental`. We have created an exception in our 
[API stability guarantees](https://www.tensorflow.org/guide/version_compat) 
for all symbols (functions, classes, methods, modules, or arguments) which
begin with `experimental`. We reserve the right to change or remove these at
will, or change their behavior.

### Moving to a separate repository

To provide an alternative avenue for contributed code, one or several new
repositories will be created which are maintained by the community. Some larger
projects, especially those which are likely to be interesting to a distinct
community, are best moved into their own repository, and maintained by a SIG
([which needs to be formed](https://github.com/tensorflow/community/blob/master/governance/SIGS.md)).
A successful example of this is `tf.contrib.probability`, which has moved to
`tensorflow/probability`.

At least one such SIG should be formed which will maintain a "generic"
TensorFlow contrib repository. This can contain community-contributed code
which doesn't belong to a larger project, but which is useful for the TensorFlow
community. Examples of such contributions include specialized and possibly
rarely used ops or layers. The scope of such a repository should be kept
reasonably focused (e.g., only ops, Layers, Metrics, Optimizers, Initializers)
as to allow its maintainers to actually maintain the code.

While the exact operating modalities for this repository will be up to the SIG,
we propose to create the repository `tensorflow/addons`. The TensorFlow team
will provide test infrastructure similar to what is running for TensorFlow
(though potentially with reduced platform coverage, TBD). The code in
`tensorflow/addons` should be written against an installed version of
TensorFlow (i.e., using `import tensorflow as tf`, not the direct file imports
used currently), and using only public APIs.

Importantly, the content of this repository will not be included in the
TensorFlow pip package, but is instead independently distributed. If the SIG
chooses, it can release a separate pip package. The TensorFlow team can assist
in setting up a build and release workflow.

Even with such a repository available, it is important that TensorFlow related
code created by the community is easy to find and use. To this end, we will
enhance the community repository to contain an index of community projects. We
invite all who have created tensorflow-related code to add their project there.
To simplify the process of publishing your code in a usable form, we will also
publish a template repository which shows how to set up a TensorFlow-related
repository including a proper build against TensorFlow, tests, and distribution
via a pip package.

### Deleting

Projects that cannot be maintained elsewhere, or which nobody is willing to
maintain, will be removed from distribution with the release of TensorFlow 2.0.
The TensorFlow 2.0 pip package will not include `tf.contrib`. The code for
`tf.contrib` will be removed from the master branch once 2.0.0-RC0 is released,
but will of course remain available on 1.x release branches.

Note that *a lot* of projects are currently marked for deletion. This is not a
reflection of the viability or quality of these projects, but rather due to the
fact that no owner willing to maintain this project separately has been
identified *yet*. If you see a project that you would like to maintain going
forward, feel free to either initiate the process of forming a SIG to maintain
it in the TensorFlow community umbrella, or simply fork the code. TensorFlow
is free software.

## List of Projects

This section lists the possible fate of all projects currently in `tf.contrib`.
The decisions here are preliminary, not all owners have been contacted (those
marked with a * have not). Named in parentheses are people with a potential
interest in this, but who are not owners.

If you have an interest in one of these projects and are not named as an owner
(or as an interested party), please speak up. I've added some (tentative) move
targets to as of yet nonexistent repositories: `tensorflow/io`,
`tensorflow/addons` (I would prefer a better name for this), and
`tensorflow/scientific` will all require new SIGs to maintain them.

Note that any project that has no other fate specified (including those moving
to a repo tentatively maintained by a SIG, but for which no SIG is actually
formed) will be deleted by default at 2.0.

| Project                 | Owner(s)    | Fate                                 |
|:----------------------- |:----------- |:------------------------------------ |
| tools/graph_transforms  | petewarden  | delete (replaced by grappler)        |
| tools/quantization | petewarden       | delete (part of tflite)              |
| all_reduce         | poxvoculi   | delete (replaced by DistributionStrategy) |
| android            | petewarden       | delete (redundant: tflite/ARM build) |
| autograph          | alexbw           | move to core                         |
| batching           | chrisolston      | delete                               |
| bayesflow          | jvdillon     | delete (moved to tensorflow/probability) |
| benchmark_tools    | saeta            | delete                               |
| bigtable           | saeta            | move to tf.data                      |
| boosted_trees     | nataliaponomareva | delete (moved to core)               |
| checkpoint         | allenlavoie      | move to core                         |
| cloud              | saeta            | merge with bigtable/                 |
| cluster_resolver   | frankchn         | move to core                         |
| cmake              | mrry             | delete (replaced with bazel)         |
| coder   | nickj-google jonycgn sungjinhwang | move to tensorflow/compression |
| compiler           | tatatodd         | move to core                         |
| constrained_optimization | acotter    | move to separate repo                |
| copy_graph         |                  | delete (no owner)                    |
| crf                |                  | merge into tensorflow/probability?   |
| cudnn_rnn          |                  | delete (available in tf.keras.layers)|
| data               | mrry             | delete (moved to core)               |
| decision_trees     | salehay          | delete                               |
| deprecated         |                  | delete (deprecated)                  |
| distribute         | josh11b priyag   | move to core                         |
| distributions      | jvdillon     | delete (moved to tensorflow/probability) |
| eager              | asimshankar      | move to core                         |
| estimator          | ispirmustafa mikecase | move to tensorflow/estimator    |
| factorization      | agarwal-ashish   | delete (rebuild in core pending)?    |
| feature_column     | ispirmustafa     | move to core (experimental)          |
| ffmpeg             | fredbertsch      | delete                               |
| framework          |               | partially move to core, delete the rest |
| fused_conv         |                  | delete                               |
| gan                | joel-shor        | move to separate repo                |
| gdr            |*byronyi (poxvoculi)  |                                      |
| graph_editor       | purpledog        | delete                               |
| grid_rnn           |(qlzh727 ebrevdo) | delete                               |
| hadoop             | yongtang (mrry)  | move to tensorflow/io?               |
| hooks              | ispirmustafa     | delete (already in core)             |
| hvx                | satok16          | delete (redundant with NNAPI)        |
| igfs (#22194)      | dmitrievanthony  | move to tensorflow/io                |
| ignite (#22210)    | dmitrievanthony  | move to tensorflow/io                |
| image              |                  | partial move to tensorflow/addons  |
| input_pipeline     | rohan100jain     | delete                               |
| integrate          | shoyer *mcoram   | move to tensorflow/scientific?       |
| kafka              | yongtang (mrry)  | move to tensorflow/io?               |
| keras              | fchollet         | delete (moved to tf.keras)           |
| kernel_methods    | petrosmol rostami | move to tensorflow/estimator?        |
| kfac               | duckworthd       | delete (moved to tensorflow/kfac)    |
| kinesis            | yongtang (mrry)  | move to tensorflow/io?               |
| labeled_tensor     | shoyer           | delete                               |
| layers             |                  | partial move to tensorflow/addons  |
| learn              | wicke       | delete (replaced by tensorflow/estimator) |
| legacy_seq2seq     | ebrevdo (qlzh727) | delete (replaced by seq2seq)        |
| libsvm             |                  | delete (no owner)                    |
| linalg             | rmlarsen langmore | delete (moved to core)              |
| linear_optimizer   | petrosmol (karmel) | move to tensorflow/estimator       |
| lite             | aselle  petewarden | move to core                         |
| lookup          | ysuematsu (ebrevdo) | move to core                         |
| losses             |                  | partial move to tensorflow/addons   |
| makefile           |  petewarden      | delete (RPI build now uses bazel)    |
| memory_stats       | wujingyue        | delete                               |
| meta_graph_transform |  petewarden    | delete                               |
| metrics            | brainnoise       | delete (replaced with OO metrics)    |
| mixed_precision    | protoget reedwm  | delete                               |
| model_pruning      | suyoggupta       | move to core                         |
| mpi                | (poxvoculi)      |                                      |
| mpi_collectives    | *jthestness (poxvoculi) |                               |
| nccl               | (tobyboyd)       | move essential parts to core         |
| nearest_neighbor   |                  | delete                               |
| nn                 |                  | partial move to tensorflow/contrib?  |
| opt                | *joshburkart apassos | partial move to tensorflow/addons      |
| optimizer_v2       |  josh11b         | merge to core                        |
| periodic_resample  |                  | delete (no owner)                    |
| pi_examples        |  petewarden      | delete (will need new examples)      |
| predictor          | ispirmustafa karmel | delete (replaced by tfhub)        |
| proto              | jsimsa ebrevdo   | move to core                         |
| quantization       |  petewarden      | delete (absorbed into tflite)        |
| quantize           | suharshs         |                                      |
| rate               |  itsmeolivia     |                                      |
| receptive_field    |                  |                                      |
| recurrent          | drpngx zffchen78 | replaced with new RNN API?           |
| reduce_slice_ops   |                  | delete (no owner)                    |
| remote_fused_graph |  satok16         |                                      |
| resampler          | fabioviola kosklain | move to tensorflow/scientific?    |
| rnn            | ebrevdo (scottzhu)   | replace with new RNN API             |
| rpc                | ebrevdo jsimsa   |                                      |
| saved_model        | karmel           | move to core                         |
| seq2seq        | ebrevdo (scottzhu)   | adapt, move to tensorflow/addons                 |
| session_bundle     |                  | delete (replaced by SavedModel)      |
| signal             | rryan            | move to core (replace existing) or tensorflow/scientific? |
| slim               |  sguada          | move to tensorflow/models?           |
| solvers            |  rmlarsen        | move to tensorflow/scientific?       |
| sparsemax          |                  | move to tensorflow/addons          |
| specs              |                  | delete                               |
| staging            |                  | delete (redundant)                   |
| stat_summarizer    |                  | delete (no owner)                    |
| stateless          | (*ebrevdo *girving) | replace our random ops with this? |
| summary            |  nickfelt        | move to core, replacing tf.summary   |
| tensor_forest      | nataliaponomareva yupbank | delete (moving to core)     |
| tensorboard        |  nickfelt  jart  | move to tensorflow/tensorboard       |
| tensorrt           | tobyboyd *samikama *aaroey *jjsjann123| move essential parts to core |
| testing            | ispirmustafa     | move to core, make private           |
| text               |                  | partial move to tensorflow/addons          |
| tfprof             |                  | delete (replaced by tf.profiler)     |
| timeseries         | bananabowl karmel terrytangyuan | move to tensorflow/estimator         |
| tpu                | saeta            | move to core                         |
| training           | ebrevdo sguada joel-shor |                              |
| util               |                  | delete (no owner), or move to tools  |
| verbs              |  (mrry tucker)   | delete (no owner)                    |

For a more detailed and constantly evolving symbol map, please refer to this [document](https://docs.google.com/spreadsheets/d/1FLFJLzg7WNP6JHODX5q8BDgptKafq_slHpnHVbJIteQ/edit?usp=sharing).


## Questions and Discussion Topics

* Who will maintain `tensorflow/addons`, `tensorflow/scientific`, and
`tensorflow/io`? I have an initial list, but a SIG is better with more people to
share the load (and fame!). I am taking names, please email
`community-team@tensorflow.org` directly. Once a critical mass of maintainers
are identified, we will form a SIG and we can discuss what should go it in (and
importantly, what should not).
* What's a better name for `tensorflow/contrib`? I would like to get rid of the
name `contrib` to avoid confusion.
* Which groups of projects currently slated for deletion should live together in
a new repo? Who will maintain them?
