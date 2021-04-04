# API & Artifact Compatibility for TFX::OSS & Libraries

| Status        | Proposed                                                 |
| :------------ | :------------------------------------------------------- |
| **Author(s)** | Zhitao Li (zhitaoli@google.com), Jiayi Zhao              |
:               : (jyzhao@google.com), Zhuo Peng (zhuo@google.com)         :
| **Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com), Zhitao Li |
:               : (zhitaoli@google.com)                                    :
| **Updated**   | 2021-03-21                                               |

[TOC]

## Objective {#objective}

Defining requirements and processes for TFX::OSS and related libraries (see list
below) to achieve necessary API & Artifact compatibility.

### Goals {#goals}

*   Clearly define a stable API surface of TFX::OSS, and lay out a
    structure to determine what falls into the stable public APIs.
*   API stability: TFX::OSS (as pipeline) and all APIs from the
    [Key Dependency Libs](#key-dependency-libraries-in-scope) exposed through
    TFX::OSS should achieve desired backward compatibility (see
    [Key Decisions](#key-decisions)).

#### Key Dependency Libraries In Scope {#key-dependency-libraries-in-scope}

The following libraries are included as key dependencies of TFX::OSS, and
subject to the compatibility scope laid out in this document:

*   [tfx-bsl](https://pypi.org/project/tfx-bsl/)
*   [ml-metadata](https://pypi.org/project/ml-metadata)
*   [tensorflow-metadata](https://pypi.org/project/tensorflow-metadata/)
*   [tensorflow-data-validation](https://pypi.org/project/tensorflow-data-validation/)
*   [tensorflow-model-analysis](https://pypi.org/project/tensorflow-model-analysis/)
*   [tensorflow-transform](https://pypi.org/project/tensorflow-transform/)
*   [ml-pipelines-sdk](https://pypi.org/project/ml-pipelines-sdk)

### Non-Goals {#non-goals}

*   LTS Version: No LTS version is targeted in the current scope of this doc.

### User Benefit {#user-benefit}

Enterprise users of TFX::OSS will benefit from a stable distribution of TFX
after these goals, and being assured that their TFX pipelines will keep working
after an upgrade, within the defined compatibility scope.

## Motivation {#motivation}

Specific Enterprise users who have requested a stable distribution:

*   SAP Concur: quote from a partner : _“I think the engineering management is
    simply waiting for a 1.x production release of TFX for the final adoption.”_

## Design Proposal {#design-proposal}

There are generally two types of compatibilities:

*   _Backward Compatibility_ allows a newer version of software to interoperate
    with older versions of the same or different software.
*   _Forward Compatibility_ allows a version of software to accept input from a
    future version of the software itself, or a different software producer.

Due to the nature of TFX libraries (python libraries which usually produces some
Artifacts), there are also several aspects for each libraries involved:

1.  Artifact Type (aka Metadata): This is the metadata which describes the
    produced Artifact(s) from a TFX lib, e.g. `Statistics`/`Schema` for
    TFDV/TFMA, `TransformGraph` for Transform;
2.  Artifact Payload: This is the actual payload of the Artifact, which is often
    files on external storage;
3.  API: This generally refers to all publicly visible protobuf definitions, and
    public Python symbols, including but not limited to functions, classes,
    variables and decorators (see our definition of public Python APIs below).
    All C++ APIs are out-of-scope for this document.

### Key Decisions {#key-decisions}

The following decisions are provided for easy digestion:

1.  Python APIs shipped by `tfx` and `ml-pipelines-sdk` packages shall be ready
    for 1.0 release and approximately SemVer compliant. Specifically:
    1.  Public Python APIs, Protobufs and RPC interfaces will be backward
        compatible within all 1.x releases, with the exception of:
        1.  Experimental APIs: either with `experimental` in its symbol/path, or
            explicitly called out in PyDoc.
            1.  We propose only use `__init__.py` for public APIs. PyDoc can be
                generated that way.
        2.  Internal APIs: Any other API which is not covered by our public API
            rules listed above are considered internal to the library, even if
            they do not have "internal" or underscore prefix. These APIs should
            not be directly used by users.
        3.  C++ APIs: none of the C++ APIs from any related libraries or
            repository will be considered public or stable. We will call this
            out in the release log and README of each project.
    2.  For Artifact Types (as well as evolution): requirements are being
        collected and will be discussed separately.
    3.  Artifact Payload: Generally required backward compatibility, no
        guarantee on forward compatibility
        1.  Exception on _Model_: TensorFlow SavedModel has its own
            compatibility rules, and other model formats are not guaranteed with
            any compatibility.
2.  Other packages are encouraged to follow above compliance requirements,
    especially if they want to graduate into a 1.0 version. This is eventually
    call of the library owner, with the exception for:
    1.  Public APIs, Protobuf definitions and RPC interfaces which are used or
        exposed by either `tfx` package, or its examples/templates shall follow
        above rules, so tfx can achieve its compliance goal. Some of the
        underlying libraries will go to 1.0 with TFX, but that is not required
        by the scope of this doc. Notable exceptions include but not limited to:
        1.  `struct2tensor` will not go to 1.0 because it is an optional
            dependency for TFX and only used in recent examples, which are
            experimental themselves.

See sections below for a more comprehensive list.

##### LifeSpan of Major Version {#lifespan-of-major-version}

LifeSpan of a major version is the difference (in days) between two major
versions. This is an important factor because all minor versions within the same
major version are required to be backward compatible, and certain cleanups can
only be completed in a new major version.

We propose a minimum **12 months lifespan** for TFX 1.0, which shall be the
minimum for future major versions. When we are close to the end of that, we will
revisit our tech debts and determine if we want to extend it.

### Compatibility with Key External Libraries {#compatibility-with-key-external-libraries}

The following external libraries are considered key dependencies, i.e we
definitely need them for key CUJs, and we don’t see us moving away from them in
the foreseeable future. This section briefly discusses how this affects our
compatibility guarantees.

It is worth noting that among all these key dependencies, only TensorFlow has
Artifact compatibility items, and everything else has only API compatibility.
This is helpful for our effort, because we can do future major version upgrades
(e.g, from Beam 2 to 3, if/when that happens) in a way hidden from our users, at
least for all the first party components which we own.

#### TensorFlow {#tensorflow}

Because TensorFlow Transform (TFT) still relies on internal APIs of TensorFlow
which is not subject to compatibility guarantees, each version of TFX libraries
is only guaranteed to work with one version of TensorFlow.

Meanwhile, there are users who need to use a different version of TensorFlow in
their training code, therefore we anticipate a need to mix TFX components which
link with different TensorFlow versions in the same pipeline. This is possible
in a containerized deployment, because we will support configuring each component
with different container images. In order to process Artifacts produced from
different TensorFlow versions, we will rely on the guarantees provided by
TensorFlow on SavedModel
([source](https://www.tensorflow.org/guide/versions#compatibility_of_savedmodels_graphs_and_checkpoints)),
most importantly:

*   _In compliance with [semver](https://semver.org/), SavedModels written with
    one version of TensorFlow can be loaded and evaluated with a later version
    of TensorFlow with the same major release._
*   _"Any SavedModel supported in TensorFlow major version N can be loaded and
    executed with TensorFlow major version N+1"_

Because we do not guarantee forward compatibility on our Artifacts payload, this
means that if a component (i.e, _Trainer_) needs to use a different version of
TensorFlow, then all _ancestor_ components which produced SavedModel (i.e,
`Transform`) must use the same or older version of TFX.

##### TensorFlow Enterprise 1.15 {#tensorflow-enterprise-1-15}

The LTS version of TensorFlow Enterprise version poses a special challenge,
because the 1.15 distribution is expected to be supported for three years
starting from Oct 30, 2019. We propose to make the TFX 1.0 minor the only
supported version for TensorFlow 1.15, and test this combination to the maximum
extent possible. Any future versions of TFX 1.x will not be supported with TFE
1.15.

#### Apache Beam {#apache-beam}

Apache Beam mostly follows semantic versioning: breakages are rare and we have
only seen one occurrence related to type hints once in the entirety of 2020.
Therefore we propose the following for all TFX 1.x versions:

1.  Each version of TFX is guaranteed to be compatible with latest version of
    `apache-beam` at release time;
2.  In addition, `tfx`'s `install_requires` will allow future minor versions of
    `apache-beam` in the same major version, i.e, `apache-beam>=2.27,<3`.
3.  When a breaking change to above from `apache-beam` is detected, TFX team
    will create patch versions to either (a) cherry-pick a fix, or (b) lower the
    upper bound;
    1.  For example, say `tfx` 1.1.0 is released with `apache-beam>=2.26,<3`,
        and `tfx` 1.2.0 is released with `apache-beam>=2.27,<3` (version numbers
        are examples rather than real). If `apache-beam` 2.30 introduces a
        breaking change deemed impossible to fix, `tfx` 1.1.1 and 1.2.1 will be
        released with `apache-beam>=2.27,<2.30`. Please also note that we commit
        to test tfx with `apache-beam` RC so such breakages should be
        exceedingly rare in the future.

#### PyArrow {#pyarrow}

TFX’s dependency on PyArrow comes from
[TFXIO](https://github.com/tensorflow/tfx-bsl/blob/19df0b53e004404ae1971d4140a3242f3490f2d4/tfx_bsl/tfxio/tfxio.py#L35).
Specific public APIs exposed through are:

*   pyarrow.RecordBatch (thus pyarrow.Array, and more)
*   pyarrow.Schema
    *   The stability of contents of the schema (i.e. how data is represented in
        RecordBatches) is guaranteed by TFX-BSL, please see the “TFX-BSL”
        section.

For the entire TFX 1.x, at least PyArrow 1.x and 2.x will be supported.

### TFX & ml-pipelines-sdk {#tfx-&-ml-pipelines-sdk}

This section discusses all parts of the `tfx` package, including the minimal SDK
package _ml-pipelines-sdk_, in detail.

#### Artifact Payload {#artifact-payload}

Almost all supported Artifact types have some payload on-disk storage. The
compatibility guarantee is the following:

*   Guaranteed Backward compatibility: within same major version, new release of
    TFX components shall take Artifacts produced by an older version of TFX;
    *   We shall record versions of all Artifacts: unless otherwise documented,
        the version of an Artifact equals to the _&lt;major>.&lt;minor>_ version
        of its producer component. This version will be further used as
        consumption hint for these Artifacts, to correctly handle version
        upgrades.
*   **No Forward compatibility guarantee on payload**: while the system will not
    reject passing Artifacts produced from newer versions of TFX to an old
    version of a component, we do not guarantee this will work. A warning will
    be generated in this case.
*   Resolvers and importers would only resolve Artifacts produced with same or
    older versions of the library, by default. A warning will be issued for
    any artifacts skipped due to newer versions.

#### SDK (aka DSL), Official Components {#sdk-aka-dsl-official-components}

All parts of SDK (aka DSL), IR, interfaces of official components shall comply
with SemVer. The functionalities form the following core CUJs, which include but
are not limited to:

*   Defining or declaring components;
*   Composing them into a pipeline graph;
*   Environment/Deployment specific decoration of nodes;
*   Advanced control semantics (condition/loop).

It’s worth noting again that APIs from underlying packages used in above core
CUJs shall also comply.

#### Custom Component {#custom-component}

There are several ways to define a Custom Component in TFX::OSS:

1.  Importing a user provided python function
    *   this process and all APIs involved will be backward compatible;
2.  Importing a user provided container image
    *   this process and APIs will be backward compatible;
3.  Defining ComponentSpec, ExecutorSpec and subclassing with component class
    *   This is a very complex user interface, and arguably exposes too much
        internal implementation details. Moreover, it’s proven that this is
        quite difficult to do for our users. A possible improvement proposed is
        to combine the component and executor classes, and have the convention
        be that users subclass a base component's class to extend or override
        functionality. Until this is settled, this API will not comply with
        backward compatibility.

#### Customization on ExampleGen {#customization-on-examplegen}

There are several ways to customize ExampleGen:

1.  Providing a PTransform for converting input source to examples by inheriting
    from BaseExampleGenExecutor and overwriting the abstract method.
    *   This is recommended and will be backward compatible.
2.  Use the above custom Component or custom Executor for use cases like join.
    *   This is a very complex user interface, and arguably exposes too much
        internal implementation details (driver:external file detection,
        span/version/split etc). Moreover, it’s proven that this is quite
        difficult to do for our users. We plan to mark this process as
        experimental in 1.0, and not comply with backward compatibility.

#### Custom Executor {#custom-executor}

Currently TFX supports “custom executor”, but essentially this means providing a
different executor to an official TFX component. So far we have found two
concrete use cases:

1.  Google-cloud optimized tuner and trainer;
2.  Customizing ExampleGen component.

Because this process requires users to understand many details of internal
implementations of an official TFX component, we plan to deprecate this support
externally and declare it internal only. We will only support currently known
use cases listed above.

#### Scaffolding Templates {#scaffolding-templates}

The scaffolding template CLI command itself is still **experimental**, therefore
excluded from compatibility compliance. Note that generated pipeline code &
config are considered user code, therefore they can expect the same
compatibility compliance as listed above.

Putting it differently:

*   The structure of code and configs generated is subject to (possibly
    breaking) change between minor versions;
*   Once generated, the produced pipeline code and configs are essentially user
    code which relies on TFX’s public API, thus guaranteed to work with newer
    minor versions within the same major version, as explained in backward
    compatibility rules.
    *   This rule ensures that even if a user upgraded the `tfx` package in
        their environment, the scaffolded pipeline can still function.

#### Examples, Notebooks and Tutorials {#examples-notebooks-and-tutorials}

Examples, Notebooks (Colab) and Tutorials are **excluded** from the
compatibility compliance. These are created mostly for demonstrating how to use
TFX in accordance with best practices, therefore often subject to a particular
version of TFX.

These examples should use the stable public API of `tfx` as much as possible.
Usage of experimental or deprecated APIs in these examples should be explicitly
called out, and the examples should pin their dependency version of `tfx` down
to a narrow range as much as possible in these rare cases.

#### New Orchestrator {#new-orchestrator}

The new orchestrator is in early stages of development and we don’t think it’ll
be production-ready for 1.0 timeline, thus it will enter 1.0 as `experimental`.

#### Portability Layer to Other Runners (aka Orchestrators) {#portability-layer-to-other-runner-orchestrators}

This portability layer in tfx refers to various DagRunners which runs TFX
pipelines on various external orchestrators, the _portable_ module and the
abstraction between CLI and Dag runner. These libraries will comply with
backward compatibility.

Notice that several Runners are experimental by their nature, or slated to be
deprecated. Specifically, we plan to support the following runners with backward
compatibility:

*   LocalDagRunner
*   KubeflowV2DagRunner

The following Dag runners will remain experimental:

*   AirflowDagRunner
    *   Honestly speaking, support of this one is on the fence. Due to lack of
        proper configuration for a distributed backend support, it’s not much
        more powerful than the LocalDagRunner when used out-of-box. However,
        some partner teams managed to connect this to a distributed backend.
        Until that is supported in TFX, we don’t think this one is stable.

The following dag runners are slated to be deprecated and therefore won’t
comply:

*   BeamDagRunner: this will be replaced by the LocaDagRunner
*   KubeflowDagRunner: this dag runner implementation based on KFP v1 API will
    be replaced by the V2 dag runner above.

#### FrontEnd {#frontend}

Similar to Orchestrator, for the upcoming TFX::OSS frontend (first version as
plugin to Tensorboard), no backward/forward compatibility is guaranteed.

#### Command Line Interface (CLI) {#command-line-interface-cli}

Command groups to fulfill key CUJs will be backward compatible, notably:

*   _pipeline_
*   _run_

_The template_ command group will remain experimental.

### ML-Metadata (MLMD) {#ml-metadata-mlmd}

We are proposing that MLMD project enters the same 1.0 readiness and
compatibility as TFX. The rest of this section will discuss various aspects of
MLMD.

#### Client Libraries {#client-libraries}

Several protobuf structs from the `metadata_store.proto` are used and exposed by
TFX to define Artifacts and need to subject to the compliance requirements.

#### Protobuf & RPC Interface {#protobuf-&-rpc-interface}

All protobufs w/o experimental or internal in their names/packages/module shall
be SemVer compliant.

MLMD has a list of gRPC interfaces. These interfaces shall be backward
compatible, unless explicitly called out in protobuf comment.

#### Database {#database}

All distributions of MLMD run on top of a database. Right now, SQLite and MySQL
are available for external users.

All data model schemas for Sqlite and MySQL backends shall be backward
compatible with the MLMD schema migration utilities.

### TFX-BSL & TFXIO {#tfx-bsl-&-tfxio}

Only code in the
_[‘public’](https://github.com/tensorflow/tfx-bsl/tree/master/tfx_bsl/public)_
module in TFX-BSL is considered public to external users.

#### TFXIO Usage {#tfxio-usage}

Usage of TFXIO in TFX components are considered implementation details. As long
as general Artifact payload backward compatibility goal is met, no additional
guarantee is provided to the user.

TFXIO reads data on disk into following formats:

*   PyArrow RecordBatches. The schema of such RecordBatches (for example,
    tf.Example data would become RecordBatches that contain ListArray columns)
    is also considered public and should be stable.
*   Tensors: the tensor presentation should be stable. This is provided by
    [`tf.io.parse_example`](https://www.tensorflow.org/api_docs/python/tf/io/parse_example),
    which is a stable public API in TensorFlow.

Public interfaces of TFXIO (in `tfx_bsl.public`) might be needed for custom
component authoring. These usage will be stable and comply with backward
compatibility guarantees.

### Bulk Inferrer Component {#bulk-inferrer-component}

Bulk inferrer in TFX::OSS is built upon the bulk inference library in tfx-bsl.
The following APIs are exposed by this component thus shall comply with SemVer:

*   <code>bulk_inferrer_pb2.DataSpec</code>
*   <code>bulk_inferrer_pb2.ModelSpec</code>
*   <code>standard_artifacts.InferenceResult</code>: this is serialized
    <code>prediction_log_pb2</code>

### Struct2Tensor {#struct2tensor}

struct2tensor can be used in TFX pipeline, together with TFXIO. This creates the
following Artifacts and components:

*   New Artifact DataView
*   New component DataViewImporter: this takes the struct2tensor query as a user
    code and provides
*   New component DataViewBinder: this binds the produced DataView Artifact
    object to Standardized Inputs in TFXIO and provides advanced inputs.

Support of struct2tensor in TFX pipelines will be `experimental` in 1.0.

### Tensorflow Data Validation (TFDV), Metadata (TFMD) {#tensorflow-data-validation-tfdv-metadata-tfmd}

We consider TFMD mostly a sub-project of TFDV for simplicity.

As discussed above, all APIs which can be used by the OSS components shall
comply with SemVer. Details:

#### Component Interface {#component-interface}

The following components are directly related to TFDV (and TFMD):

1.  StatsGen:_ tfdv.StatsOption is exposed through the StatsGen component
    interface, therefore this class shall be stable.
2.  SchemaGen: its current component interface shall be stable, except that
    _infer_feature_shape_’s default value will be changed to True.
3.  ExampleValidator: there are no configuration options on this component yet
    so its interface is naturally stable.

#### Artifacts {#artifacts}

The following Artifacts are produced by related components, and their payload
shall be backward compatible within 1.x versions, after some clean ups:

1.  ExampleStatistics: this is a serialized
    <code><em>statistics_pb2.DatasetFeatureStatisticsList</em> </code>;
2.  Schema: this is a <code>schema_pb2.Schema</code> object.
3.  ExampleAnomalies: this is a <code>anomalies_pb2.Anomalies</code> object.

There is no direct dependency to TensorFlow APIs in exposed API surfaces of TFDV
and TFMD.

### Tensorflow Transform (TFT) {#tensorflow-transform-tft}

As discussed above, all APIs which can be used by the OSS Transform component
shall comply to the SemVer:

#### Component Interface {#component-interface}

Transform component takes the following configs/code from user:

*   SplitConfigs
*   preprocessing_fn
*   custom_config
*   Schema (see TFDV section for discussion)

All of them shall comply with the SemVer requirements listed above.

#### Artifacts {#artifacts}

Transform component produces following Artifacts, which shall comply with the
guaranteed backward compatibility:

1.  Transform_graph: this is a TensorFlow saved model so general compatibility
    rule for SavedModel additionally applies here.
2.  Transformed_schema: this is currently hidden in transform_graph but might be
    promoted into its own output.
3.  Transformed_examples
    1.  TFExamples are simpler than saved model, so maybe we can offer some
        limited forward compatibility
4.  Analyzer_cache: we propose to fully comply with backward compatibility
    within the same major version
    1.  When cache is invalid, TFT will discard it and analyze from scratch.
5.  (Future work) Pre and post transform statistics

#### Discussion on compatibility with TensorFlow {#discussion-on-compatibility-with-tensorflow}

In Addition to [Compatibility with Tensorflow](#tensorflow):

*   How long do we need to support 1.15?
    *   Current plan: make `tfx==1.0` version the only minor in 1.x to support
        `tensorflow==1.15`, and allow TFX dev to move forward and not think
        about TensorFlow 1 compatibility after that.

#### Notable projects worth watching {#notable-projects-worth-watching}

All projects here already take backward compatibility already in consideration
during design:

*   native tf.func support
*   Sparse tensor support.
*   Keras Preprocessing Layer (KPL) integration: this might affect the Trainer's
    signature exporting part.

### Tensorflow Model Analysis (TFMA) {#tensorflow-model-analysis-tfma}

TFMA is generally used in the _Evaluator_ component in TFX. _Model Validator_ is
already **deprecated** and shall be **removed** in `tfx` 1.0 codebase.

Note that the library of `tensorflow-model-analysis` may not go to 1.0 at the
same time of `tfx` goes 1.0, but rather some time after.

#### Component Interface {#component-interface}

APIs referred by Evaluator component which comply:

*   FeatureSlicingSpec (from evaluator.proto): to be deprecated in 1.0
*   tfma.EvalConfig
*   tfma.EvalSharedModel
*   tfma.slicer.SingleSliceSpec: to be deprecated in 1.0
*   tfma.extractors.Extractor

Possible Exceptions, which are not part of core CUJ yet:

*   `tfma.post_export_metrics`

#### Artifacts {#artifacts}

Evaluator interacts with following Artifacts which comply with above
[rules](#artifact-payload) (backward compatible, optional forward compatible):

Inputs:

*   Examples
*   Model: saved model for TF, so this should be subject to same SavedModel
    rules as listed above.
*   Schema
*   Future Input: statistics (from TFDV)

Outputs:

*   ModelEvaluation
*   ModelBlessing

#### Projects worth watching: {#projects-worth-watching}

*   Batched inputs: likely a breaking change to extractor authors, and also
    consider order of extractors.

#### Discussion on compatibility with TensorFlow {#discussion-on-compatibility-with-tensorflow}

TFMA does not use internal APIs of TensorFlow, therefore its compatibility with
TensorFlow is only on SavedModel. See
[Compatibility with Tensorflow](#tensorflow) for details.
