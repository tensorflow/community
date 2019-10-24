# TFX Generic Container-based Component Proposal

Status        | Accepted
:------------ | :-------
**RFC #**     | [146](https://github.com/tensorflow/community/pull/146)
**Author(s)** | Ajay Gopinathan (<a href="mailto:ajaygopinathan@google.com">ajaygopinathan@google.com</a>), Hongye Sun (<a href="mailto:hongyes@google.com">hongyes@google.com</a>), Makoto Uchida (<a href="mailto:muchida@google.com">muchida@google.com</a>)
**Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com)
**Updated**   | 2019-09-04

## Objective

This document proposes a design to enable users to attach an arbitrary
containerized program as a component to a pipeline authored using the TFX DSL,
in a way that inter-operates with other components.

This RFC assumes some clarification on the TFX’s use of artifacts and metadata
as explained in [this section of TFX user guide](https://www.tensorflow.org/tfx/guide#artifacts).

## Motivation

A key value proposition provided by
[Kubeflow Pipelines (KFP)](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
is letting users orchestrate arbitrary containers as part of a Machine Learning
(ML) pipeline. Many users already have custom ML applications written in
languages other than Python (e.g. in R, Java, C++ etc), and the ability to chain
existing containerized application programs with other pipeline steps through a
Python DSL is valuable. As of 2019/09
([tfx 0.14](https://github.com/tensorflow/tfx/blob/0.14.0/RELEASE.md#version-0140),
even though TFX supports `KubeflowDagRunner` as an orchestrator, the
[TFX DSL](https://github.com/tensorflow/tfx/tree/0.14.0/tfx) does not provide
a mechanism to accomplish this.

## User Benefit

This document proposes a way to define a proper component based on any
containerized program as a solution to address this problem. The proposed
**container-based component** is realized by a simple DSL extension for custom
containers to the TFX DSL. It enables users to easily declare input and output
artifacts of a pipeline step implemented as a custom container-based program, as
well as a prescription for how to invoke the container’s entrypoint while
passing in the relevant artifact metadata and execution parameters.

In doing so, we will not only enable custom containerized application programs
to be used in TFX, but also augment KFP-based pipelines with the following
capabilities:

*   **Metadata-centric interface**: The proposed container-based component
    provides a framework to clearly declare input- and output- signatures of the
    container-based component in terms of
    [strongly typed](https://github.com/google/ml-metadata/blob/ba69ae039bd2205ec2d7b982b3bfdda4718bf8df/ml_metadata/proto/metadata_store.proto#L55)
    [artifact metadata](https://github.com/google/ml-metadata/blob/ba69ae039bd2205ec2d7b982b3bfdda4718bf8df/ml_metadata/proto/metadata_store.proto#L29).
    This is a key value proposition of TFX DSL.

    *   As of [kfp 0.1.26](https://github.com/kubeflow/pipelines/tree/0.1.26),
        the
        [KFP DSL](https://github.com/kubeflow/pipelines/tree/0.1.26/sdk/python/kfp/dsl)
        has type system to input- and output- of a component. However, it is in
        a way that it doesn’t attach semantic meaning to the input- and output-
        variables in a way compatible to ML Metadata, and the way how TFX
        uitilizes it to realize its functionalies.

*   **Metadata-driven component execution**: Metadata-centric interface of
    components enables TFX Driver and Publisher logic to be applied
    consistently, thus enabling caching of component execution as well
    component-specific (meta)data-driven execution decisions

    *   *Example*: ModelValidator can validate models trained with *any*
        component, so long as the produced artifact is of the *type* Model.

*   **Inter-Component communication via ML Metadata store**: This enables higher
    order component-specific driver logic that depends on another component’s
    behavior.

    *   *Example*: Pusher can choose to proceed or halt pushing depending on
        output artifact from ModelValidator.

*   **Ability to share and reuse a wrapped component as drop-in replacement with
    another**: As a result of the artifact centric, strongly typed input- and
    output- signatures, it enables robust sharing and drop-in replacement of
    components, so long as signatures (the list of input- and output- artifact
    *types*) are the same, and payload is compatible (see also appendix).

Additionally, the proposed container-based component will enable the following
new features in TFX DSL, which already exist In Kubeflow Pipelines:

*   **Ability to use any containerized application program in a pipeline**: The
    proposed container-based component does not preclude any containerized
    application programs from being used as a pipeline step.

*   **Ability to have fine-grained control on the underlying k8s execution
    environment**: The proposed container-based component preserves the user’s
    ability to control underlying Kubernetes runtime configuration for
    containerized program execution.

## Design Proposal

As stated previously, custom containers may be written in arbitrary languages.
The input interface to containers is restricted to the command-line used to
invoke the application, while the output interface is through file I/O (either
STDOUT, STDOUT, or container-local files).

Since container applications may not have access to TFX libraries, they are not
able to (or don’t even wish to) serialize/de-serialize
[ML Metadata](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md)
representing input- and output- artifacts, which is what defines the interface
to all TFX components[^1]. Hence, this design proposes DSL extensions that allow
users to declare the types of the input and output artifacts, and directly
reference the URIs of these artifacts when invoking their containers, while
retaining the ability to invoke the containerized program in exactly the way it
expects with regards to flags, arguments and environmental variables. The TFX
framework itself will generate the necessary driver-level code so that metadata
is logged and used for caching executions.

[^1]: ML Metadata is also what enables inter-component communication to realize
    artifact driven component-specific behavior (such as ExampleValidator and
    ModelValidator)

In order to make this proposal concrete, let’s consider a motivating example of
creating a simple 2-step pipeline. The first step generates examples, and the
second trains a model based on the previously produced examples. The following
shows some example code for these two steps:

*   `my_example_gen.py`

```python
import argparse
import pandas as pd

def convert_and_save(df, file):
  ...

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # URI location of input CSV file
    parser.add_argument('--input_csv_file', ...)
    # URI location of output CSV file
    parser.add_argument('--output_csv_file', ...)
    # Number of CSV columns in the input file
    parser.add_argument('--num_csv_columns', ...)

    args = parser.parse_args()
    arguments = args.__dict__

    df = pd.read_csv(arguments['input_csv_file'],
                     usecols=range(0, int(arguments['num_csv_columns'])))

    # implementation of business logic to ingest dataset
    convert_and_save(df, arguments['output_csv_file'])

```

*   `my_trainer.py`

```python
import argparse
import pandas as pd
import sklearn.linear_model import Logisticregression

def load_dataset(file):
  ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # URI location of input file with pre-processed examples.
    parser.add_argument('--input_examples', ...)
    # URI location to output the trained model
    parser.add_argument('--output_model', ...)
    # Number of training steps to use.
    parser.add_argument('--train_steps', ...)

    args = parser.parse_args()
    arguments = args.__dict__

    x_train, y_train, x_eval, y_eval = load_datset(arguments['input_examples'])

    model = LogisticRegression()
    model.fit(x_train, y_train, args.train_steps)

    #
    # ... perform grid search, etc, ...
    #

    # Write the trained model.
    joblib.dump(model, argument['output_model'])
```

Given the two (containerized) applications above, our goal is to chain them into
a 2-step pipeline which will automatically track metadata related to:

*   **Artifacts**: These include the input CSV file, the output CSV file with
    training and eval examples, and the final Model file.
*   **Executions**: These include execution steps `MyExampleGen`, with runtime
    execution property `num_csv_columns`, and `MyTrainer`, with runtime
    execution property `train_steps`.

Notably, since `my_example_gen.py` produces *Examples*, we make the pipeline
understand that it is *Examples*, and record it as such in ML Metadata. Doing so
will enable downstream components, such as Trainer, to understand that it is
receiving *Examples*, and also realize higher-level functionality such as
ExampleValidation. Similarly, since `my_trainer.py` produces Model, other parts
of the pipeline should understand it to enable higher-level functionality such
as ModelValidation and Pusher to serving.

Hence, we propose to provide an extension to the DSL that allows users to
declare inputs and outputs in terms of ML Metadata of artifacts during pipeline
construction. The proposed DSL extension would translate the input and output
artifacts at pipeline runtime, abstracting reading and writing artifact metadata
from and to storage on behalf of the wrapped containerized application program.

## Detailed Design

We propose the following syntax for wrapping user’s containers in the DSL by
introducing `ExecutorSpec`. This syntax complements and extends the
[ComponentSpec](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/types/component_spec.py#L73)
design previously implemented in TFX, and generalize `EXECUTOR_CLASS` to
[EXECUTOR_SPEC](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/components/base/base_component.py#L63)
attribute of components. We propose to use the same `ComponentSpec` base class
to describe the custom container-based component’s input and output artifacts,
and its parameters.

*   Base `ExecutorSpec` class for container based component.

```python
class ExecutorSpec(with_metaclass(abc.ABCMeta, object)):
  """Base class that specifies 'what' to execute for the component."""
  pass


class ExecutorClassSpec(ExecutorSpec):
  """Execution spec for a Python class-based Executor."""

  # An in-process Python Executor class, derived from TFX
  # base_executor.Executor class.
  executor_class = ...

  def __init__(self, executor_class):
    ...


class ExecutorContainerSpec(ExecutorSpec):
  """Execution specification of the container-based component."""

  # Container image that has executor application.
  # Assumption is that this container image is separately release-managed,
  # and tagged/versioned accordingly.
  container_image = ...

  # Container entrypoint command line program and args.
  # The Jinja templating mechanism is used for constructing a user-specified
  # command-line invocation based on input and output metadata at runtime.
  # The entry point can be as generic as `/bin/sh -c "..."`, which retains
  # the ability to control inputs and/or exec_props with environment
  # variables.
  command = ...
  args = ...

  # Additional specifications of execution specific to Runner's environment.
  # For example, k8s pod configuration for launching the containerized
  # application would be included here.
  # Note that each Runner also has a way to specify it’s orchestrator-specific
  # configuration, such as KubeflowDagRunnerConfig for KubeflowDagRunner.
  # Details and relationship between platform_config and Runner’s config
  # are subject to change. Detailed design document particularly on
  # this point to follow.
  platform_config = ...

  def __init__(self, container_image, command, args, platform_config):
    ...
```

*   `ExecutorSpec` and `ComponentSpec` for `my_example_gen.py`

```python
# Container-based ExampleGen's execution spec.
my_example_gen_exec_spec = ExecutorContainerSpec(
  # Container image of my_example_gen.py.
  # Assumption is that this container image is separately release-managed,
  # and tagged accordingly. This example demonstrates the ':stable' tag.
  container_image='gcr.io/my_project/my_example_gen:stable',
  command=['python'],
  args=['my_example_gen.py',
          '--input_csv_file', '{{ inputs.raw_input.uri }}',
          '--output_examples', ' {{ outputs.examples.uri }}',
          '--num_csv_columns', ' {{ exec_props.num_csv_columns }}' ],
  platform_config=...
)


# One can choose to define a dev-instance of MyExampleGen container-based
# component, based on a different (newer) version of the container image.
#
# Alternatively, by using continuous integration tools, it is possible to
# dynamically build the docker image, and inject its image SHA id to
# this code via a flag.
my_dev_example_gen_exec_spec = ExecutorContainerSpec(
  container_image='gcr.io/my_project/my_example_gen:dev',
  command=['python'],
  args=['my_example_gen.py',
        '--input_csv_file', '{{ inputs.raw_input.uri }}',
        '--output_examples', ' {{ outputs.examples.uri }}',
        '--num_csv_columns', ' {{ exec_props.num_csv_columns }}' ],
  platform_config=...
)


# Container-based ExampleGen's component spec.
# Notice that this is similar to FileBasedExampleGenSpec,
# but with a different set of PARAMETERS.
class MyContainerBasedExampleGenSpec(ComponentSpec):
  """ComponentSpec to drive my_example_gen.py as a Component."""
  # Input artifacts.
  INPUTS = {
      "raw_input": ChannelParameter(type=standard_artifacts.ExternalArtifact),
  }

  # Output artifacts.
  OUTPUTS = {
      "examples": ChannelParameter(type=standard_artifacts.Examples),
  }

  # Parameters.
  PARAMETERS = {
      "num_csv_columns": ExecutionParameter(type=int),
  }
```

*   `ExecutorSpec` and `ComponentSpec` for `my_trainer.py`

```python
# Container-based trainer's executor spec
my_trainer_exec_spec = ExecutorContainerSpec(
  container_image='gcr.io/my_project/my_trainer:stable',
  command=['python']
  args=['my_trainer.py',
        '--input_examples', '{{ inputs.my_inputs.uri }}',
        '--output_model', '{{ outputs.my_model.uri }}',
        '--train_steps', '{{ exec_props.train_steps }}',]
  # Platform config would specify use of GPU node pool for k8s, for example.
  platform_config = ...
)

# Container-based trainer's component spec.
# Notice that this is quite different from TrainerSpec, because of
# the command line flags that my_trainer.py takes are different from what
# TFX stock trainer takes. Nevertheless, it does produce an instance of
# Model artifacts, which can then be consumed by downstream components.
class MyContainerBasedTrainerSpec(ComponentSpec):
  """ComponentSpec to drive my_trainer.py as a component."""
  # Input artifacts.
  INPUTS = {
      "my_input": ChannelParameter(type_name=standard_artifacts.Examples),
  }

  # Output artifacts.
  OUTPUTS = {
      "my_model": ChannelParameter(type_name=standard.Artifacts.Model),
  }

  # Parameters
  PARAMETERS = {
      # Execution properties.
      "train_steps": ExecutionParameter(type=int),
  }
```

### Component definition based on a generic containerized program

With the introduction of `ExecutorContainerSpec`, the way to define a component
based on a containerized application is no different from any other custom
component. Below are illustrations to define the components from the previous
section in full, and their use in an end-to-end pipeline.

*   Component definitions

```python

class MyContainerBasedExampleGen(BaseComponent):
  """Wraps my_example_gen.py."""
  SPEC_CLASS = MyContainerBasedExampleGenSpec

  EXECUTOR_SPEC = my_example_gen_exec_sepc

  # Optionally, if custom driver behavior is desired, such as checking
  # mtime for file updates, one can define a custom Driver class to control
  # the behavior of my_exmaple_gen.py inside the container.
  DRIVER_CLASS = ...


class MyContainerBasedTrainer(BaseComponent):
 """Wraps my_trainer.py."""
  SPEC_CLASS = MyContainerBasedTrainerSpec

  EXECUTOR_SPEC = my_trainer_exec_spec

```

*   `pipeline.py`

```python

def create_pipeline():
  my_csv_file = dsl_utils.external_input(uri="/path/to/csv_file")

  my_example_gen = MyContainerBasedExampleGen(
      raw_input=my_csv_file, num_csv_column=20)
  my_trainer = MyContainerBasedTrainer(
      my_input=example_gen.outputs.examples, train_steps=200)

  return pipeline.Pipeline(
    pipeline_name = 'my_container_based_pipeline',
    pipeline_root = 'gs://path/to/root',
    components = [my_example_gen, my_trainer],
    ...
  )


# It may be the case that some TfxRunner implementation (the ComponentLauncher
# thereof) does not have the ability to run a container-based component,
# in which case, an Exception is raised at the time when the logical pipeline
# is compiled for execution by the TfxRunner.
# See the next section of this doc about ComponentLauncher.
_ = KubeflowDagRunner().run(create_pipeline())

```

### ComponentLauncher to launch the container-based application

With the introduction of `ExecutorContainerSpec` which does not specify
`executor_class`, the default implementation of `BaseComponentLauncher` may
not be able to execute the container-based component. Furthermore, different
orchestrator (i.e. an instance of `TfxRunner`) may have different ways to launch
the containerized application program.

We propose to extend the `BaseComponentLauncher` to define orchestrator-specific
ways to execute the containerized program. It includes the ways to translate
input artifacts to the complete command line, by filling the [Jinja template](https://jinja.palletsprojects.com/en/2.10.x/)
for `ExecutorContainerSpec.command` and `ExecutorContainerSpec.args`, and to
translate output from the containerized application to keep track of metadata of
it and write back to Metadata storage.

Below is one possible implementation of `BaseComponentLauncher` that implements
a way to launch container-based components with `KubeflowDagRunner`, with
ability to configure low level k8s configurations. This
`KubeflowComponentLauncher` would use the k8s Pod API to launch the container
through underlying Kubeflow Pipelines SDK implementation [ref](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/orchestration/kubeflow/base_component.py#L118).
On top of this, `KubeflowDagRunner` allows to apply [additional k8s APIs](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/orchestration/kubeflow/kubeflow_dag_runner.py#L188),
such as volume mount and secret management to pods.

```python
# Demonstration of a ComponentLauncher that has the ability to launch
# container-based component, in addition to executor-class based component,
# with KubeflowDagRunner.
class KubeflowComponentLauncher(BaseComponentLauncher):
  """Demonstration of a ComponentLauncher specific to KubeflwoRunner."""

  def __init__(self, ..., platform_config=...):
    # platform_config delines any Runner-specific, for example k8s-specific
    # configurations for launching containerized application programs.
    ...

  def _run_driver(self, ...):
    # runs driver, which may be custom to each container-based component
    ...

  def _run_executor(self, ...):
    spec = self._executor_spec
    if isinstance(spec, ExecutorContainerSpec):
      # Launch the container entrypoint with the specified image,
      # by the Runner-specific way to execute the container application.
      # In KubeflowDagRunner's case, it is with Argo on k8s.
      # The platform_config is applied here.
      ...
    else:
      # Launch executor_class as usual.

  def _run_publisher(self, ...):
    # runs publisher. When launching container-based executor, this method
    # is responsible for capturing output from the containerized program,
    # and write back to ML Metadata.
    ....

```

Another example of `_run_executor()` to the above illustration may be to execute
`docker run` locally.

The Runner should implement a suitable subclass of `BaseComponentLauncher`
accordingly. A pipeline may have different `ExecutorSpec`s for different
components. In case the Runner, and corresponding `BaseComponentLauncher`
subclasses, does not have a way to execute a containerized program with
`ExecutorContainerSpec`, a runtime error would be raised. If a Runner's `run()`
method has a compilation step from logical pipeline to orchestrator-specific
representation of the pipeline, such error could be caught at compile time.

### Artifact Properties after Execution is complete

It is worth noting that some (custom) properties of output artifacts can only be
determined after executor completed. For instance, `is_blessed` property of
`ModelBlessing` artifact can only be determined after execution finishes.

When a custom image is used by the proposed `ExecutorContainerSpec`, we need to
capture the output of the component, decode value of these properties and send
them to Publisher so that published artifacts have correct final outcome. This
must be done before we transition the output artifact to Published state, so
immutability of published artifacts is preserved.

There are few choices to realize this.

#### Option 1: Disallow artifacts with such custom properties as output

Simplest option is not to support such properties in output artifact from the
proposed container-based component. It is too limiting, and loses the main value
of it such that arbitrary business logic can be implemented in the
container-based application in a way that controls downstream component’s
behavior via output artifacts.

#### Option 2: Capture container STDOUT

Containerize application may indicate the result of execution to STDOUT. The
proposed container-based component could capture it and implement the logic to
translate into the (custom) property of the output artifact. This is also what
the previous Airflow based operators for TFX were doing, before the work to
combine driver/executor/publisher into a single Airflow operator was complete.
We do not see a reason to not generalize STDOUT to any file I/O interface.

#### Option 3 (preferred): Use Files in shared temp directory to capture output artifacts

This is a generalized version of Option 2, in that to capture output from the
containerized program via file I/O, and have the proposed container-based
component to capture it as properties of output artifacts. File I/O is
consistent with the way how `KubeflowDagRunner` passively logs output artifact
metadata as of `tfx` 0.13, hence natural extension to it.

### Binding between ComponentSpec and ExecutorContainerSpec

(A subclass of) `ComponentSpec` defines input and output artifact specification,
and execution property of the component, but does not define ‘what’ to execute
and ‘how’. (An instance of) `(Container)ExecutorSpec` defines ‘what’ and ‘how’
to execute for this component. (A subclass of) `BaseComponent` defines the
binding of `ComponentSpec` and `ExecutorContainerSpec`, in order to be launched
by (a subclass of) `ComponentLauncher`.

There are few possible design options as to where and how to define those
specifications and their bindings.

#### Option 1 (illustrated above): Complete Separation between ExecutorContainerSpec, ComponentSpec and Component

This is as illustrated as the code snippet in the previous sections.

*   **Pros**:
    *   It achieves clear separation between `ComponentSpec`, which is meant to
        define executor-agnostic specification of a component, from
        specification of execution which may be tied to a particular Runner’s
        implementation, as illustrated in extension to `ComponentLauncher`.
*   **Cons**:
    *   Component specifications are defined separately, and developer needs to
        make sure to keep them consistent.
        *   Using `my_example_gen.py` in the above example, all of the below
            needs to be defined in different places and kept in tightly
            consistent.
            1.  Command line flags to `my_example_gen.py`.
            1.  Jinja template defined in `my_example_gen_exec_spec.command` and
                `my_example_gen_exec_spec.args`.
            1.  `INPUT`, `OUTPUT` and `PARAMETERS` in
                `MyExampleGenComponentSpec`.
            1.  The binding between `my_example_gen_exec_spec` and
                `MyExampleGenComponentSpec`, which is done in
                `MyContainerBasedExampleGen` class.
        *   If any of the above is inconsistent, the containerized
            `my_example_gen.py` won’t be invoked properly, or output artifact is
            not logged to ML Metadata thus not usable by downstream components.
    *   Such consistency check needs to be implemented outside of the component
        class, possible as a part of `Runner` or `ComponentLauncher`.

#### Option 2: ExecutorContainerSpec as a part of (subclass of) ComponentSpec

Define a special subclass of `ComponentSpec`, that specifically holds
`ExecutorContainerSpec` as its member.

```python

class ContainerComponentSpec(ComponentSpec):
  ...
  # An instance of ExecutorContainerSpec
  executor_spec = ...

  def _validate_spec(self):
    ...
    assert(instanceof(self.executor_spec, ExecutorContainerSpec))


class MyContainerBasedExampleGenSpec(ContainerComponentSpec):
  ...
  executor_spec = ExecutorContainerSpec(...)


class MyContainerBasedExampleGen(BaseComponent):
  SPEC_CLASS = MyContainerBasedExampleGenSpec

  EXECUTOR_SPEC = MyContainerBasedExampleGenSpec.executor_spec

```

*   **Pros**:
    *   Component’s `INPUT`, `OUTPUT` and `PARAMETER` definitions are co-located
        with `ExecutorContainerSpec.command` and `ExecutorContainerSpec.args`’s
        Jinja template in one place.
        *   It reduces cognitive load to properly define a container-based
            component.
        *   It also makes it possible to place a static validation between them.
*   **Cons**:
    *   `ContainerComponentSpec` defines not only specification of `INPUT`,
        `OUTPUT` and `PARAMETERS` of the component, but also defines ‘how’ and
        ‘what’ to execute, which violates the original design intention of
        `ComponentSpec`.

#### Option 3: Extend BaseComponent specifically for ContainerBasedComponent

Provide a base class of `ContainerBasedComponent`, which defines all the specs
in one place as nested members. `ComponentLauncher` specific to a `Runner`
defines its behavior for subclasses of `ContainerBasedComponent`.
`ContainerBasedComponent` can be thought of as a convenience wrapper that puts
together `ComponentSpec` and `ExecutorContainerSpec` in one place, and provides
additional validation check on integrity between the two.

```python


# Abstract base class that has extra facility to support ExecutorContainerSpec
class ContainerBasedComponent(BaseComponent):

  EXECUTOR_SPEC = _abstract_property()

  @classmethod
  def dynamic_spec_class(cls, inputs, outputs, parameters):
    class _ComponentSpec(ComponentSpec):
      INPUTS=inputs
      OUTPUTS=outputs
      PARAMETERS=parameters

    return _ComponentSpec

  def _validate_entrypoint(self):
    # Make sure SPEC_CLASS, executor_spec.command and executor_spec.args c
    # are consistent.
    ...

  def __init__(self, ...):
    # It must execute containerized program.
    assert(isinstance(self.executor_spec, ExecutorContainerSpec))

    # SPEC_CLASS and EXECUTOR_SPEC must be consistent.
    self._validate_entrypoint()

    # Instantiate Component with given ComponentSpec and ExecutorSpec
    # and other auxiliary configurations.
    super(ContainerBasedExampleGen, self).__init__(...)


# Implementation of a component based on my_example_gen.py
class MyContainerBasedExampleGen(ContainerBasedComopnent):

  # dynamic_spec_class() is a syntactic sugar to be able to inline
  # SPEC_CLASS definition at declaration of ContainerBasedComponent subclass.
  # In case the same ComponentSpec may be shared with another component but
  # with different EXECUTOR_SPEC (and DRIVER_CLASS, etc), this class should
  # be defined explicitly and shared.
  SPEC_CLASS = ContainerBasedComponent.dynamic_spec_class(
    # Implementation of ComponentSpec specific to MyExampleGen. This is
    # exactly the same as `MyContainerBasedExampleGenSpec` illustrated above.
    INPUTS=...
    OUTPUTS=...
    PARAMETERS=...
  )

  # This is the same as my_example_gen_exec_spec illustrated above.
  EXECUTOR_SPEC = ExecutorContainerSpec(...)

```

*   **Pros**:
    *   All specifications of container-based component is co-located in one
        place, making it possible to perform static validation check for
        consistency between specs there.
    *   `ComponentSpec` remains purely about `INPUTS`, `OUTPUTS` and
        `PARAMETERS` definitions, detached from ‘what’ and ‘how’ to execute the
        compononent.
*   **Cons**:
    *   Nested `ComponentSpec` class style may be cumbersome.
    *   Porting a pipeline to a new runner would involve changing all components
        to derive from a new base class, if `ComponentLauncher` of the new
        runner doesn’t know how to launch `ExecutorContainerSpec`.

#### Option 4 (preferred): Utility to create inline specs and do static validation check hook

This is built on top of Option 1.

This option is similar to option 3 and generalizes it to all executor types. The
same pattern can also be applied to python class executor. Proposal is to:

*   Create a `types.dynamic_spec_class()` method to facilitate to create an
    inline `ComponentSpec`.
*   Define an abstract `validate_component_spec()` method in `ExecutorSpec` base
    class to perform executor specific static validation.

```python


class BaseComponent:
  def __init__(self, spec, ...):
    …
    # Call ExecutorSpec.validate_component_spec to validate the component spec.
    # Subclass of ExecutorSpec should implement the validation hook to validate
    # component spec at compile time.
    self.executor_spec.validate_component_spec(spec)


class ExecutorContainerSpec(ExecutorSpec):
  def validate_component_spec(self, component_spec):
    # Call Jinja parser to validate the entry-points with component_spec data.
    …

# Implementation of a component based on my_example_gen.py
class MyContainerBasedExampleGen(BaseComponent):

  # dynamic_spec_class() is a syntactic sugar to be able to inline
  # SPEC_CLASS definition at declaration of BasedComponent subclass.
  # In case the same ComponentSpec may be shared with another component but
  # with different EXECUTOR_SPEC (and DRIVER_CLASS, etc), this class should
  # be defined explicitly and shared.
  SPEC_CLASS = types.dynamic_spec_class(
    # Implementation of ComponentSpec specific to MyExampleGen. This is
    # exactly the same as `MyContainerBasedExampleGenSpec` illustrated above.
    inputs=...
    outputs=...
    parameters=...
  )

  # This is the same as my_example_gen_exec_spec illustrated above.
  EXECUTOR_SPEC = ExecutorContainerSpec(...)

```

*   **Pros**:
    *   It generalizes to all executor types and keeps current component class
        model unchanged.
    *   All specifications of a component is co-located in one place.
    *   Make it possible to perform executor specific static validation check
        for consistency between specs.
    *   `ComponentSpec` remains purely about `INPUTS`, `OUTPUTS` and
        `PARAMETERS` definitions, detached from ‘what’ and ‘how’ to execute the
        component.
    *   Potentially, `BaseExecutor` can extend this model to support a class
        method `validate_component_spec()` to support user executor static
        validation of any logic.
*   **Cons**
    *   `dynamic_spec_class()` style may be cumbersome.
    *   The dynamic class cannot be shared like static class.

## Appendix

### Pipeline Compilation and Release

The proposed `ExecutorContainerSpec` and any related extension of DSL APIs
will reside in the TFX repository. Pending code completion, we may choose
to place some or all of the new APIs under `experimental` namespace until
we admit it to core APIs.  

If run with `KubeflowDagRunner`, it will be executed by `run()` method to
compile into Argo pipeline spec. As a result, there is no need to have any
additional code to be included inside the user’s container image. Other
orchestrators, such as `AirflowDAGRunner`, may have to have a newer version of
TFX SDK with the new `ExecutorContainerSpec` and implementations of
corresponding `ComponentSpec` subclasses installed in the environment in which
the component is executed. Nevertheless, it is no different than any other TFX
component’s execution in which it needs to have the TFX SDKs for components
installed on the Airflow execution environment.

### Componentize a Python function, as opposed to a container image

Kubeflow Pipelines SDK helps users to define a Python function and convert them
to a container-based application as a part of the pipeline (by the
`kfp.compiler.build_python_component()` API). In order for this to become fully
metadata-aware component as proposed in this document, a gap still remains that
it doesn’t help defining input- and output- of the Python function in terms of
the typed Artifacts to be tracked in ML Metadata.

This proposed container-based component could further help filling the gap to
help declaratively configure `INPUTS`, `OUTPUTS` and `PARAMETERS` for the given
naked Python function and componentize it. Furthermore, there is an opportunity
to create a helper to build an image, configure a `python` command entrypoint
from a naked Python function, and construct command line arguments under the
hood, as a specialized subclass of it. Such helper shall eventually converge to
the other way of implementing a
[custom component](https://github.com/tensorflow/tfx/tree/0.14.0/tfx/examples/custom_components)
for TFX via a custom `Executor` class written in Python, and package it in a
container image to release.

Detailed RFC particularly on this point will follow.

### Component Archetypes

As of `tfx` 0.14, there are
[10 known artifact types](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/types/standard_artifacts.py)
defined and used.

*   ExternalArtifact
*   Examples
*   Schema
*   ExampleStatistics
*   ExampleAnomalies
*   TransformGraph
*   Model
*   ModelEvaluation
*   ModelBlessing
*   PushedModel

Based on the above known artifact types, TFX defines the following
[9 component archetypes](https://github.com/tensorflow/tfx/tree/0.14.0/tfx/components).

| **Component**    | **Inputs**                                    | **Outputs**                            |
| :--------------- | :-------------------------------------------- | :------------------------------------- |
| ExampleGen       | ExternalArtifact (optional)                   | Examples                               |
| StatisticsGen    | Examples                                      | ExampleStatistics                      |
| SchemaGen        | ExampleStatistics                             | Schema                                 |
| ExampleValidator | ExampleStatistics, Schema                     | ExampleAnomalies                       |
| Transform        | Examples (raw), Schema                        | TransformGraph, Examples (transformed) |
| Trainer          | Examples, TransformGraph (optional), Schema   | Model                                  |
| Evaluator        | Examples, Model                               | ModelEvaluation                        |
| ModelValidator   | Examples, Model                               | ModelBlessing                          |
| Pusher           | Model, ModelBlessing                          | PushedModel                            |

The proposed generic container-based component will enable scenarios where, so
long as the wrapped component adheres to one of the above the input- and output-
archetypes, it enables drop-in replacement, while retaining interactions with
the rest of the components in the pipeline, and also the pipeline not having to
know the actual business logic inside the container application.

### Specification of Artifacts

As of tfx 0.14, the schema (list of properties) of metadata for each artifact
type is defined implicitly when it is created and used
([example](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/components/transform/component.py#L118)).

In order for the proposed generic component-based component to utilize artifacts
and its metadata in a standardized way, such metadata schema definition needs to
be made explicit, possibly as a Python class (In TFX 0.14.0, the base `Artifact`
class defines the common set of [properties](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/types/artifact.py#L103),
with option for each sub-classed type to extend it). In other words, unless the
known artifact types are explicitly defined and accessible in a common
repository, custom built container-based component would not be able to
implement the interaction with other components via such artifact types, in turn
the custom container-based component would not be able to make use of the
interoperability and shareability with other components in a pipeline.

We anticipate that
[standard_artifacts.py](https://github.com/tensorflow/tfx/blob/0.14.0/tfx/types/standard_artifacts.py)
will serve as the catalog of known artifact types. We also anticipate that this
catalog might evolve with more properties of a type, or more types themselves.

### Interoperability of Artifact Payload

In order for a custom component to be interoperable with other parts of the
TFX system, the payload of artifacts must be compatible with the way metadata
(via properties of artifacts) defines, which allows downstream components to
properly consume artifacts. In fact, in TFX 0.14.0, there is implicit assumption
on payload of artifacts. For example, payload of *Model* artifact is always
TensorFlow SavedModel with a certain signatures that downstream component,
such as Pusher (and the serving system it pushed to), can consume. Likewise,
payload of *Example* artifact is GZipped tensorflow.Example TFRecord.

Any custom componenent, regardless of the proposed container-based component
or [Python class](https://www.tensorflow.org/tfx/guide/custom_component),
implementation, mismatch in assumed payload would cause runtime error.
This is analogous to the fact that Pandas [`DataFrame.to_csv()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)
and subsequent [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) must implement the same format options
(such as delimiter, quote, header).

We believe that best-possible way handle such ambiguity is to enforce
project-internal consistency within user projects by convention on artifact
properties. This approach will retain capability to implement a logic in
custom components to enforce payload compatibility between components at
DAG complination time. Once this has proven sufficiently generally useful,
some of such convention would be admitted to the central artifact type/property
repository as mentioned in the previous section, and compile time payload
compatibility check logic would be admitted to the TFX's core library.
