# TFX Container Component Execution Proposal

Status         | Proposed
:------------- | :-------
**Author(s)**  | Ajay Gopinathan (<a href="mailto:ajaygopinathan@google.com">ajaygopinathan@google.com</a>), Hongye Sun (<a href="mailto:hongyes@google.com">hongyes@google.com</a>), Makoto Uchida (<a href="mailto:muchida@google.com">muchida@google.com</a>), Ruoyu Liu (<a href="mailto:ruoyu@google.com">ruoyu@google.com</a>)
**Sponsor(s)** | Konstantinos Katsiapis (katsiapis@google.com), Pavel Dournov (dournov@google.com), Ramesh Chandra (rameshuc@google.com)
**Updated**    | 2019-08-29

## Objective

This RFC proposes an orchestrator agnostic way to reliably execute a user’s
container in the TFX pipeline. The proposal can support:

*   Running an arbitrary container in either a local Docker environment or a remote
    Kubernetes cluster.
*   Passing data into the container
*   Passing output data from the container
*   Capturing logs from the container
*   Handling errors and retries
*   Cancelling the container execution if the pipeline is terminated

## Motivation

Currently, the execution of a generic container as a step in a TFX pipeline is 
not supported. Without this feature, users cannot bring their own containers
into the pipeline. This blocks the following use cases:

*   User already has a docker image and wants to run the image as one of the
    steps in a TFX pipeline.
*   User wants to use non-Python code (R, for example) as one of the steps in a TFX
    pipeline.
*   User wants to have an isolated Python environment for their component code.

This RFC is a follow-up design for
[Container-based Component RFC](https://github.com/tensorflow/community/pull/146).
This design defines how to execute the container spec as part of a TFX pipeline.
The execution may occurs in local Docker container or in a remote Kubernetes cluster.

### Existing solutions

#### Kubeflow Pipeline (KFP) ContainerOp

Today, KFP’s ContainerOp leverages
[Argo container template API](https://github.com/argoproj/argo/blob/master/pkg/apis/workflow/v1alpha1/workflow_types.go)
to launch user’s container in a Kubernetes pod. Argo, as the orchestrator, controls when
to launch the POD and it uses a sidecar container to report output files back
and wait for user’s container to complete. We are not proposing to use Argo API
because of the following reasons:

*   Argo’s API is orchestrator-specific and cannot be ported to Airflow or local
    runners.
*   Argo’s API doesn’t provide an extensible way to run custom code before and
    after POD API, which is critical to support metadata tracking and caching
    features.
*   Argo doesn’t provide an easy way to recover from user’s transient errors,
    which is critical in production workload.

#### Airflow Kubernetes pod operator

Airflow supports launching a Kubernetes pod by an
[operator](https://github.com/apache/airflow/blob/master/airflow/contrib/operators/kubernetes_pod_operator.py).
This approach is closer to what we are proposing in the document. However, we
cannot directly use the operator because:

*   Airflow operator requires to be run inside an Airflow pipeline which is not
    the case for local and KF runners.
*   Airflow operator exposes a subset of POD’s API, where we want to expose the
    full pod spec to the user.
*   Airflow operator doesn’t provide a reliable way to retry user’s container
    and recover from transient errors.
*   Airflow does not support initializing an operator inside another operator.
    Going back to using multiple Airflow operators for a component is a regression
    now that we have `BaseComponentLauncher` ready.

## Proposed Design

### TLDR

We propose to solve the above problems with the following design:

*   Define a container as an executor spec.
*   Launch a container via a component launcher in either a local docker or Kubernetes pod.
*   Use a platform config to specify a platform-specific settings config.

The proposed solution has the following parts:

*   Extensible `ExecutorSpec` concept which can support a container as an
    executor.
*   Extensible `BaseComponentLauncher` concept to support pluggable component
    launchers in a TFX runner.
    *   `DockerComponentLauncher` which launches `ExecutorContainerSpec` in
        a Docker environment.
    *   `KubernetesPodComponentLauncher` which launches `ExecutorContainerSpec`
        in a Kubernetes environment.
*   Extensible `PlatformConfig` framework.
    *   `KubernetesPodPlatformConfig` to support Kubernetes pod spec as a config.
    *   `DockerPlatformConfig` to support docker run configs.

### Architecture

Architecture that allows local container execution.

![TFX local container execution](20190829-tfx-container-component-execution/tfx-local-container-execution.png)

Architecture that allows Kubernetes container execution.

![TFX Kubernetes container execution](20190829-tfx-container-component-execution/tfx-k8s-container-execution.png)

Class diagram that allows container execution

![TFX container execution_classes](20190829-tfx-container-component-execution/tfx-container-execution-classes.png)

### Python DSL experience

In order to use container base component in TFX DSL, user needs follow these
steps. Step 1 and Step 2 follow the DSL extension proposed by [TFX Generic Container-based Component](https://github.com/tensorflow/community/pull/146).

#### Step 1: Define the container based component by `ExecutorContainerSpec`

```python
class MyContainerBasedExampleGen(BaseComponent):

  SPEC_CLASS = types.make_spec_class(
    inputs={
      "raw_input": ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }
    outputs={
      "examples": ChannelParameter(type=standard_artifacts.Examples),
    }
    parameters={
      "num_csv_columns": ExecutionParameter(type=int),
    }
  )

  EXECUTOR_SPEC = ExecutorContainerSpec(
    container_image='gcr.io/my_project/my_example_gen:stable',
    command=['python'],
    args=['my_example_gen.py',
          '--input_csv_file', '{{ inputs.raw_input.uri }}',
          '--output_examples', '{{ outputs.examples.uri }}',
          '--num_csv_columns', '{{ exec_props.num_csv_columns }}' ],
  )
```

#### Step 2: Create pipeline from container based component

```python
def create_pipeline():
  my_csv_file = Channel('CSVFile', uri="/path/to/csv_file")

  my_example_gen = MyContainerBasedExampleGen(
        raw_input=my_csv_file, num_csv_columns=20)

  return pipeline.Pipeline(
    pipeline_name = 'my_container_based_pipeline',
    pipeline_root = 'gs://path/to/root',
    components = [my_example_gen],
    ...
  )
```

#### Step 3(a): Set docker config via runner’s config

```python
_ = BeamRunner(platform_configs={
  'MyContainerBasedExampleGen': [DockerPlatformConfig(volumes={...})]
}).run(create_pipeline())
```

#### Step 3(b): Set Kubernetes platform config via runner’s config

```python
_ = KubeflowDagRunner(platform_configs={
  'default': [KubernetesPodPlatformConfig(Pod().use_gcp_secret().spec()]
  'MyContainerBasedExampleGen': [
      KubernetesPodPlatformConfig(Pod(cpu=2, memory='1GB').spec())]}
).run(create_pipeline())
```

### Component launcher

A component launcher launches a component by invoking a driver, an executor and
a publisher. It understands how to launch a component executor from an
`ExecutorSpec`. The `BaseComponentLauncher` is an abstract base class with two
abstract methods:

*   `can_launch`: public method to check whether the launcher can launch an
    instance of `ExecutorSpec` with a specified `PlatformConfig` instance. The
    method will be used by `TfxRunner` to choose launcher for a component.
*   `_run_executor`: a protected method to launch an `ExecutorSpec` instance.
    The method is invoked by `BaseComponentLauncher.launch` method.

Subclasses of the base component launcher can support launching executors in
different target platforms. For example:

*   `InProcessComponentLauncher` can launch an executor class in the same Python
    process.
*   `DockerComponentLauncher` can launch a container executor in a Docker
    environment.
*   `KubernetesPodComponentLauncher` can launch a container executor in a Kubernetes
    environment.
*   A Dataflow launcher can launch a beam executor in Dataflow service.

Pseudo implementation:

```python
class BaseComponentLauncher(with_metaclass(abc.ABCMeta, object)):
  @abc.abstractmethod
  @classmethod
  def can_launch(cls, executor_spec: ExecutorSpec,
                 platform_spec: Optional[PlatformConfig]) -> bool:
    return False

  @abc.abstractmethod
  def _run_executor(execution_id: int,
                   input_dict: Dict[Text, List[types.Artifact]],
                   output_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> Any:
    pass

class InProcessComponentLauncher(BaseComponentLauncher):
  # InProcessComponentLauncher implements default launcher for python executor.
  # It doesn't support platform_spec.
  @classmethod
  def can_launch(cls, executor_spec: ExecutorSpec,
                 platform_spec: Optional[PlatformConfig]) -> bool:
    if platform_spec:
      return False
    return isinstance(executor_spec, ExecutorClassSpec)

  def _run_executor(execution_id: int,
                   input_dict: Dict[Text, List[types.Artifact]],
                   output_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> Any:
    # Python in process launcher implementation.
    # Subclass should override this method to implement platform launcher
    …

class DockerComponentLauncher(BaseComponentLauncher):

  @classmethod
  def can_launch(cls, executor_spec: ExecutorSpec,
                 platform_spec: Optional[PlatformConfig]) -> bool:
    if not isinstance(executor_spec, ExecutorContainerSpec):
      return false

    if not platform_spec:
      return true

    return isinstance(platform_spec, DockerPlatformConfig):

  def _run_executor(execution_id: int,
                   input_dict: Dict[Text, List[types.Artifact]],
                   output_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> None:
    # Docker launcher implementation
    ...

class KubernetesPodComponentLauncher(BaseComponentLauncher):
  @classmethod
  def can_launch(cls, executor_spec: ExecutorSpec,
                 platform_spec: Optional[PlatformConfig]) -> bool:
    if not isinstance(executor_spec, ExecutorContainerSpec):
      return false

    if not platform_spec:
      return true

    return isinstance(platform_spec, DockerPlatformConfig):

  def _run_executor(execution_id: int,
                   input_dict: Dict[Text, List[types.Artifact]],
                   output_dict: Dict[Text, List[types.Artifact]],
                   exec_properties: Dict[Text, Any]) -> None:
    # Kubernetes pod launcher implementation
    …
```

### Platform config

Platform config carries platform specific configs. Usually, one platform config
type maps to one component launcher type. For example,
`DockerPlatformConfig` can only be used by `DockerComponentLauncher` and
`KubernetesPodPlatformConfig` can only be used by
`KubernetesPodComponentLauncher`.

Each platform config can be merged with another config with the same type. This
capacity is needed to support a layered configuration system in runner’s config:

*   User can define a default platform config list which will be applied to all
    components in the pipeline.
*   User can define component specific config by using component’s name as a
    selector.
*   Component specific config should override the default config.

Pseudo implementation:

```python
class PlatformConfig(with_metaclass(abc.ABCMeta, object)):
  def merge(self, platform_config: PlatformConfig) -> PlatformConfig:
    """Merge the current config with a new config.
    Usually, it only happens when component config is merged with default config.
    """
    # Simple recursive dictionary merge logic

class DockerPlatformConfig(PlatformConfig):
  def __init__(self, **kwargs):
    # The kwargs is the same as the list defined in
    # https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.ContainerCollection.run
    self.run_kwargs = kwargs

class KubernetesPodPlatformConfig(PlatformConfig):
  def __init__(self, pod_spec: V1PodSpec):
    self.pod_spec = pod_spec
```

#### Pod spec layers

A final pod spec is merged by 3 layers of pod specs. They are:

*   Base pod spec layer
*   Default config spec layer
*   Component specific config spec layer

The merge logic follows
[strategic merge patch](https://kubernetes.io/docs/tasks/run-application/update-api-object-kubectl-patch/#use-a-strategic-merge-patch-to-update-a-deployment)
to merge layers in order: base -> default -> component config.

Strategic merge patch is different from JSON patch by merging lists and maps
instead of replacing them entirely. In this way, the patch layer doesn’t have to
specify the full content of a list or map.

The base pod spec layer is created from user’s container spec. The pod spec
includes a main container spec with image path and entrypoint of the container.

Default and component platform configs are configured by runner’s constructor.

For example:

```yaml
# base pod spec
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: main
    image: tensorflow/tensorflow:v1.13
    command: ["python", "-c", "ml/app.py"]

# pipeline pod spec
spec:
  serviceAccountName: PipelineRunner
  containers:
  - name: main
    resources:
      limits:
        memory: "128Mi"
        cpu: "500m"

# component config pod spec
spec:
  containers:
  - name: main
    env:
    - name: MYSQL_ROOT_PASSWORD
      value: "password"

# final pod spec
apiVersion: v1
kind: Pod
spec:
  serviceAccountName: PipelineRunner
  containers:
  - name: main
    image: tensorflow/tensorflow:v1.13
    command: ["python", "-c", "ml/app.py"]
    resources:
      limits:
        memory: "128Mi"
        cpu: "500m"
    env:
    - name: MYSQL_ROOT_PASSWORD
      value: "password"
```

### TFX runner

A `TFXRunner` compiles a logical pipeline into the underlying orchestrator’s DSL. In
this proposal, the base runner should accept launchers and `platform_configs`
and provide a default strategy to choose launcher for each component.

The default choosing logic is:

*   If `platform_configs` is set, use it along with executor spec to find the
    first launcher which can support them.
*   Otherwise, find the first launcher which can support the executor spec
    without `platform_configs`.
*   `platform_configs` has higher priority than `default_platform_configs`.

Pseudo implementation:

```python
class TfxRunner(with_metaclass(abc.ABCMeta, object)):
  def __init__(self, launchers: List[BaseComponentLauncher],
               platform_configs: Dict[Text, List[PlatformConfig]]):
    self.launchers = launchers
    self.default_platform_configs = platform_configs.get('default')
    self.platform_configs = platform_configs

  def _get_component_launch_info(
      self, component: BaseComponent) -> ComponentLaunchInfo:
    component_platform_configs = self.platform_configs.get(component.name)
    # Use PlatformConfig.merge to merge configs with the same type.
    component_platform_configs = self._merge_platform_configs(
      component_platform_configs, self.default_platform_configs)
    # Select launcher by platform config.
    for platform_config in component_platform_configs:
      for launcher in self.launchers:
        if launcher.can_launch(component.executor_spec, platform_config):
          return ComponentLaunchInfo(component, launcher, platform_config)
    for launcher in self.launchers:
      if launcher.can_launch(component.executor_spec):
        return ComponentLaunchInfo(component, launcher)

  def run(self, pipeline) -> Optional[Any]:
    component_launcher_infos = {c.name: self._get_component_launch_info(c)
                                for c in pipeline.components)}
    return self._run(self, pipeline, component_launcher_infos)

  @abc.abstractmethod
  def _run(self, pipeline, component_launcher_infos) -> Optional[Any]:
    pass
```

### Output interface

User container can receive a
[tmp directory path from default artifact store](https://github.com/tensorflow/community/blob/2c0b009ef955975b15a3cc18b1378e0ed38f394e/rfcs/20190904-tfx-generic-container-based-component.md#artifact-properties-after-execution-is-complete)
to write output data. The directory parameter will be called
`exec_properties.tmp_path`, which can be passed in as a command line argument.
The executor will look for `output.json` file under `exec_properties.tmp_path`
to get the outputs from the component. The output file follows the following
schema:

```yaml
"$id": https://pipeline.mlx.org/output.schema.json"
"$schema": http://json-schema.org/draft-07/schema#"
type: object
title: Output
properties:
  error_status: { "$ref": "#/definitions/OutputErrorStatus" }
  outputs:
    type: object
  exec_properties:
    type: object
definitions:
  OutputErrorStatus:
    type: object
    properties:
      code:
        type: string
        enum: [PERMANENT_ERROR, RETRYABLE_ERROR]
      message:
        type: string
```

The output.json file is optional, but if the user’s container writes to the file. It
overrides the default handling of the Kubernetes pod launcher. The output fields are:

*   error_status: tells the executor whether it should retry or fail
*   outputs and exec_properties: used to override the execution and
    output artifact metadata in MLMD.

The output interfaces rely on `BaseComponentLauncher` to update states back to
MLMD from executor.

### Auth context resolution

The Kubernetes pod launcher internally uses the Kubernetes Python client. The auth context resolution
logic is as follows:

1.  If the current env is in a cluster, use `load_incluster_config` to load k8s
    context.
1.  If not, use default Kubernetes active context to connect to remote cluster.

### Pod launcher resiliency

In this design section, we focused more on the launcher resiliency under
`KubeflowDAGRunner`. In `AirflowDAGRunner`, the launcher code is running in the
same process of Airflow orchestrator, and we rely on Airflow to ensure the 
resiliency of the process.  `BeamDAGRunner`, however, is considered mainly for local testing
purpose and we won't add support for it to be resilient.

In `KubeflowDAGRunner`, a pipeline step will create two pods in order to execute
user’s container:

*   A launcher pod which contains the driver, Kubernetes pod launcher, and publisher code.
*   A user pod with user’s container.

A pod in Kubernetes is not resilient by itself. We will use Argo’s retry feature to make
the launcher pod partially resilient. The details are as follows:

*   Each Argo launcher step will be configured with a default retry count.
*   Argo will retry the step in case of failure, no matter what type of error.
*   The launcher container will create a tmp workdir in `pipeline_root`.
*   It will keep intermediate results (for example, the ID of the created pod) in the tmp workdir.
*   The Kubernetes pod launcher will be implemented in a way that it will resume the
    operation based on the intermediate results in the tmp workdir.
*   The launcher will also record a permanent failure data in the tmp workdir so
    it won’t resume the operation in case of non-retriable failures.

### Default retry strategy

K8s pod launcher supports exponential backoff retry. This strategy applies to
all runners which can support Kubernetes pod launcher. Docker launchers are not in the
scope of the design as it is mainly for local development use case.

The retry only happens if the error is retriable. An error is retriable only
when:

*   It’s a transient error code from Kubernetes pod API.
*   Or, the output.json file from artifact store indicates it’s a retriable error.
*   Or, the pod get deleted (For example: GKE preemptible pod feature).

### Log streaming

The container launcher streams the log from user’s docker container or Kubernetes pod through the
API. It will start a thread which constantly pulls new logs and outputs them to
local stdout.

## Discussions

*   In the Argo runner, each step requires 2 pods with total 3 containers (launcher
    main container + launcher argo wait container + user main container) to run.
    Although each launcher container requires minimal Kubernetes resources,
    resource usage is still a concern.
    
    * With an additional pod, it gives launcher more control over execution and reduce the discrepancy between different orchestrators. We decided to go with the platform launcher approach and the additional container resource can be ignored.
    
*   For executor container spec, will environment variables be supported?

    *   It’s not added right now to the container spec. Most things could be passed down using command line and arguments. So there is a workaround right now. Also, environment variables can be platform specific. Kubernetes for example, has certain conventions that don’t apply in other platforms. Hence, this could be part of platform_config instead of spec.
    
*   Step 3 indicates we first create a DAG, then use node identity to apply platform configuration. Another possibility is to do it directly during DAG construction. For example, if user goes back and changes the DAG, the platform configuration may stop mapping well, and will need to be revisited by the user. Did we consider the second option?

    *   We don’t want users to add platform configuration at the pipeline level since it won’t be portable. We want the same pipeline to be compatible with local run and say running on k8s. Right now, we’d like to keep the pipeline specification itself clean from platform-specific abstractions.
    
 *  The current proposal uses the component name as the key for binding. Different instantiations may have late binding for their names. For example, if I have 3 ExampleGen, should we be binding to the name instead of the type?
 
    *   The names need to be unique else compilation fails. The instance names are actually component id, which is enforced to be unique at compile time.
    
 *  How is caching affected by container tags? We should be careful with using container tags, since these are mutable. We should be relying on digest instead. If we cannot safely get digest, we should disable caching so we don’t fail due to the inability to obtain the digest at runtime. E.g. ‘latest’ and ‘nightly’ tags are not good candidates
 
    *   By default, if we provide a tag name, the image will be cached in the cluster. We should log an error if caching requirement cannot be met at runtime. Note that currently, executor code changes don’t affect caching behaviour. We should change the above and ensure caching takes the above into account as well.
    
 *  Will we ensure we have adequate test coverage?
 
    *   We will add e2e tests for both Docker container and Kubernetes container launchers. The former using Beam as orchestrator, and the latter using Kubeflow orchestrator.
    
    
 *  What are the major user-facing differences between using TFX DSL with these extensions compared with KFP’s SDK today?
 
    *   here is a difference in how users will specify platform-specific configuration in the pipeline. In KFP’s SDK, the user specifies this in-place when writing the logical pipeline. In TFX DSL, the need to ensure the logical pipeline is portable necessarily means the platform configuration needs to be specified outside the logical pipeline, which may be slightly more cumbersome than the KFP experience today. Note that the separation of configuration sometimes appears in KFP SDK too, when users want to apply global settings.
    
 *  We don’t support or provide mechanisms for users to control container lifetime, e.g. container cancellation.

    *   A lot of cancellation operations are best effort anyway. Failures on cancel operations are hard to handle. Users need to understand from the document that we are not aiming to enable such operations.
    *   If user starts a long-running job from a container, and the pipeline is canceled, users may want the container to receive this message and cancel gracefully.
    *   Can we guarantee that workflows will not stop until we get confirmation of cancellation of long-running operations?
    *   This seems difficult, and best effort may be enough, given that this is all Kubernetes itself does today.
