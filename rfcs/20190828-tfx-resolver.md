# TFX Resolver

Status        | Accepted
:------------ | :--------------------------------------------
**Author(s)** | Ruoyu Liu (ruoyu@google.com)
**Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com)
**Updated**   | 2019-08-28

## Objective

This RFC proposes the design of Resolver, which serves as an optional plugin in
TFX DSL to handle artifact resolution before a component execution. The
following can be achieved with this design:

*   Enable artifact resolution across pipeline run boundaries
*   Make artifact resolution easy to customize directly through pipeline
    definition

## Motivation

In the original design of TFX orchestration, Driver is used to prepare all
artifacts needed for a component execution and feed the result artifacts into
Executor for execution. The default behavior of input artifact resolution is to
take the outputs of upstream components in the same pipeline run. Any behavior
other than that requires a customized driver. While Driver is sufficient in
terms of functionality, it is essentially a blackbox for TFX end users that is
hard to reason about. Customization and maintenance are also hard since a Driver
also contains other logic such as execution decisions making.

To address the aforementioned problem, we propose to extract the artifact
resolution part into a separate unit, named Resolver. It has the following
attributes:

*   It is an optional plugin. Users that do not need this feature do not need to
    understand Resolver so that simple use cases still remain simple
*   It is easy to understand. Pipeline authors and users no longer need to dig
    into hidden driver code to reason about the artifacts' flow into a component
*   It is easy to write and test. A Resolver definition is no more than a lambda
    expression

## Detailed Design

### API

A Resolver contains the definition of how to query back artifacts given source
Channels, an optional configuration and the access to historical context of
previous artifacts and executions. The API is similar as below:

```python
class BaseResolver(object):
  def __init__(self, configuration: Optional[Any] = None):
    self._configuration = configuration

  @abstractmethod
  def resolve(
      self,
      metadata_handler: metadata.Metadata,
      input_dict: Dict[Text, Channel]) -> Dict[Text, Iterable[Artifact]]:
    raise NotImplementedError
```

The parameter `metadata_handler` passed into `resolve()` is read-only since no
write should be allowed during artifact resolution stage. The other parameter
input is a mapping from tags to Channels. Each Channel provides the type
information that will be used when querying ML metadata.

### DSL integration

There are two options to integrate *Resolver* into TFX DSL:

1.  Make *Resolver* an optional parameter for component interface
2.  Build a special node *ResolverNode* as the wrapper of *Resolver* logic and
    make it independent of existing component interface. The definition of
    *ResolverNode* is shown below

```python
class ResolverNode(BaseNode):
  def __init__(self,
               name: Text,
               resolver: Type[BaseResolver],
               **kwargs: Channel):
    ...
```

We choose to adopt option (2) for the following reasons:

*   It keeps simple cases simple. Users do not need to care about *Resolver* if
    there is no need for cross-pipeline-run artifact resolution
*   It has cleaner and clearer interface than Option (1), especially when
    cross-pipeline-run artifact resolution is needed only for some of the inputs
    to a component
*   It allows not only resolution logic sharing but also resolution results
    sharing. Instead of repeating the same *Resolver* multiple times,
    ResolverNode allows reusing artifact resolution results with little work

### Example

The following example demonstrate our design. There are a couple of requirements
in this scenario:

*   Train with the latest n pieces of Example artifacts, including the one
    produced within the pipeline run
*   Transform and Trainer should operate on the same set of Example artifacts

First, create a new resolver that implements the desired artifact resolution
logic:

```python
# This class implements an artifact resolution logic that will return the latest
# n artifacts for each given Channel.
class LatestRollingWindowResolver(BaseResolver):
  def resolve(
      self,
      metadata_handler: MetadataStore,
      Input_dict: Dict[Text, Channel]) -> Dict[Text, Iterable[Artifact]]:
    result = {}
    for key, source_channel in input_dict.items():
      result[key] = self._get_artifacts_from_channel(
          metadata=metadata_handler,
          channel=source_channel,
          sort_fn=_latest_first,
          maximum_count=self._configuration.window)
    return result
```

Next, create a new ResolverNode instance in the pipeline definition. An instance
of `LatestRollingWindowResolver` is passed in to serve as the resolution logic
unit. Since `transform` and `trainer` all use the output of the same
ResolverNode instance, they will share the same artifact resolution results.

```python
def create_pipeline():
  ...

  example_gen = CsvExampleGen(input_base=...)

  resolver_node = ResolverNode(
      examples=example_gen.outputs['examples'],
      resolver=LatestRollingWindowResolver(generate_config(window=5)))

  transform = Transform(
      examples=resolver_node.outputs['examples'],
      ...)

  trainer = Trainer(
      examples=resolver_node.outputs['examples'],
      transform_output=transform.outputs['transform_output'],
      ...)
  ...

```

## Future work

With the ability to resolve artifacts from past runs, continuous training can be
enabled to take us one step further in ML production automation.
