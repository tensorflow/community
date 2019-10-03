# TFX Iterative Notebook Proposal

Status        | Approved
:------------ | :-------
**Author(s)** | Charles Chen (<a href="mailto:ccy@google.com">ccy@google.com</a>), Joe Lee (<a href="mailto:joeyounglee@google.com">joeyounglee@google.com</a>), Kenny Song (<a href="mailto:kennysong@google.com">kennysong@google.com</a>), Kevin Haas (<a href="mailto:khaas@google.com">khaas@google.com</a>), Pushkar Joshi (<a href="mailto:pushkarj@google.com">pushkarj@google.com</a>)
**Sponsor**   | Konstantinos Katsiapis (katsiapis@google.com)
**Updated**   | 2019-09-17

## Objective

We want to build a notebook user experience for modeling / iterative development
using TFX Components. This will provide a fast, familiar environment for
developing model and pipeline code with standard TensorFlow + TFX utilities,
plus automatic notebook → pipeline export:

*   Imperative,
    [define-by-run](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html),
    cell-by-cell workflow
    *   Start directly from Notebook/Colab – no running pipeline needed
    *   Run TFX components as you need them, in separate cells
    *   No explicit DAG definitions or continuous execution
*   Simple Python API per TFX component
    *   ExampleGen, StatsGen, SchemaGen, Transform, Trainer, Evaluator
    *   100% TFX compatible for automatic notebook → pipeline export
*   Analyze artifacts natively in Notebook/Colab
    *   Built-in TensorBoard, Facets, TFMA visualizations
    *   Dataset, stats, eval metrics available in notebook for custom analysis
*   Zero-setup, interactive onboarding tool for new TFX users on
    [tensorflow.org](http://tensorflow.org)

## Motivation

The benefits of using a notebook include rapidly editing and running code,
immediately seeing the execution and outputs of commands, and running quick
one-off analyses in Python. It’s a simple, no-mental-overhead REPL environment
for iterating on ideas.

By combining the notebook experience + TFX components, users can easily run

*   ExampleGen to generate the initial dataset used for training
*   StatsGen to generate and visualize a statistical report of your data
*   SchemaGen to generate a schema of your data (required input of Transform)
*   Transform to write feature engineering strategies
*   Trainer that wraps standard TF.Estimator or Keras code
*   Evaluator to generate, slice, and visualize evaluation metrics
*   Custom analyses on the output of any of these components with standard
    Python

To close the loop, the notebook will be automatically exported as a pipeline
configuration that users can directly deploy as a scalable TFX pipeline. There
is no additional modification required.

## Target Users

We target users who want to manually iterate on their models & components, and
prefer a notebook environment for the benefits outlined above. This is a wide
range of potential users, and from our user research, spans software engineers
and data scientists within and outside of Google.

## Design Proposal

This proposal proposes a set of primitives that match concepts in the current
TFX SDK.

### Component definition; inputs and outputs

#### Proposal: components should take inputs, produce outputs (instead of taking predefined upstream components)

This proposal proposes a set of primitives that match concepts in the current
TFX SDK. We propose to follow the current TFX style of having components
explicitly take input channels (i.e. streams of artifacts of a specific type)
and produce output channels (of another specific type). This could look like
this:

```
# Here, with an input_base as an execution parameter with a given
# file path.
example_gen = CsvExampleGen(input_base=examples)

# Next, we use the 'examples' named output of ExampleGen as the
# input to StatisticsGen.
statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

# We then similarly use the statsgen output in SchemaGen.
infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'])

# Next, we do example validation.
validate_stats = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=infer_schema.outputs['schema'])
```

### Component execution, execution result objects, visualization

#### Proposal: InteractiveContext.run(component) returns an ExecutionResult, whose output artifacts can be visualized using InteractiveContext.show(artifacts)

##### Part 1 (Option 1): add InteractiveContext.run() **[recommended]**

We propose to add a new `InteractiveContext` class. Distinct from a pipeline
runner which takes in an entire TFX pipeline, an instance of this class allows
interactive execution of individual components. Here, a user would construct
components with appropriate parameters and execution properties, and the
`InteractiveContext.run(component)` method would execute that component, thereby
materializing any output artifacts of that component.

An advantage of this style is that it does not bifurcate the TFX pipeline runner
concept into "pipeline runners" and "component runners", and it is very clear
that this API is only meant for interactive usage (as opposed to the two
alternatives below). A disadvantage is that we may not want to introduce
interactive usage as a first class citizen, preferring to merge it with the
runner concept.

(A prototype for this is implemented in
[taxi_pipeline_interactive.ipynb](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_interactive.ipynb)).
See the "Example notebook usage" section below.

##### Part 1 (Option 2): add Component.run()

In this alternative, we propose to add a run() method to the
[BaseComponent](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_component.py)
class. Given the appropriate parameters and execution properties, this will run
that component of your pipeline. This will be in-process and not involve any
external TFX orchestrators (like Airflow or Kubeflow) and is suitable only for
small development datasets.

An advantage of the Component.run() style is that it is simple and intuitive in
the notebook setting. A disadvantage is that this does not encourage the best
practice for production pipeline definition (i.e. defining all pipeline
components and subsequently calling something like Pipeline.run()). To mitigate
this, we can emit a warning when this is called outside a Jupyter notebook
environment.

An advantage of returning an explicit ExecutionResult is that we now separate
component definition (configuration) from results for a specific run
(execution).

##### Part 1 (Option 3): don't add Component.run(); have separate run_component()

Alternatively, we don't have to put the run() method on the Component class. We
can factor out a utility method `run_component(component)` that does the same
thing. This style is less intuitive for the notebook use case but may better
encourage best practices during production.

##### Part 2 (Option 1): a user can visualize outputs of an ExecutionResult by using Jupyter visualizer for artifact class, or by using InteractiveContext.show(artifact) **[recommended]**

Here, after a `InteractiveContext.run(component)` call, we get an
ExecutionResult, on which we can retrieve artifacts with
`result.outputs[output_name]`. This will return the Artifact pointers emitted by
that specific component execution. Next, the user may return
`component.output[output_name]` as the return value from a notebook cell.
Alternatively, a user may call
`InteractiveContext.show(component.output[output_name])` which hooks into
artifact-specific logic to visualize each artifact type (see Part 3 below).

##### Part 2 (Option 2): Artifact execution via show(artifact)

In this alternative, instead of running components and retrieving artifacts
after they are run, artifacts are "run" implicitly when show(artifact) is
called. This will implicitly execute the component necessary to generate the
artifact.

Pros: One show() call rather than separate run() and show(). Dependencies can be
handled under the hood, and we can avoid visualizing stale results.

Cons: Not intuitive as this is not what “show” means. Not the simplest mental
model and potentially confusing. If a user wants to always show artifacts after
running, it is very natural to put run() and show() in the same notebook cell.
Running code and components that are not part of the current executed cell is
also not a notebook-friendly pattern.

##### Part 3: Notebook visualizations for well-known artifact types can be registered

We introduce a `NotebookArtifactVisualizationRegistry` class on which we may
register visualizations (e.g. HTML renderings for Colab / Jupyter notebooks),
which are to be returned from ExecutionResult.read() when run in the notebook
environment. For specific artifact types, we allow registration of handlers to
return visualizations for those types. We will write visualizations for
well-known artifact types we use. For example, the `ExampleStatistics` Artifact
type output by StatisticsGen could be visualized by producing an interactive
display of the resulting statistics
[using Facets](https://pair-code.github.io/facets/).

##### Example notebook usage

Here is an example of what notebook execution may look like in this scheme.

**Input[0]:**

```python
# To begin, we initialize an interactive context. Here, by not passing
# in a base directory or metadata configuration, we create an ephemeral
# context whose outputs will be in a temporary directory.
context = InteractiveContext()

# Alternatively, we may pass in these directories for a context using a
# persistent store:
#
# context = InteractiveContext(base_dir=my_base_dir,
#                              metadata_connection_config=my_config)
```

**Input[1]:**

```python
# First, ExampleGen with a run / read.
example_gen = CsvExampleGen(input_base=examples)

# Note that the output will be of type 'ExamplesPath', for which we
# may have registered a notebook visualization handler.
example_gen_result = context.run(example_gen)

example_gen.outputs['examples']

# alternative style: explicit context.show() method
context.show(example_gen.outputs['examples'])
```

**Output[1]:**

_(notebook visualization indicating we have N examples at some temp path)_

**Input[2]:**

```python
# Next, StatisticsGen with a run / read.
statistics_gen = StatisticsGen(input_data=example_gen.outputs['examples'])

context.run(statistics_gen).outputs['statistics']

# alternative styles:
# context.show(context.run(statistics_gen).outputs['statistics'])
# context.run().read('output', visualization_handler=blah)
# context.run().show('output', visualization_handler=blah, visualization_args=)
```

**Output[2]:**

_(notebook visualization for statsgen output)_

**Input[3]:**

```python
# Next, SchemaGen without a run / read.
infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'])

# Finally, ExampleValidator with a run / read. Note that SchemaGen
# will be implicitly run (see Note 2 below).
validate_stats = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=infer_schema.outputs['schema'])

context.run(validate_stats)
```

**Output[3]:**

_(ExecutionResult object for ExampleValidator)_

Note that the user may have forgotten to run InteractiveContext.run() on
upstream components in the dependency graph. Instead of implicitly running these
upstream components, we remind the user to explicitly run upstream notebook
cells (with a readable error message). We think this explicit mode of component
execution is more notebook-friendly, and is easy to use with common notebook
actions such as “Run All”, “Run Cells Before”, and “Run Cells After”.

### Export to a selected orchestration engine (v0)

#### Filter out InteractiveContext() objects

##### Option 1: Replace InteractiveContext instances with dummy versions.

1.  Search for possible import alias, e.g. `from
    tfx.orchestration.interactive.interactive_context import InteractiveContext
    as FooBar`

*   Search for all instances of string ".*InteractiveContext", or the alias name
    if found from prior step.
*   Replace each instance with `DummyInteractiveContext`, which inherits from
    InteractiveContext and basically does nothing / returns empty
    ExecutionResult on .run().

    ```
    class DummyInteractiveContext(InteractiveContext):
    def run(self,
      component: base_component.BaseComponent,
      enable_cache: bool = True):
    return None
    ```

    1.  This should cover the case where the class definition is aliased.

        ```
        aliased_class = interactive_context.InteractiveContext
        context = aliased_class()
        ```

    1.  This should cover subsequent aliases of InteractiveContext instances.

        ```
        a = InteractiveContext()
        b = a
        c = InteractiveContext()
        d = c
        ```

Cons:

*   DummyInteractiveContext is now present/clutters the production pipeline code
    (it's a no-op so mainly affects readability, not execution).
*   Down the line, converting back to a notebook (replacing
    DummyInteractiveContext with InteractiveContext) could be fragile.

##### Option 2: Ensure InteractiveContext only runs in notebook context. **[recommended]**

*   If InteractiveContext is run outside of a notebook context, just log a
    warning and return.
*   Bi-directional import to notebook from pipeline would "just work".

Cons:

*   InteractiveContext is still present in the production pipeline as a no-op /
    affects readability.
*   Puts the burden on user to scrub out calls to InteractiveContext.

##### Option 3: Mark lines/cells to be skipped during export.

Add custom magic to mark lines/cells as skip_for_export, can also be used by the
user to skip scratch work in cells.

Example line magic:

```
%skip_for_export context = InteractiveContext()
...
%skip_for_export context.run(example_gen)
```

Example cell magic:

```
%%skip_for_export
# Cell contains scratch work that doesn't need to be exported.
...
```

Cons:

*   Puts burden on the user to filter out the InteractiveContext objects. User
    may forget to mark some usages of InteractiveContext, meaning
    InteractiveContext instances can get leaked to the final pipeline.

##### Option 4: Delete the lines containing InteractiveContext variables.

Cons:

*   Not robust to duplicate references.
*   We can find the references to InteractiveContext by either keeping track of
    them weakly within the class on __init__, or we can use gc module to
    dynamically find the references. But then finding and deleting all
    associated lines with each instance seems hard.
    *   What if user makes a helper function and passes in a context variable?
        (not likely, but possible)

Note each of these options only filters the InteractiveContext usage in the
exported python script, and does not prevent the user from adding it back
afterwards.

#### Export notebook contents to pipeline

1.  Present the user with a Beam pipeline export template cell. Airflow/Kubeflow
    template code can be linked to in documentation, or populated in additional
    cells with code commented.
    1.  User fills out any globals/configuration code specific to
        Beam/Airflow/Kubeflow.
    1.  User fills out a `pipeline.Pipeline()` instance to export.
        1.  We can alternatively have the user wrap the pipeline.Pipeline()
            instance in a function, like `_create_pipeline(...)` in the existing
            [pipeline examples](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py),
            but this could make pipeline export more cumbersome for users who
            have not organized their notebook in such a way. We could also
            potentially update the notebook example to push users into a
            particular notebook organization.
1.  When the user runs the cell, or more specifically, when
    `context.export_to_pipeline()` is executed, export the notebook code to .py
    file.
    1.  It seems beneficial to keep the export_to_pipeline() line in the same
        cell as the pipeline.Pipeline() declaration so the user can fix any
        errors before the export happens.
    1.  As a first pass, we can export the entire notebook.
        1.  We may consider using IPython magics to filter out specific
            lines/cells in the future.
        1.  This step requires the user to fill out the notebook filename as
            there does not seem to be a robust way for us to programmatically
            retrieve this (see comment in examples below).
    1.  We can try to hide away parts of the template cells in the notebook and
        move them into Jinja template files, but if the user has to fill in
        pipeline-specific config, it might be more straightforward for them to
        see everything in one place.

##### Airflow template cell

```
# Airflow template cell.

from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner

##############################################################################
# TODO(USER): Configs

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
    'metadata.db')

# Airflow-specific configs; these will be passed directly to Airflow.
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}
##############################################################################

# TODO(USER)
p = pipeline.Pipeline(
    pipeline_name=,
    pipeline_root=,
    components=[
        example_gen, statistics_gen, infer_schema, validate_stats, transform,
        trainer, model_analyzer, model_validator, pusher
    ],
    enable_cache=True,
    metadata_connection_config=metadata.sqlite_metadata_connection_config(
        metadata_path))

airflow_pipeline = AirflowDAGRunner(_airflow_config).run(p)

# Export notebook contents.
context = InteractiveContext()
# TODO(USER): Name of the notebook file to be used for retrieving
# notebook contents. IPython kernels are agnostic to notebook metadata by design,
# and it seems that existing workarounds to retrieve the notebook filename are not
# universally robust (https://github.com/jupyter/notebook/issues/1000).
context.export_to_pipeline(notebook_filename='taxi_pipeline_interactive.ipynb',
                           pipeline_name='')
```

##### Kubeflow template cell

```
# Kubeflow template cell.

from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.runner import KubeflowRunner


##############################################################################
# TODO(USER): Configs

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
_tfx_root = os.path.join(_output_bucket, 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'my-gcp-project'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
_module_file = os.path.join(_input_bucket, 'taxi_utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_output_bucket, 'serving_model',
                                  _pipeline_name)

# Region to use for Dataflow jobs and AI Platform training jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {...}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {...}

# Beam args to run data processing on DataflowRunner.
_beam_pipeline_args = [...]

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.001  # Generate a 0.1% random sample.

# This is the upper bound of FARM_FINGERPRINT in Bigquery (ie the max value of
# signed int64).
_max_int64 = '0x7FFFFFFFFFFFFFFF'

# The query that extracts the examples from BigQuery.  The Chicago Taxi dataset
# used for this example is a public dataset available on Google AI Platform.
_query = ...
##############################################################################

# TODO(USER)
p = pipeline.Pipeline(
    pipeline_name=,
    pipeline_root=,
    components=[
        example_gen, statistics_gen, infer_schema, validate_stats, transform,
        trainer, model_analyzer, model_validator, pusher
    ],
    additional_pipeline_args={
        'beam_pipeline_args': beam_pipeline_args,
        # Optional args:
        # 'tfx_image': custom docker image to use for components.
        # This is needed if TFX package is not installed from an RC
        # or released version.
    },
    log_root='/var/tmp/tfx/logs')

kubeflow_pipeline = KubeflowRunner().run(p)

# Export notebook contents.
context = InteractiveContext()
# TODO(USER): Name of the notebook file to be used for retrieving
# notebook contents. IPython kernels are agnostic to notebook metadata by design,
# and it seems that existing workarounds to retrieve the notebook filename are not
# universally robust (https://github.com/jupyter/notebook/issues/1000).
context.export_to_pipeline(notebook_filename='taxi_pipeline_interactive.ipynb',
                           type='kubeflow')
```

