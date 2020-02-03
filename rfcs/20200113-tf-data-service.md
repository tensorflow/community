# Distributed tf.data service

| Status        | Accepted                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [195](https://github.com/tensorflow/community/pull/195) |
| **Author(s)** | Andrew Audibert (aaudibert@google.com) Rohan Jain (rohanj@google.com) |
| **Sponsor**   | Jiri Simsa (jsimsa@google.com)                          |
| **Updated**   | 2019-01-30                                              |

## Objective

Provide an API and implementation of a tf.data service which can process tf.data
datasets in a distributed manner. The service can be run outside the TensorFlow
cluster or be exported as a gRPC service by TensorFlow servers.

Goals:

-   Enable horizontal scaling of dataset computation to improve performance of
    input-bound dataset pipelines.
-   Improve tf.data integration with the tf.distribute API. In particular,
    support dynamic sharding of data across multiple processes.
-   Provide visitation guarantees for distributed training jobs.

Non-goals:

-   Process non-dataset data.
-   Distribute datasets that rely on external / non-serializable state.
-   Support non-graph computation (e.g. py_function).

## Motivation

### Host machine input pipelines can't always keep up with accelerators.

Some input pipelines require significant resources to produce their data, e.g.
due to image transformations. When the host machine isn't powerful enough to
generate input data at the rate the attached accelerator(s) consume the data,
the accelerator(s) will idle. This slows down training time, and also wastes
valuable accelerator resources. The tf.data service solves this problem by using
N input workers to feed M accelerators. The number of input workers can be
scaled up or down as needed to keep up with the accelerators.

### Distributed training requires a distribution-aware input pipeline.

Today tf.data supports the tf.distribute API by providing mechanisms for
sharding, cloning, and re-batching. The tf.distribute API uses these primitives
to implement their own version of a distributed dataset. If distributed datasets
become a core feature of tf.data, tf.data can provide a public API for
tf.distribute (and users who wish to implement their own distribution) to use
instead. This will also allow us to support feature requests that require
cross-worker coordination, such as dynamic sharding.

## User Benefit

### Input-bound models

Users with input-bound models can leverage the tf.data service to distribute
input processing across horizontally-scaling compute resources. This can improve
utilization for valuable accelerator resources, reducing total cost.

### Dynamic load balancing

Today, the tf.distribute API statically shards data across accelerators. This
can lead to suboptimal utilization because some shards may contain more data
than others. The tf.data service provides a mechanism for dynamically sharding,
reducing the data imbalance across accelerators. Note that dynamic load
balancing and deterministic output are mutually exclusive; if users require
deterministic output, they must trade off dynamic load balancing.

### Visitation guarantees

Model accuracy can often be improved when each training sample is trained on
exactly once per epoch. The tf.data service can coordinate across workers to
provide this guarantee.

## Design Proposal

The tf.data service is a master-worker system which iterates through datasets,
producing outputs to be consumed by accelerators. The service is comprised of a
few components:

*   User-facing Python API for interacting with the tf.data service.
*   Dataset splitting API for determining how to split up datasets for parallel
    processing.
*   Master and worker gRPC services.

### Architecture

The tf.data service is comprised of master and worker gRPC services which could
be run in a couple of different configurations:

#### Glossary

**Master**: The single master coordinating the tf.data service.

**Worker**: A tf.data service worker which performs dataset processing and
provides dataset elements to consumers over RPC.

**Consumer**: A machine which consumes data from the tf.data service. The
consumer may be attached to a GPU or TPU, or use data for on-CPU training.

#### Option 1: Separate Cluster Architecture

Each server is run on a separate host from the TensorFlow cluster. This
configuration gives users a way to provide horizontally scaling CPU for
processing their input pipelines and quickly feeding data to accelerators.

#### Option 2: Embedded Cluster Architecture

Each TensorFlow server runs the tf.data worker gRPC service, and one server also
runs the master gRPC service. This lets users leverage the tf.data service
without needing to provision additional compute resources. and gives all the
benefits of the tf.data service except for horizontal scaling.

#### Option 3: Hybrid Architecture

Users could run tf.data workers embedded in their TensorFlow cluster, and also
run additional tf.data workers (and potentially the tf.data master) outside the
cluster. This allows for horizontal worker scaling, while still leveraging the
compute resources of the TensorFlow cluster for input processing.

### User-facing Python API

This API is how users will interact with the tf.data service from their Python
code. The steps for distributed iteration over a dataset are

1.  Create a dataset like usual.
2.  Apply the `distribute` transformation to indicate that the dataset should be
    processed by the tf.data service.
3.  Begin an *iteration* by calling `create_iteration`. An *iteration* is a
    single pass through the dataset. Multiple consumers can read from the same
    iteration, resulting in each consumer receiving a partition of the original
    dataset. We represent an iteration with an iteration id, which is generated
    by the tf.data service when you call `create_iteration`.
4.  Share the iteration id with all consumer processes which are participating
    in the iteration.
5.  Create per-consumer iterators using `make_iterator`, and use these iterators
    to read data from the tf.data service.

We move away from the idiomatic `for element in dataset` control flow because
there is now an extra step when going from dataset to iterator: creating an
iteration. A higher layer API such as tf.distribute may use the API presented
here to implement datasets which produce per-replica elements, enabling
idiomatic control flow.

```python
def tf.data.experimental.service.distribute(address_or_resolver):
  """Marks that a dataset should be processed by the tf.data service.

  ds = ... # dataset to distribute
  ds = ds.apply(
    tf.data.experimental.service.distribute(address_or_resolver))

  Args:
    address_or_resolver: The address of the tf.data service master, or a
      cluster resolver that can be used to determine the master address.

  Returns:
    A function that can be passed to `dataset.apply()`.
  """

def tf.data.experimental.service.create_iteration(
    dataset, num_consumers=1, num_tasks=None, deterministic=False):
  """Begins distributed iteration over a dataset.

  It is expected that the dataset contains at least one `.distribute(address)`
  transformation, otherwise this method will print a warning and do nothing.

  `create_iteration` will first register the dataset with the tf.data service
  if it isn't already registered. It will then request the creation of
  `num_consumers` dataset iterators which divide the dataset `num_consumers`
  ways. The returned object can be used to read from one of the
  iterators using
  `tf.data.experimental.service.make_iterator(ds, obj, consumer_index)`.

  ds = ... # dataset to distribute
  ds = ds.apply(tf.data.experimental.service.distribute(address))
  if consumer_index == 0:
    # The iteration object is a byte array which needs to be shared among all
    # consumers. Here we suppose there are broadcast_send and broadcast_recv
    # methods available.
    iteration_id = tf.data.experimental.service.create_iteration(ds, 3)
    broadcast_send(iteration_id)
  else:
    iteration_id = broadcast_recv()
  it = tf.data.experimental.service.make_iterator(
      ds, iteration_id, consumer_index)
  for element in it:
    # process element

  Args:
    dataset: The dataset to begin iteration over.
    num_consumers: The number of consumers to divide the dataset between. Set
      this if you require determinism.
    num_tasks: The number of tasks to use for processing. Tasks run for
      the duration of an epoch, and each worker should typically process a single
      task. Normally it is best to leave this as None so that the master can
      choose a reasonable number of tasks. Setting `num_tasks` is useful for
      producing deterministic results.
    deterministic: Whether the iteration should be performed
      deterministically. Fully deterministic output also requires setting
      `num_tasks` to a fixed number, and that the input dataset is itself
      deterministic.

  Returns:
    An iteration_id which can be used to created iterators via
      `tf.data.experimental.service.make_iterator`
  """

def tf.data.experimental.service.make_iterator(
    dataset, iteration, consumer_index=0):
  """Creates an iterator for reading from the specified dataset.

  Args:
    dataset: The dataset to read from.
    iteration: An iteration_id object generated by
      `tf.data.experimental.service.create_iteration`.
    consumer_index: The consumer index within the iteration to read from. If
      the iteration was created with `n` consumers, `consumers_index` must be
      less than `n`.

  Returns:
    A Python iterator which iterates over the dataset elements.
  """
```

### Dataset splitting API

To parallelize dataset processing, the tf.data service needs a way to split up
datasets. We will achieve this by adding a splitting API that allows source
datasets to express how they can be split.

Our goals for the API are

*   Performance: The splitting API can be used to performantly split and process
    datasets.
*   Extensibility: User-defined datasets can be split as long as they implement
    the splitting API.
*   Minimize Surprises: Users write their datasets as though they will not be
    split, so introducing splitting can easily lead to unexpected outcomes. To
    mitigate this, we will be conservative about which dataset transformations
    support splitting.

The API will be used internally by the tf.data service to distribute datasets.
It will be entirely in C++, and we don't currently have any plans to expose
splitting through Python.

The API focuses on producing and consuming `Split`s. A `Split` is a variant
Tensor that can be subclassed to represent arbitrary types of splitting. The
`Split` base class is intentionally general so that subclasses have the
flexibility to define splits however they like.

```cpp
class Split {
 public:
  virtual std::string DebugString() const = 0;
  // Methods to support being used as a Variant tensor.
  virtual std::string TypeName() const = 0;
  virtual void Encode(VariantTensorData* data) const = 0;
  virtual bool Decode(const VariantTensorData& data) = 0;
};
```

To iterate over splits for a dataset, we will use a new
`DatasetBase::MakeSplitGenerator()` method. This method creates a
`SplitGenerator`, which is responsible for generating all of the splits for the
dataset. We use an intermediate `SplitGenerator` object instead of generating
splits directly because there could be a large number of splits, and the
`SplitGenerator` gives us as way to tune split size in response to pipeline
performance.

```cpp
class SplitGenerator {
 public:
  virtual Status GetNext(std::unique_ptr<Split>* split,
                         bool* end_of_splits) = 0;
  // Instructs the SplitGenerator to adjust the size of future splits by the
  // specified percent. 100% means no change, 50% means half-sized splits, and
  // 200% means double-sized splits. The SplitGenerator will make a best effort
  // to incorporate the feedback when creating splits.
  virtual void AdjustSplitSize(int percent) = 0;
};
```

It is tempting to process each split independently, but this would cause issues
when splits are small. tf.data pipelines need to populate internal buffers for
shuffling, prefetching, and batching. If we use a separate pipeline to process
each split, our shuffling will be lower quality, we will have performance jitter
as we keep needing to refill prefetch buffers from scratching, and we will
produce many more partial batches (each split might not even have enough data to
fill a full batch). To avoid these issues, we use a small number of tasks, where
each task processes many splits as a single pipeline.

To enable processing of multiple splits in a dataset, we will add an optional
`SplitProvider` field to the `IteratorContext` passed to
`IteratorBase::Initialize`. The `SplitProvider` produces splits which tell the
iterator what source data to iterate over. For example, if splits are
represented by filenames, and a SplitProvider produces `["file1", "file6",
"file11"]`, an iterator initialized by that `SplitProvider` should process those
three files only.

```cpp
class SplitProvider {
 public:
  virtual Status GetNext(std::unique_ptr<Split>* split,
                         bool* end_of_splits) = 0;
};
```

When processing datasets, tf.data service workers will use `SplitProvider`s
which provide splits by querying the tf.data service master for which splits to
process. A few splits will be prefetched to hide the latency of needing to
request a new split from the master.

#### Supported Datasets

Not all dataset sources and transformations are easily splittable. For example,
`take`, `skip`, and `scan` require a global view of the dataset to produce
correct results. Datasets which require multiple input datasets such as `zip`
are also difficult to support, since we don't have a good way of aligning the
splits of multiple input datasets. Users who rely on these unsupported datasets
will need to move those datasets to come after the distributed part of their
pipeline.

Initially, we will support splitting for the following dataset sources and
transformations:

*   `batch`, `CsvDataset`, `dense_to_sparse_batch`, `filter`,
    `FixedLengthRecordDataset`, `flat_map`, `from_tensor_slices`,
    `group_by_window`, `ignore_errors`, `interleave`, `list_files`, `map`,
    `range`, `repeat`, `padded_batch`, `prefetch`, `shuffle`, `SSTableDataset`,
    `TextLineDataset`, `TFRecordDataset`, `unbatch`, `window`.

### Master and worker services

This section discusses the design for the master and worker services. These
services are used by the Python API to provide distributed dataset processing,
and these services use the splitting API as a part of their implementation.

#### Master API

The master is responsible for registering datasets, generating and tracking
iteration and worker ids, and generating dataset splits for processing on
workers.

Below is a sketch of the Master API. This API is not public and is subject to
change.

```cpp
/// ---- Methods called by consumers ----

// Registers a dataset and returns an id for the dataset. If the dataset is
// already registered, its dataset id is returned.
int GetOrRegisterDataset(GraphDef dataset);

// Creates and returns `num_consumers` iterator ids which partition the
// specified dataset. This also creates an internal `iteration_id` used to
// track the overall dataset iteration. `num_tasks` defines how many tasks to
// create. If `num_tasks` is -1, it is up to the master to determine how many
// tasks to create.
list<int> CreateIterators(int dataset_id, int num_consumers,
                          int num_tasks);

// Returns the list of tasks processing data for `iterator_id`. Consumers query
// this to find which worker addresses to read data from.
list<TaskInfo> GetWorkersForIterator(int iterator_id);

///---- Methods called by input workers ----

// Registers a worker and returns its worker id.
int RegisterWorker(WorkerInfo worker_info);

// Requests the next splits to process on the given worker for the given
// iteration_id.
List<Split> GetSplits(int worker_id, int iteration_id);
```

#### Worker API

The worker is responsible for processing datasets and providing dataset elements
to consumers.

Below is a sketch of the Worker API. This API is not public and is subject to
change.

```cpp
/// ---- Methods called by consumers ----

// Gets the next element for the specified iterator_id.
list<Tensors> GetElement(iterator_id);

/// ---- Methods called by master ----

// Requests that the worker process the specified dataset. This will trigger the
// worker to start requesting splits from the master using the `iteration_id`.
void ProcessDataset(int dataset_id, int iteration_id, list<int> iterator_ids);
```

#### Visitation Guarantees

When iterating over a deterministic dataset, the tf.data service will process
all input data exactly once, even in the presence of master or worker failures.
We achieve exactly-once by having consumers keep track of their index within
each task, and having restored tasks skip elements to reach the requested index.
For the skipping to give exactly-once semantics, the dataset must produce
outputs deterministically.

If the dataset is not deterministic, the user can choose either at-least-once or
a close-to-exactly-once visitation guarantee. We can achieve
close-to-exactly-once by using the same skipping technique that we use to
achieve exactly-once for deterministic datasets. If users prefer an
at-least-once guarantee, we can instead start restored tasks from their latest
checkpoint.

In some cases, we can provide an exactly-once visitation guarantee to
non-deterministic pipelines. If input workers are brought down gracefully, they
can first write checkpoints of their tasks. This way, tasks can begin exactly
where they left off.

#### Determinism

Deterministic processing is a cornerstone of tf.data. Determinism is valuable
for debugging and experimentation. This section discusses how the tf.data
service will provide determinism.

To get deterministic behavior, the tf.data service will require three things:

1.  The dataset being distributed has deterministic output.
1.  The user sets `num_consumers`, `num_tasks`, and `deterministic=True` when
    calling `tf.data.experimental.service.create_iteration`.
1.  Each consumer uses a unique `consumer_index` when calling `make_iterator`.
1.  The consumers do not fail.

In the absence of failures, determinism is achieved by distributing splits
round-robin among `N` input workers and having input workers earmark every `ith`
element for consumer `i`.

To provide determinism even when servers fail, consumers can keep track of which
element index they have processed up to for each task. Input workers would
attach per-task element indices when they produce elements, so consumers can
ignore duplicate elements caused by worker restarts.

#### Failure Recovery

The tf.data service can recover from master and worker failures while preserving
determinism and its at-least-once visitation guarantee. The master achieves this
by writing its unrecoverable state to a persistent journal, and taking
checkpoints of its recoverable state to improve recovery time. When workers
reconnect to a restarted master, they update the master with their state so that
the master can recover its knowledge of its workers.

The unrecoverable state includes

*   **Registered datasets**
*   **ID generators** for iteration ids, iterator ids, dataset ids, and worker
    ids.
*   **In-progress iteration state**:
    *   **dataset id** for the iterated dataset so that we can recover the
        iteration's split generator
    *   **iteration id**
    *   **assignments from splits to tasks**, so that we can restart failed
        tasks on new workers.

Recoverable state includes

*   **Split generators**: Recoverable from our information about in-progress
    iterations.
*   **Worker addresses**: Recoverable when workers reconnect.
*   **Worker loads**: Recoverable when workers reconnect.
*   **Assignment from tasks to workers**: Recoverable when workers reconnect.

To improve recovery time, the master will periodically write checkpoints of its
split generators and outstanding splits, so that split generators don't need to
be run from the beginning during master recovery.

Workers have no unrecoverable state. If a worker crashes, a new worker can take
its place. It is up to the master to reassign splits from the crashed worker to
the new worker.

To improve worker recovery time, workers will periodically write checkpoints of
their iterators to directories named using their worker ids. When the restarted
worker connects, the master will tell it which iterator checkpoints to recover
from.

We will read and write this state through a MasterState interface which can be
implemented using various storage backends. For use cases that require fault
tolerance, the user must configure a fault-tolerant MasterState, e.g. Cloud
Spanner or etcd. If fault tolerance isn't required, the user could configure
state to be held in memory only.

#### Leadership Transfer

The master writes state to journal files so that the state can be recovered on
restart. It is possible that a new master could be brought up while the old
master is still running. If we aren't careful, this could result in corruption
of the journal as both masters try to write to it.

Ideally we could rely on a distributed coordination service such as ZooKeeper.
However, this would add a significant burden to users who don't have access to a
ZooKeeper cluster, and it would also require adding a new dependency on a
ZooKeeper client.

What TensorFlow does have is a FileSystem API. We will leverage this API to
perform leadership transfer by creating empty files and inspecting file
modification times.

```
files = list_directory(leadership_directory)
if all_files_older_than(files, leadership_transfer_interval):
  file = create_unique_file(leadership_directory);
  if file_is_strictly_newest(file, leadership_directory):
    become_leader()
# Another master may be leader. Wait for some time before trying again.
wait_random_interval()
```

The leader master will periodically write files to the leadership directory to
indicate that it is still leading.

The above scheme relies on the filesystem's create_file() and list() operations
being strongly consistent . Users may opt to use a filesystem that doesn't
support strong consistency, but they do so at the risk of two concurrently
running masters thinking they are leader. Common filesystems such as POSIX,
HDFS, and GCS support such strong consistency, but S3 does not.

#### Caveats

This section calls out caveats that users will need to be aware of when using
the tf.data service.

-   Due to the nature of dataset splitting, elements will not be processed in
    the same order as they were in the pre-distributed dataset. If a dataset
    relies on the order of the input files, the user's assumptions will be
    violated when splitting causes each input worker to process only a subset of
    the input files.
-   If a particular dataset operation doesn't support splitting, it must be
    moved after the part of the dataset which is distributed. Alternately, the
    user could set num_tasks=1 to avoid the need for splitting, but this will
    have a heavy performance cost since it only allows a single worker to
    generate dataset elements. The most commonly used but unsupported datasets
    are `from_generator` and `zip`.

#### Framework Integration

Many users interact with TensorFlow through a framework such as
[TFX](https://www.tensorflow.org/tfx). A framework can make leveraging the
tf.data service as simple as toggling a configuration boolean, triggering the
framework to bring up tf.data service servers and add a
`tf.data.experimental.service.distribute` transformation at the end of the
users' data pipeline. By inspecting the amount of time blocked on the input
pipeline, the framework could dynamically scale the number of input workers up
and down to find the minimum number of workers needed so that the input pipeline
can keep up with the model.

### Alternatives Considered

#### Use Beam for distributed dataset processing.

Beam is an open-source data processing framework capable of large-scale parallel
computation. Instead of implementing distributed computation ourselves, we could
execute Beam jobs to perform dataset processing.

We chose not to follow this direction to avoid creating a dependency on Beam.
Many users don't depend on Beam, and it would be a limitation to require that
dependency. If we depend on Beam, it will not be possible to use the tf.data
service with out-of-the-box TensorFlow. This is especially important as tf.data
service is expected to be used by the tf.distribute API.

### Performance Implications

With tf.data workers running in a separate cluster, we expect to be able to
horizontally scale until the input pipeline is no longer the bottleneck,
improving performance for input-bound pipelines.

If a pipeline input-bound or close to input-bound, tf.distribute could see
performance regressions when it uses the tf.data service to serve elements
across replicas. The issue is that the tf.data service will incur the cost of
transferring elements over the network to feed replicas, instead of having each
replica perform its input processing locally. On the other hand, if the input
pipeline is not the bottleneck, tf.distribute could see training speedups as
dynamic sharding mitigates the time spent waiting for stragglers.

### Dependencies

This proposal does not add any new dependencies to TensorFlow.

### Engineering Impact

The tf.data service will be maintained by the tf.data team.

### Platforms and Environments

The tf.data service is compatible with all platforms supported by TensorFlow.

### Best Practices, Tutorials and Examples

The tf.data performance guide will be updated to explain when to use the tf.data
service. We will also provide a tutorial for using the tf.data service.

### Compatibility

*   Does the design conform to the backwards & forwards compatibility
    requirements?
    -   Yes, this design only adds new functionality, so it doesn't break any
        backwards or forwards compatibility guarantees.
*   How will this proposal interact with other parts of the TensorFlow
    Ecosystem?
    -   How will it work with TFLite?
        *   We aren't planning any integration with TFLite, where we haven't
            seen a need for distributed input processing. Traditionally TFLite
            is used for inference, while tf.data is used for training.
    -   How will it work with distribution strategies?
        *   Distribution strategies will be able to leverage the tf.data service
            to replace its static sharding with dynamic sharding, and to support
            efficient splitting for a wider range of datasets.
    -   How will it interact with tf.function?
        *   The tf.data service APIs will work both inside and outside of
            tf.functions.
    -   Will this work on GPU/TPU?
        *   This proposal does not change the status quo of support for
            executing tf.data pipelines on GPU/TPU.

## Questions and Discussion Topics

*   How should we communicate that distributing a dataset will change the order
    in which elements are processed? If users' datasets rely on elements being
    processed in a certain order, they could face unpleasant surprises.
    -   Final decision: Address this through documentation.
*   Should we support splitting `skip`, `take`, and `scan` by having them
    operate at a per-task level (e.g. skip or take the first `N` elements within
    each task)?
    -   Final decision: Prohibit distributing these transformations, and tell
        users to instead use these transformations *after* applying the
        `distribute` transformation.
*   Is there a more user-friendly way to share iteration ids across consumers?
    Distribution strategy is well-equipped with collective ops to share the
    iteration ids, but sharing the iteration id could be a heavy burden for
    some users.
    -   Final decision: It is a reasonable expectation for users to either use
        distribution strategies, or distribute their own iteration ids.
        TensorFlow will soon have public APIs for collective operations that
        would make it easy to broadcast iteration ids.
*   Can `service.distribute` take a `ClusterResolver` so that the master
    hostname isn't baked into the dataset definition?
    -   Final decision: Accept `master_address_or_resolver`, and wait to resolve
        the master address until iteration begins. The `ClusterResolver` will be
        stored in the Python `Dataset` object. In the future, we may want C++
        implementations of `ClusterResolver` so that we can represent the
        resolver within the dataset graph.
