# tf.data Snapshot

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [193](https://github.com/tensorflow/community/pull/193) |
| **Author(s)** | Frank Chen (frankchn@google.com), Rohan Jain            |
|               | (rohanj@google.com)                                     |
| **Sponsor**   | Jiri Simsa (jsimsa@google.com)                          |
| **Updated**   | 2020-01-07                                              |

## Objective

With ever faster accelerators available in Cloud and hyperparameter tuning
consuming larger chunks of accelerator time, TensorFlow users are increasingly
finding that they don’t have enough CPU resources to keep up with these
accelerators, leaving valuable accelerator resources idle.

To alleviate this problem, we are proposing a `snapshot` API within `tf.data`,
to allow users to transparently persist the output of their preprocessing
pipeline to disk, and materialize the pre-processed data on a different training
run.

This API enables repeated preprocessing steps to be consolidated, and allowing
re-use of already processed data, trading off disk storage and network bandwidth
for freeing up more valuable CPU resources and accelerator compute time.

## Motivation

Large TensorFlow users have indicated that they have complicated input
processing pipelines which saturate their CPUs before saturating their
accelerators (TPUs in particular). Since they often experiment with
hyperparameter tuning or tweaks to existing model without affecting their input
pipeline, they are asking for ways to avoid similar repeated preprocessing of
data by either saving a dataset or caching it to disk.

## User Benefit

Users will be able to transparently persist partially or fully processed data
from `tf.data` input pipelines to disk or Cloud storage systems, and materialize
the pre-processed data during subsequent runs from the same pipeline. This will
cut down on the input pipeline processing overheads during second and subsequent
runs.

## Design Proposal

We propose that we add a new `snapshot` transformation to tf.data. To illustrate
the usage of the transformation, we can start with some sample code:

```python
dataset = Dataset.list_files("/raw/data/*").shard(num_workers, i)
dataset = dataset.parallel_interleave(TFRecordDataset)
dataset = dataset.map(my_preprocessing_fn)
dataset = dataset.apply(tf.data.snapshot("/saved/data", options...))
dataset = dataset.repeat()

model = ...
model.fit(dataset)
```

As we can see, the end user simply has to add this transformation in order to
use this functionality. In essence, the transformation is similar to the
existing `tf.data.Dataset.cache`, with the key difference is being that, unlike
`cache`, `snapshot` is intended to re-used across different executions of the
same input pipelines.

### Proposed API

We are proposing the following API for the snapshot transformation.

```python
def snapshot(path,
             compression=None,
             reader_fn=None,
             writer_fn=None,
             pending_snapshot_expiry_seconds=None):
  pass  # Implementation goes here.
```

1.  `path`: Required. A directory where we want to save our snapshots and/or
    read from a previously saved snapshot.

1.  `compression`: Optional. The type of compression to apply to the snapshot
    written to disk. This will support `GZIP`, `SNAPPY` or None. Defaults to
    AUTO.

1.  `reader_fn`: Optional. The input pipeline transformation specified by 
    `reader_fn` is executed when the snapshot detects that there is an existing, 
    valid snapshot available.

    `reader_fn` is a user specified function that accepts a single argument: 
    (1) a Dataset of Datasets, each representing a "splits" of elements of the 
    original dataset. The cardinality of the input dataset matches the 
    cardinality of the output of `writer_fn` (see below). The function should 
    return a Dataset of elements of the original dataset.

    A default `reader_fn` will look like the following:

    ```python
    def default_reader_fn(datasets):
      # shuffle the datasets splits
      datasets = datasets.shuffle(NUM_DATASETS)
      # read datasets in parallel and interleave their elements
      return dataset.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)
    ```

1.  `writer_fn`: Optional. The input pipeline specified by `writer_fn` is 
    executed when the snapshot op detects that there are no valid snapshots
    and no other threads are currently attempting to write a snapshot.

    `writer_fn` is a user specified function that accepts a single argument: 
    (1) a Dataset of elements to be written out. The function should return
    a Dataset of Datasets, each representing "splits" of elements of the
    original dataset. The tf.data snapshot implementation will then persist
    splits in parallel.

    A default writer_fn will look like the following:

    ```python
    def default_writer_fn(dataset):
      # add a component with element index
      dataset = dataset.enumerate()
      # split input dataset in a round-robin fashion
      return dataset.split(num_splits=NUM_CORES, key_fn=lambda i, _: i % NUM_CORE
    ```

1.  `pending_snapshot_expiry_seconds`: Optional. How long to wait (in seconds)
    before the snapshot op considers a previously unfinished snapshot to be
    stale and starts writing a snapshot from scratch again. Defaults to 86400
    seconds (1 day).

#### Achieving Parallelism

`reader_fn` and `writer_fn` will default to passing the dataset through unchanged
by default. In other words, the default implementation will result in 
single-threaded reads and writes on snapshots. Parallelism can be achieved in
`writer_fn` by splitting up the dataset into multiple datasets, and using
`num_parallel_calls` in the `interleave` function of the `reader_fn`.

#### Computing Graph Fingerprints

Snapshot attempts to determine whether a run of an input pipeline is the same
as a previous run by computing the fingerprint of the nodes within the pipeline.

However, some input pipelines might vary in insignificant ways from run to run
that causes the fingerprinting of them to differ. For instance, consider the
following preprocessing function:

```python
features_to_multiply = {"feature1", "feature2", "feature3", "feature4"}

def preprocessing_fn(value):
  keys_to_features = {
    "feature1": tf.FixedLenFeature([], tf.float32, 0.0),
    "feature2": tf.FixedLenFeature([], tf.float32, 0.0),
    "feature3": tf.FixedLenFeature([], tf.float32, 0.0),
    "feature4": tf.FixedLenFeature([], tf.float32, 0.0)
  }

  parsed = tf.parse_single_example(value, keys_to_features)
  combined_feature = 1.0
  for item in features_to_multiply:
    combined_feature *= parsed[item]

  return combined_feature

dataset = ...
dataset = dataset.map(preprocessing_fn)
```

In the above example, our `features_to_multiply` variable uses a `set`, which is 
not guaranteed to be ordered in Python. When we iterate over the set in the 
for loop within `preprocessing_fn`, we may get a different graph on each 
run (i.e. one run could have us multiplying `feature2` first, then `feature4`, 
etc..., while another run may have us multiplying `feature1`, then `feature3`, 
and so on).

In cases like these, we can ask fingerprinting to use a fixed value for the
fingerprint of the map function with a new `with_snapshot_fingerprint`
transformation, which asks the fingerprinting function to not compute the 
fingerprint of the previous node but to use a user-specified value instead:

```python
dataset = ...
dataset = dataset.map(preprocessing_fn) 
dataset = tf.data.experimental.with_snapshot_fingerprint(
    dataset, fingerprint="my_fixed_fp")
```

### External API Guarantees

Externally, we guarantee that snapshots written by a particular version of
TensorFlow will be readable by that specific version of TensorFlow.

We are not currently handling the case where workers do not go through the
entire training set at least once.

### Alternatives Considered

An alternative proposal for an API would be `save()` and `load()`, where the
saving and loading of the input pipeline would be made more explicit, avoiding
some of the logic needed in determining whether to snapshot or read from a
snapshot of a model.

The downside here would be that the user would have to split the preprocessing
and training into potentially different files, and users would be forced to
select whether to train or preprocess on their own, which is not good.

### Performance Implications

Benchmarks for this feature will be included as part of Dataset microbenchmarks.

### Dependencies

No new dependencies will be introduced as part of this project to TensorFlow.
Dependent projects may be able to use this additional op, but there should be no
significant changes otherwise.

### Engineering Impact

Binary sizes increases slightly with the inclusion of this new op, and this code
will be maintained by the `tf.data` team.

### Platforms and Environments

This op will work on all TensorFlow-supported platforms. We do not anticipate
this to work on embedded systems as it is not useful in resource-constrained
environments.

### Best Practices, Tutorials and Examples

A user guide for snapshot will be published to guide new users in using this
feature.

### Compatibility

This introduces a new op, which will impact future backwards compatibility.

### User Impact

A new python function and a new op are the only user-facing changes visible.

## Detailed Design

### Implementation Assumptions

The following implementation is based on the following assumptions that define
the MVP this is designed for:

1.  We assume that at least for one pipeline run, you can go through the entire
    training dataset and be able to store that data on disk. Otherwise, a
    snapshot will never get created.

2.  In the cases where there are multiple workers and the dataset is sharded with
    `Dataset.shard`, we assume that the number of workers remains the same from 
    the initial (writing) run through to the reading runs.

    If the number of workers change, then the `num_shards` parameter to
    `Dataset.shard` will change, and this will result in a different graph
    fingerprint and another snapshot write will be triggered.

    If all workers use the exact same input pipeline with no sharding (e.g. all
    workers will read from all the files), then snapshot will still be able to
    read from previous snapshots even if the number of workers is different.

3.  Any `repeat`s in the dataset should be moved to after the `snapshot` op, to
    avoid writing large (or infinite) amounts of data during a snapshot writing
    run.

### New `SnapshotDatasetOp`

To implement the transformation, we are introducing a new `SnapshotDatasetOp`
dataset kernel that will implement all of the functionality in TensorFlow C++.
Python code is mostly glue code to pass relevant parameters into the op kernel.

### Internal Directory / File Structure

Given a user directory path (e.g. `/path/to/snapshot`), the directory will look
like:

*   /path/to/snapshot
    *   `fingerprint`/
        *   snapshot.metadata
        *   `run-id`/
            *   0000000.snapshot
            *   0000001.snapshot

The `fingerprint` is a hash of the input processing graph. The `run-id` is
unique training run ID generated.

### Standard Kernel Workflow

_Note: This is an implementation detail, and may change in the future. This
should not be relied upon except as a reference to the current implementation._

By default, the `snapshot` operation will, upon startup, make a determination
using the following algorithm as to whether the operation should be in the
WRITE, PASSTHROUGH, or READ state.

1.  We will compute a graph fingerprint containing all the information from the
    Dataset preprocessing graph before the `snapshot` op. We’ll use the
    `AsGraphDefInternal` method on DatasetBase for this.

1.  We will attempt to enter the corresponding fingerprint directory. For
    instance, if the computed fingerprint is `f-abc123` and the base snapshot
    directory is `/saved/data`, then we will attempt to enter
    `/saved/data/f-abc123`.

1.  If the snapshot directory is non-existent, empty or it doesn’t contain a
    `metadata` file, we will enter the **WRITE** state.

1.  If the snapshot directory contains a `metadata.final` file, we will read
    the final metadata file and proceed to the **READ** state.

    1.  The file contains the following fields:
        1.  A training run ID,
        1.  A boolean indicating if the snapshot is complete.
        1.  A training run start-time.

1.  If the snapshot directory contains a `metadata` file but not a 
    `metadata.final` file, we will read the metadata file.

1.  If the training run start-time is more than the (configurable) training run
    timeout (set with the `pending_snapshot_expiry_seconds` parameter), we will
    enter the **WRITE** state.

1.  If the training run start-time is less than the training run timeout, but
    the snapshot is not complete, then we will enter the **PASSTHROUGH** state.

1.  If the snapshot is complete, we will enter the **READ** state.

#### WRITE State

1.  We generate a random training run ID.

1.  We write (possibly overwriting) the `snapshot.metadata` file.

1.  We proceed to create a subdirectory containing the training run ID, and
    start writing data asynchronously in chunks.

1.  At the end of the dataset (when `end_of_sequence == true`), we will check
    the snapshot.metadata file to determine whether it contains the same
    training run ID.

    1.  If it does, we write a `metadata.final` file containing the 
        same information as the `metadata` file but with the complete
        bit set to true.
    1.  If it does not, it means that someone else is concurrently writing the
        snapshot and we lost the race to them. We delete all data in the
        training run directory.

For the current implementation, we will store the data in chunked TFRecord
files. Eventually we may move to other more higher performance data stores or
support additional storage systems such as Cloud BigTable.

#### PASSTHROUGH State

1.  This is a no-op, where we simply pass through the tensors to the downstream
    operations.

#### READ State

1.  We will read from the snapshots contained within the subfolder with the
    correct graph fingerprint and specified training run ID.

1.  Optionally, the user may choose to tell us to specify that the snapshots
    should be read back in shuffled order.

### Concurrency: Handling Multiple Input Workers

If input workers are sharded, then they will generate different graph
fingerprints as their shard indexes will be different. This will result in each
worker writing to a different subdirectory.

If input workers are not sharded, then this will result in a race and
potentially multiple workers writing data (still with different training run
IDs). Eventually, if each worker finishes, we will be left with one copy of the
data as all the other workers will determine that they have lost the race and
delete their own copy of the snapshot data.

## Questions and Discussion Topics

*   Should we implement this as three ops (a control opt o determine whether a
    snapshot is to be read from/written to) and a write and read op to do the
    respective operations?
    *   Pros include:
        *   Modularizes the implementation into smaller chunks
        *   Allows someone else to do the "control"
    *   Challenges include:
        *   Where/how the "control" runs?
        *   How do we construct the dataset graph properly?
*   How should autotuning be integrated into the snapshot transformation?
*   Are the configuration options well named? Is it possible to consolidate some
    of these options?
*   What other compression/decompression options would you like to see
    supported?
*   Any other performance / feature tuning knobs we should make available?
