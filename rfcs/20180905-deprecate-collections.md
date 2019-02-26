# Deprecating collections 

| Status        | Accepted                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Rohan Jain (Google)                                  |
| **Sponsor**   | Asim Shankar (Google)                                |
| **Updated**   | 2018-09-05                                           |


## Motivation

TF 2.0 gives us a great opportunity to clean up API's that are not as desirable. The collections API, although it permits a compact coding style and is widely used, poses a fair amount of technical difficulties and challenges and so we'd like to get rid of it. The reasons for why we would want to do this are as follows



1.  **Eager execution:** There isn't a concept of a graph when executing eagerly and therefore no support for collections.

    Firstly, this has led to a fair amount of code been written as


    ```python
    if not context.executing_eagerly():
      ops.add_to_collections(...)
    ```

    And developers need to be careful that in the eager execution code path no collections stuff is handled.


    Furthermore, there are cases where collections have been used to keep graph based state (e.g. ensuring that only one variable for shared embedding columns is created). Such use cases need to be completely redesigned for eager.


    With the assumption that eager execution would become more dominant in 2.0, this places undue burden on developers, makes code more error prone and results in divergence between writing code that executes eagerly and code that builds and executes graphs.

2.  **Graph mode:** Collections are tied to a graph and assume that we build only one model per graph. As use cases get more and more complex, we have situations where we might build multiple models in a graph. In those scenarios, collections end up causing undesirable state updates e.g. when we intend to re-initialize variables for one model, it'll cause re-initing variables for all models in the graph.

    Functions cause further issues because functions are graphs of their own. Any collections API usage inside a function would create state that is purely local to that function and the default graph would lose all that state. The functions use case is incredibly important since they are used a lot for TPU's and moving forward, we plan to move more and more logic into defuns.



## **Collections use cases**

Here we bucket the various use cases for collections with code pointers and examples.



1.  **Variables**
Collections: `GLOBAL_VARIABLES, LOCAL_VARIABLES, MODEL_VARIABLES, TRAINABLE_VARIABLES, MOVING_AVERAGE_VARIABLES, CONCATENATED_VARIABLES`.
Use cases:
    1.  Asserting certain variables got created in testing code: [feature_column_test](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/feature_column/feature_column_test.py#L5890), [ops_test](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/framework/ops_test.py#L2151)
    2.  Finding a particular variable(s): [assign_from_values](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/framework/python/ops/variables.py#L534), [warm_start_util](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/warm_starting_util.py#L269), [base_layer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/layers/base.py#L226),[meta_graph](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/framework/meta_graph.py#L805), [show_and_tell_model](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/show_and_tell_model.py#L194), [gan](https://github.com/tensorflow/models/blob/master/research/gan/progressive_gan/train.py#L243)
    3.  Figuring out if new vars were created in a fn call: [template](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/template.py#L301)
    4.  Get a list of variables: [InMemoryEvaluatorHook](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/estimator/python/estimator/hooks.py#L152), [supervisor](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/supervisor.py#L470), [MovingAverageOptimizer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/opt/python/training/moving_average_optimizer.py#L141), [ExponentialMovingAverage](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/moving_averages.py#L385), [meta_graph_transformer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/meta_graph_transform/meta_graph_transform.py#L429), [beam](https://github.com/tensorflow/tensorflow/blob/r1.10/py/tensorflow_transform/beam/impl.py#L723)
    5.  Generically initializing all (uninitialized) vars [similar to iv]: [estimator](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/estimator/estimator.py#L1541), [keras_backend](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L658), [variables_initializer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/variables.py#L2166), [monitored_session](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/monitored_session.py#L198), [sync_replicas_optimizer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/sync_replicas_optimizer.py#L258)
    6.  Saving / Checkpointing: [estimator](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/learn/python/learn/estimators/estimator.py#L1419), [saved_model](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/saved_model/builder_impl.py#L278), [saver](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/saver.py#L1318)
    7.  Moving average variables track vars created by the [MovingAverageOptimizer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/opt/python/training/moving_average_optimizer.py#L141), [some Quantization vars](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/quantize/python/quant_ops.py#L50). Usually this collection is not included in the trainable_vars list, so clients use it to get a list of model_vars or vars_to_restore.
    8.  `CONCATENATED_VARIABLES` are only created by the [rnn_cell](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L60) implementation.

2.  **Initializers**
    1.  `TABLE_INITIALIZERS` contain `InitializeTableV2 / InitializeTableFromTextFileV2` ops ([lookup_ops](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/lookup_ops.py#L376)), but in some scenarios have been used to store iterator initializers ([dataset_builder](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/dataset_builder.py#L44))
        1.  Storing them helps run them in the beginning of a TF program.
        1.  Detecting whether tables etc. are created in a map etc. function ([dataset_ops](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/data/ops/dataset_ops.py#L2172))
    2.  `_SUMMARY_WRITER_INIT_COLLECTION_NAME`: A collection to hold the `CreateSummaryFileWriter / CreateSummaryDBWriter` ops that need to be run before a TF program runs.

3.  **Summaries**
    1.  SUMMARIES: Merges all the summaries into one giant tensor (SummaryMergeOp). Shouldn't be needed with SummaryV2.
    1.  SUMMARY_OP: A collection that just contains the MergeSummary op that merges all the summaries collected in the SUMMARIES collection above. Shouldn't be needed with SummaryV2.
    1.  [arbitrary_collections](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/summary_op_util.py#L29): The summary ops could be added to an arbitrary collection. This should go away with SummaryV2
    1.  `_SHOULD_RECORD_SUMMARIES_NAME`: A collection that contains a single boolean value turning summary recording on / off
    1.  `_SUMMARY_COLLECTION`: Stores all V2 summary names. Used by [control_flow_ops](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/control_flow_ops.py#L1893) to identify new summaries created before / after fn call.

4.  **Queue runners:** Collects up all the queue runners so that their threads can be started before most code execution. Most queues used for input pipeline which would go away with tf.data but some other use cases remain
    *   `SyncReplicasOptimizer` to synchronize variable updates.

5.  **Losses**
    1.  `REGULARIZATION_LOSSES`: Stashes away tensors that need to be added to the loss function that are used for regularization. During variable creation, the [regularizer fn is applied](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/variable_scope.py#L894) and the output tensor is stored in the collection ([get_variable](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/variable_scope.py#L907), [layers](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/layers/base.py#L137)). While the loss fn is being computed, the collection is retrieved and applied ([general loss computation code](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/losses/util.py#L58), [inception_train](https://github.com/tensorflow/models/blob/master/research/inception/inception/inception_train.py#L271), [object_detection](https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib.py#L284)).
    1.  LOSSES: In addition to accounting for regularization losses using collections, we also do the same trick for general loss functions. Whenever the loss function is computed, we add the loss tensor to this collection. Examples include [estimator code](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/estimator/estimator.py#L1391), [every loss computation method in losses_impl](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/losses/losses_impl.py#L166). [get_losses()](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/losses/util.py#L44) and [get_total_loss()](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/losses/util.py#L89) (deprecate) in losses/util.py then use this to provide the loss back.

6.  **Saving / Checkpointing**
    1.  SAVEABLE_OBJECTS: A collection of all non-variable objects that should be checkpointed. This should probably go away with object based checkpointing. [SavedModel](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/saved_model/loader_impl.py#L295) (?? allenl@)
    1.  SAVERS: A collection of Saver objects created during graph construction. Various parts of the code might create savers and this collection tracks them. Although a lot of usage code does "Get first from collection" ([supervisor example](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/supervisor.py#L469)). Should go away with object based checkpointing.

7.  **Conditionals:** COND_CONTEXT and WHILE_CONTEXT are collections to which stuff gets added to in [control_flow_ops.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/control_flow_ops.py#L2124). 

1.  **Assets:** The ASSET_FILEPATHS collection tracks all external files ([lookup_ops](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/lookup_ops.py#L550), [tf_transform](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/analyzers.py#L830), [estimator](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/estimator/estimator.py#L941)) that are needed by the model for instance vocabulary files etc. These are used in a few places
    1.  SavedModel: [builder_impl.py](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/saved_model/builder_impl.py#L354)
    1.  TF Transform:  [analyzers.py](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/analyzers.py#L98), [beam/impl.py](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/beam/impl.py#L560), [impl_helper.py](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/impl_helper.py#L405), [saved_io_transform.py](https://github.com/tensorflow/transform/blob/master/tensorflow_transform/saved/saved_transform_io.py#L213)
    1.  TF Hub:
        [saved_model_lib.py](https://github.com/tensorflow/hub/blob/master/tensorflow_hub/saved_model_lib.py#L278)

1.  **Resources:** A few contrib libraries don't keep track of resources to init / create them but instead just "[register](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/resources.py#L37)" them in these collections. Estimator etc. code makes sure that the [init ops registered are run](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/learn/python/learn/estimators/estimator.py#L1405) before hand. Some example contrib libraries are [BoostedTrees](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/boosted_trees/python/ops/model_ops.py#L112), [TensorForest](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/tensor_forest/python/ops/model_ops.py#L118)

1.  **MonitoredSession specific collections**
    1.  INIT_OP usually holds one initializer op - group(global_vars_initializer, shared_resources_initializer): [monitored_session](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/monitored_session.py#L179)
    1.  LOCAL_INIT_OP usually holds one initializer op - group(local_vars_initializer, tables_initializer, local_resources_initializer): [monitored_session](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/monitored_session.py#L271)
    1.  READY_OP is only used by the MonitoredSession code to hold one op to test whether the session is ready to run or not (usually just [reports uninitialized variables](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/monitored_session.py#L188))
    1.  READY_FOR_LOCAL_INIT_OP is also just used by MonitoredSession code, holds one op to check whether [all global vars are initialized](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/monitored_session.py#L197) and hence LOCAL_INIT_OP can run.
    1.  GLOBAL_STEP collects the variable tracking the global step counter. This is used in [training_util](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/training_util.py#L106), and some other [models](https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/show_and_tell_model.py#L347).

1.  **Updates:** (UPDATE_OPS) Layers such as BatchNormalization [create mean and variance update ops](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/layers/normalization.py#L418) that need to be run at training time. These are [thrown into an UPDATE_OPS collection](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/layers/base.py#L366) that are then run at training time ([estimator code](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/estimator/canned/head.py#L1546))

1.  **Iterators** (GLOBAL_ITERATORS) All [iterators are stored in this collection](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/data/ops/iterator_ops.py#L102) so that the [CheckpointInputPipelineHook](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/data/python/ops/iterator_ops.py#L182) that is responsible for checkpointing iterators can find them.

1.  **Features**
    1.  Shared embedding columns: In order to ensure that only one variable is created for a collection of shared embedding columns. This will go away with the new Feature Column implementation.
    1.  weight_collections can be passed to [input_layer](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/feature_column/feature_column.py#L219) and all the variables created by the InputLayer are added to these collections. But as cols_to_vars are returned, one can track the vars created.

1.  **Session bundle:**
    This is deprecated in lieu of SavedModel. Ignoring collections here.

1.  **SavedModel**

Collections: ASSETS_KEY, MAIN_OP_KEY, TRAIN_OP_KEY

Saved model needs to save what main op and train op to run when it is loaded back up. There are some other collections like INIT_OP etc. that are also needed by SavedModel.

1.  **Misc**
    1.  _XLA_SCOPE_KEY: A collection that basically acts like a global variable tracking the [experimental_jit_scope](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/compiler/jit.py#L39) usage and depth
    1.  iterator_ops: A collection just used in the [dataset serialization test code](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/data/python/kernel_tests/serialization/dataset_serialization_test_base.py#L611) to restore init and get_next ops for an iterator.
    1.  _[_VARSTORE_KEY](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/variable_scope.py#L1338): Very similar to _XLA_SCOPE_KEY 
    1.  [_VARSCOPESTORE_KEY](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/variable_scope.py#L1320): Similar to _XLA_SCOPE_KEY


## **Proposal overview**

At a very high level, we can bucket the usage of collections into roughly three categories

**Case 1:** (Most common) We create some objects during graph construction (e.g. variables, tables, iterators etc.) but we need to call some ops on them in a particular order during execution time (e.g. initialization). Keeping track of these objects is painful and tedious, so we just throw them into a collection and retrieve them when we want to.

**Case 2:** (Quite common) There is a special case of 1) when this happens across a save to disk (e.g. SavedModel) which we should point out separately (i.e. graph gets constructed, we save to disk and then we need to execute / run it). Collections is effectively a serialization format in this scenario.

**Case 3:** (Less common) We use collections as global objects that can hold state. Example usage would be making sure that we only create a shared embedding variable once for a collection of columns.

We'll handle each of the cases as follows:

**Case 1:** We propose that python code be responsible for tracking these objects themselves. This means that for most API's that do a `get_collection`, we'd have to explicitly pass in a list of objects. For the regular keras model or estimator code, we'll make sure that this works seamlessly but for more complex use cases beyond them, developers will have to track this on their own. For some specific objects such as variables and tables that are pervasive, we'll provide collectors that act like global python variables and can collect up these particular things.

**Case 2:** We'll provide some version of the collections API that can serialize and deserialize these groups of items. Again, the gathering of these items (which is 1) would be handled as we handle 1) but to retrieve them after writing to disk, we'll have some version of the collections API. This API would be purely local to the SavedModel implementation.

**Case 3:** This is a bit tricky and will have to be handled on a case by case basis. In some cases, we might do a redesign like we did for SharedEmbeddingColumns i.e. create explicit state managers that hold this state (which is the recommended way of dealing with this). Or in some cases, we might just create global variables and handle it that way. 


## **Details**

For each of the collections listed above, we list out how we'll handle their 'collection' in the regular Keras model writing case and the more general custom graph construction case.


<table>
  <tr>
   <td><strong>Use case </strong>
   </td>
   <td><strong>Collections affected</strong>
   </td>
   <td><strong>Regular case (Keras model)</strong>
   </td>
   <td><strong>More complex case (Custom graph)</strong>
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>GLOBAL_VARIABLES
   </td>
   <td>Model.variables
   </td>
   <td>variable_creator_scope
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>TRAINABLE_VARIABLES
   </td>
   <td>Model.trainable_variables
   </td>
   <td>variable_creator_scope
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>LOCAL_VARIABLES
   </td>
   <td>Metric.variables
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>MODEL_VARIABLES
   </td>
   <td>Not really used much
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>MOVING_AVERAGE_VARIABLES
   </td>
   <td>ExponentialMovingAverage.variables
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Variables
   </td>
   <td>CONCATENATED_VARIABLES
   </td>
   <td>Global variable
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Tables
   </td>
   <td>TABLE_INITIALIZERS
   </td>
   <td>Keras backend
   </td>
   <td>table_creator_scope
   </td>
  </tr>
  <tr>
   <td>Summaries
   </td>
   <td>_SHOULD_RECORD_SUMMARIES_NAME
   </td>
   <td>Global variable
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Summaries
   </td>
   <td>_SUMMARY_COLLECTION
   </td>
   <td>Not needed with CondV2
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Summaries
   </td>
   <td>_SUMMARY_WRITER_INIT_COLLECTION_NAME
   </td>
   <td>Global variable
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Summaries
   </td>
   <td>SUMMARIES
   </td>
   <td>Not needed with SummaryV2
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Summaries
   </td>
   <td>SUMMARY_OP
   </td>
   <td>Not needed with SummaryV2
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Queues
   </td>
   <td>QUEUE_RUNNERS
   </td>
   <td>Not needed with tf.data, Distribution strategies
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Losses
   </td>
   <td>REGULARIZATION_LOSSES
   </td>
   <td>Not needed after variables 2.0
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Losses
   </td>
   <td>LOSSES
   </td>
   <td>Model.losses
   </td>
   <td>Use Losses returned from functions in loss_impl.py
   </td>
  </tr>
  <tr>
   <td>Saving
   </td>
   <td>SAVEABLE_OBJECTS
   </td>
   <td>Not needed with object based checkpointing
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Saving
   </td>
   <td>SAVERS
   </td>
   <td>Not needed with object based checkpointing
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Conditionals
   </td>
   <td>COND_CONTEXT, WHILE_CONTEXT
   </td>
   <td>Not needed with CondV2
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Assets
   </td>
   <td>ASSET_FILEPATHS
   </td>
   <td>Keras backend
   </td>
   <td>table_creator_scope
   </td>
  </tr>
  <tr>
   <td>Resources
   </td>
   <td>RESOURCES, LOCAL_RESOURCES
   </td>
   <td>Not used much
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Updates
   </td>
   <td>UPDATE_OPS
   </td>
   <td>Model.updates
   </td>
   <td>BatchNormalization.updates
   </td>
  </tr>
  <tr>
   <td>Iterators
   </td>
   <td>GLOBAL_ITERATORS
   </td>
   <td>Object based checkpointing handles saveables generically
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>SavedModel
   </td>
   <td>INIT_OP
   </td>
   <td>Keep
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>SavedModel
   </td>
   <td>LOCAL_INIT_OP
   </td>
   <td>Keep
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>SavedModel
   </td>
   <td>READY_OP
   </td>
   <td>Keep
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>SavedModel
   </td>
   <td>READY_FOR_LOCAL_INIT_OP
   </td>
   <td>Keep
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>SavedModel
   </td>
   <td>GLOBAL_STEP
   </td>
   <td>Keep
   </td>
   <td>
   </td>
  </tr>
</table>


For some specific use cases, we'll elaborate how we will handle them


### **Global variables, trainable variables**

For the keras models use case, Model.variables already tracks all variables created in a Keras model. Similarly, for trainable variables, Model.trainable_variables tracks all trainable variables. 

In case of custom graph code, variable_creator_scope can be used to collect up variables.

Example code might look like

```python
class VariableTracker(object):
  def __init__(self):
    self.variables = []

  def variable_tracker(self, next_creator, **kwargs):
    v = next_creator(**kwargs)
    self.variables.append(v)
    return v

VariableTracker tracker
with tf.variable_creator_scope(tracker.variable_tracker):
  ...
  a = tf.Variable(0)
  ...

assert tracker.variables == [a]
```

We might want to expose VariableTracker as well.


### **Local variables**

Local variables are primarily used for metrics. With the move to objects for metrics, one can access the variables created for the metrics via the Metric object. 


### **Moving average variables**

Collections are used in two places here. We'll remove these calls and ask callers to explicitly pass in a variable list 



*   [MovingAverageOptimizer.swapping_saver](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/opt/python/training/moving_average_optimizer.py#L140)
*   [ExponentialMovingAverage.variables_to_restore](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/training/moving_averages.py#L473). We'll have to expose the variables created by ExponentialMovingAverage as they would have to be passed in here for this method.


### **Tables**


#### **Keras models**

We'll provide a `track_table` method in the backend (similar to `track_variable`) and we'll make sure that when model.fit is called, tables are initialized similar to the way we initialize variables. Changes we'll have to make here are the following



*   Add a `track_table` method similar to `track_variable` method in keras backend.py
*   Change feature column code to call `track_table` when it creates tables
*   Modify `get_session` to initialize the tracked tables in addition to variables


#### **Custom graphs**

Similar to `variable_creator_scope` we'll create a `table_creator_scope` that will allow tracking logic to be introduced each time a table is created. This can then be used to collect up all the tables. 

Sample user code might look like


```python
table_creator = tf.TableCreator()

with table_creator:
  … <graph creation code>
  table = tf.HashTable(tf.KeyValueTensorInitializer(keys, values), -1)
  … 

assert table_creator.tables() == [table]
```


We can add some helper methods on TableCreator to collect up all `initializers, vocabulary_filenames`.


#### **Alternative proposal for tables**

Initialize tables on first use. Similar to the [OneShotIterator](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/core/ops/dataset_ops.cc#L684), we'll pass in a function argument that can initialize the table. We can even explore the idea of the function to return a dataset and then initialize the table from that dataset. Having a function would avoid overheads of passing in large tensors that might be required to initialize on each lookup (index_table_from_tensor for instance might run into this problem). The Table API would remain the same since we hide initialization from the user and we wouldn't have to run an initializer explicitly. 

The concern with this solution is that in serving, the first request would take more time than the rest of the calls and that sort of inconsistent performance is not desirable. 


### **Asset file paths**

Asset files currently only include vocabulary files that are used to initialize tables with. The tracking of tables should be sufficient to track vocabulary files. 


### **Summaries**

Summary V1 uses collection extensively but the new Summary V2 implementation uses collections minimally. Those uses of collections all fall under Case 3 and are replaceable with global variables. Summary V1 deprecation is part of TF 2.0 plans as of now.


### **Iterators**

(`GLOBAL_ITERATORS` collection) Iterators are checkpointable (saveable) objects and we should handle them just as we handle other non-variable saveable objects. The challenge here is that iterators are objects that are created outside the model function and in fact by the [estimator code itself](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/estimator/estimator.py#L1035). As a result, in the regular object based checkpointing flow, no "user" object would be depending on the iterator. Hence, before passing the saveables to the Saver, we'll staple together the iterator created by the estimator with the saveables from the user created object. We might want to control this behaviour with a RunConfig flag somewhere perhaps.


### **Queue runners**

Queue runners are used by two use cases that should go away with TF 2.0

1.  **Input pipelines:** tf.data is the replacement and it doesn't use queues at all
1.  **Sync Replicas Optimizer:** Distribution strategies is the replacement


### **Conditionals**

The current Cond implementation uses collections but the new CondV2 implementation doesn't rely on it. Although moving to CondV2 is not part of TF 2.0 plans since its not a breaking change but it's quite likely that the timelines would line up. This effort (deprecating collections) would introduce a dependency on that effort.


### **Updates**

Batch normalization is the only use case for updates. For keras models, all updates are tracked via Model.updates. For the graph construction case, the updates are accessible via the updates attribute on the BatchNormalization layer itself and then they could be added to the train op. We will deprecate the [functional batch_normalization API](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/layers/normalization.py#L158) so that users deal with objects only.

A sample estimator model_fn might look like
```python
def model_fn(features, labels, mode, params, config):
  logits = ...
  batch_norm = tf.BatchNormalization(momentum=0.99)
  logits = batch_norm(logits)

  train_op = …
  train_op = tf.group(train_op, *batch_norm.updates)
  return tf.EstimatorSpec(..., train_op=train_op, ...)
```


### **Losses**

Regularization losses can be obtained from the list of global variables. So that collection can be removed. The generic "LOSSES" collections seems quite pointless right now and removable as each loss function in loss_impl.py provides the loss tensor.


## **Dependencies**

This effort has dependencies on the following efforts



1.  Queue runners removal and replacement with tf.data
1.  [Variables 2.0](https://github.com/tensorflow/community/pull/11)
1.  Object based checkpointing
1.  Summary V1 deprecation and replacement with Summary V2
1.  [Cond V2](https://github.com/tensorflow/community/blob/master/rfcs/20180507-cond-v2.md)
1.  New OO metrics implementation


## **Implementation plan**

Tentatively, we have the following plan for deprecating collections



1.  Discourage further graph collections usage by slapping TF 2.0 deprecation notices on
    1.  tf.get_collection, tf.get_collection_ref
    1.  tf.add_to_collection
    1.  tf.layers.batch_normalization
1.  Get rid of Case 3) collections usage. Example tasks here
    1.  Summaries V2 to use global variables instead of collections
    1.  Moving Average Optimizer changes to not use collections
    1.  Remove losses collection usage
    1.  If object based checkpointing comes to Estimator, remove GLOBAL_ITERATORS collection
    1.  Get rid of concatenable variables collection.
1.  Add tracking for things such as tables in Keras backend. At the end of this step, Keras models shouldn't be dependent on collections.
1.  Make estimators code (except for saving / loading) collection free. All canned estimators might be a good starting point. (Removing collections logic from scaffold might be bulk of the work here)
1.  Make "collection" of stuff while saving collections free. 3), 4) should mostly take care of it. After this, collections should be purely being used for serialization purposes.
1.  Create collectors for tables. Potentially create a default one to be used with variables.


## **Open questions / TODOs**



1.  Figure out a concrete plan for removing CONCATENABLE_VARIABLES collection.

# Design review notes
* Going through dependencies of this effort.
    * Object based checkpointing slated for 2.0. 
    * V2 metrics are in 2.0 (so old metrics deprecated).
    * Functional control flow slated for 2.0
* Question: What is special about tables?
    * Answer: Tables are immutable, they have to be initialized. We currently handle variables specially in defun, we could generalize that to tables, but haven’t looked into that yet. Tables are not variables today because they have non-Tensor representations (e.g., a std::hash_map)
* Question: Can immutable tables be variables with a variant in them?
    * Answer: This is a worthy direction to explore but we might not have enough time to do this.
* Question: How do we create the “init_op” for serving? Do we need to track tables in tf.Checkpointable like we do variables? 
    * Answer: Making tables be variables with variants does this automatically. Serialized format (for SavedModel) doesn’t need to change.
* Question: How do we track all the tables?
    * Answer: These tables are created by feature columns, which are used to create a Layer object, that Layer object can track all the tables/assets of interest. Make tables Checkpointable, and use the infrastructure for tracking checkpointables. Or a parallel mechanism? For V1, have a table_creator_scope() that can track all created tables
* Other collections
    * UPDATE_OPS: defun the layer and it is no longer a problem? This may be problematic for parameter server training if we have to wait for the updates before next iteration. Can be addressed by playing with strictness of function execution.
