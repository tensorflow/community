# Keras pre-made models

| Status        | Accepted                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com)|
| **Sponsor**   | Francois Chollet (fchollet@google.com), Alexandre Passos (apassos@google.com) |
| **Updated**   | 2019-06-10                                           |

## Objective

This document proposes several pre-made Keras models that would allow users to:
* build basic machine learning models easily
* compose them with other keras Layers
* replace [Canned Estimators](https://www.tensorflow.org/guide/premade_estimators) in TF 2.0.

## Motivation

Scikit-learn is the de-facto framework for creating basic machine learning models. From perspective of TensorFlow,
it would be useful to provide users with several key pre-made models to support the following features:
* Distributed mini-batch training
* Accelerated training (GPU & TPU)
* Enable AutoML to select different pre-made models
* Enable users to freely compose pre-made models with other Keras layers and models.

Currently, most pre-made models from TensorFlow have been exposed through [CannedEstimator](https://www.tensorflow.org/guide/premade_estimators).
However, the Estimator API:
* Relies on continuous graph rebuilding and checkpoints reloading which slows down training
* Relies on global collections and isn't TF 2.0 friendly
* Makes many advanced features such as meta/transfer learning difficult
* Enforces user to create input functions when not necessary

Given these considerations, we propose the following pre-made models as a starting point:
1. Linear models including Linear Regression and Logistic Regression
2. Wide-Deep models including Wide-Deep Regression and Wide-Deep Classification
3. BoostedTree models including BoostedTree Regression and BoostedTree Classification

Users can use [feature_columns](https://www.tensorflow.org/guide/feature_columns) with pre-made models for feature engineering.

Several side notes: This document should serve as a foundation for other pre-made models that are not included in this proposal, such as Kmeans, random forest, collaborate filtering, which are yet to come; Detailed design regarding implementation details for BoostedTree, such as distributed training, saving & restoring, acceleration, supported feature columns should live in separate docs. 

## Design Proposal

### Challenges

Many premade models, including BoostedTrees, KMeans (as well as WideDeep and many others more in the future) are highly complicated and do not follow the simple foward & backprop workflow, an assumption heavily relied on by Keras. Building such models, while not compromising the basic composability of Keras (i.e., compose layers on top of layers), is the main challenge we need to resolve in this document. Another challenge is that these models can have multiple training phases (such as collaborative filtering models). We propose the following two approaches to address them:

### Proposal 1: Customized training function & composability

We propose to let each subclassed pre-made model override the training function. It is optional to provide a special subclass `PremadeModel` if other methods such as `compile` and `fit` needs to be overriden as well. In traditional Keras models, such training function is dominated by autodiff - given the forward pass of the model, the gradients for each Operation is generated and backprop computation is automatically laid out for the entire model. However such assumption is only valid for neural network based supervised learning architecture. For many other scenarios, we would need to break this assumption:

1. gradients may not be used, e.g., any un-supervised learning tasks
2. gradients may only be used partially, e.g., tree based models
3. both gradients and higher order statistics are used for training, e.g., linear model with BGFS algorithm that approximates hessian
4. only gradients are used, but the backprop path is composed of customized training path, e.g., wide-deep model
5. primal-dual formulation, e.g., SDCA linear model, and constrained optimization algorithms in general.

The pre-made models also inherit from tf.keras Layer so it should allow user to freely compose them with other tf.keras.layers. Linear, DNN, Wide-Deep models are differentiable and can easily be composed with other layers. Trees and KMeans are usually not differentiable (though some [research work](https://arxiv.org/pdf/1711.09784.pdf) has been done to approximate the backprop), but it is desired that they can still be composed with any other layer inside any keras models (sequential, functional or subclassed) and enable many other research ideas. Take an example: a Keras sequential model where a BoostedTree  layer is stacked between a DNN model and a dense layer for distillation of DNN, i.e., feeding the learned embeddings from DNNs to boosted trees , i.e., dnn -> boosted_trees -> dense. During backprop, the gradients should be stopped at boosted_trees layer and not propagate through dnn. By default, we let backprop of boosted_tree layer return zero gradients by [tf.no_gradient](https://www.tensorflow.org/api_docs/python/tf/no_gradient). As **implementation detail, we would also need to allow user to provide customized backpropagation**, which shall be discussed separately in the detailed design doc of BoostedTrees and KMeans.

### Proposal 2: Phased training based on AutoGraph

Another major challenge for many basic machine learning models is that many models have several training phases. In a majority of use cases these training phases are executed in sequential, for example 1) KMeans has an initialization phase to select the initial clusters and train phase to update the clusters, 2) BoostedTrees has an quantization phase (that converts float to int) and tree growing phase. In other cases, training phases are executed alternatively until convergence criteria is reached, for example collaborative filtering model alternates between row-solving phase and col-solving phase. 
Our proposal is to split each phase into separate sub-training phases, and relies on [AutoGraph](https://www.tensorflow.org/guide/autograph) for controlling the execution ordering of phases.

Combining the above proposals, below is an example (pseudo) code snippet for BoostedTrees:

```python
BoostedTreesClassifier(Model):

def __init__(self, n_batches_per_update, num_classes=2, num_trees=10, learning_rate=0.1,
             l1=0., l2=0., max_depth=None, min_samples_split=2, min_samples_leaf=1)
  self._quantile_accumulator = QuantileAccumulator()
  self._stats_accumulator = StatsAccumulator()
  self._tree_ensemble = TreeEnsemble(num_classes, num_trees, max_depth)
  self._are_quantiles_ready = tf.Variable(False, dtype=tf.bool)
  self._center_bias_ready = tf.Variable(False, dtype=tf.bool)
  self._cond_accumulator = tf.ConditionalAccumulator(dtype=tf.int32)
  // Implementation details.

def _make_train_function(self):
  updates = self._train_phase(self._feed_inputs, self._feed_outputs)
  inputs = self._feed_inputs + self._feed_outputs + self._feed_sample_weights
  outputs = [self.total_loss] + metrics_tensors
  self.train_function = tf.keras.backend.function(inputs, outputs, updates)

@autograph.convert()
def _train_phase(self, inputs, outputs):
  if (self._are_quantiles_ready):
    return self._grow_tree_phase(inputs, targets)
  else:
    return self._quantile_phase(inputs, targets)

@autograph.convert()
def _grow_tree_phase(self, inputs, targets):
  logits = self._tree_ensemble.training_predict(inputs)
  loss = loss_fn(logits, output)
  gradients = tf.gradients(loss, targets)
  hessians = tf.gradients(gradients, targets)
  stats_summary = self._stats_accumulator.add_summaries(gradients, hessians)
  while (self._center_bias_ready is False):
    self._center_bias_phase()
  return self._tree_ensemble.grow_tree(stats_summary)
  
@autograph.convert()
def _quantile_phase(self, inputs, targets):
  del targets
  self._quantile_accumulator.add_summaries(inputs)
  self._cond_accumulator.apply_grad(1)
  if (self._cond_accumulator.num_accumulated() > 5):
    self._quantile_accumulator.flush()
    return self._are_quantiles_ready.assign(True)
  else:
    return no_op

@autograph.convert()
def _center_bias_phase(self):
  // Implementation details.
```
Note this is a single machine version of implementation. For distributed training scenarios it will be more complicated and discussed in separate documents.

## Detailed Design

This section describes the API signatures of the premade models that we propose.

### SparseFeatures Layer
The SparseFeatures layer will be responsible for preprocessing categorical feature columns and serve as input to Premade models. Just as DenseFeatures (which is undergoing implementation changes), the interface is:
```python
`tf.keras.layers.SparseFeatures`
SparseFeatures(Layer):

  def __init__(self, 
               feature_columns=None, 
               sparse_combiner="sum", 
               name=None, 
               **kwargs):
    """tf.keras.layers.SparseFeatures

    Args:
      feature_columns: An iterable containing all the feature columns used by the model. 
        All items in the set should be instances of classes derived from `FeatureColumn`.
      sparse_combiner: A string spcifying how to reduce if a categorical column is multivalent. "mean", "sqrtn", and "sum" are supported.
      name: Name of the layer
    """
    pass

  def call(self, features):
    """
    Args:
      features: Mapping from keys to tensors. `FeatureColumn`s look up via
        these keys. For example `numeric_column('price')` will look at 'price'
        key in this dict. Values can be a `SparseTensor` or a `Tensor` depends
        on corresponding `FeatureColumn`.
    """
    pass

def get_config(self): serialize the configuration of feature columns.
def from_config(self): deserialize the feature columns from config.

```

### Linear Models
These linear models support mini-batch training. For large models, model parallelism needs to be performed via sharding the model properly, see [partition strategy in 2.0](https://github.com/tensorflow/community/blob/master/rfcs/20190116-embedding-partitioned-variable.md).

#### Linear Regression
For linear regression models, mean squared error loss is used by default. Metrics can be customized.
```python
LinearRegressor(Model):

  def __init__(self,
               output_dim=1,
               use_bias=True,
               kernel_regularizer=None, 
               bias_regularizer=None, 
               *args, **kwargs):
    """tf.keras.premade.LinearRegressor

    Args:
      output_dim: Dimensionality of the output vector (scalar regression is the default)
      use_bias: whether to calculate the bias/intercept for this model.
          If set to False, no bias/intercept will be used in calculations, e.g., the data is already centered.
      kernel_regularizer: Regularizer instance to use on kernel matrix.
      bias_regularizer: Regularizer instance to use on bias vector.
    """
    pass

  def compile(self, optimizer="rmsprop", loss="mse", metrics=None):
    pass
```
#### Logistic Regression
For logistic regression models, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
LinearClassifier(Model):

  def __init__(self,
               num_classes=2, 
               use_bias=True, 
               kernel_regularizer=None,
               bias_regularizer=None,
               *args, **kwargs):
    """tf.keras.premade.LinearClassifier

    Args:
      num_classes: Number of output classes (>=2).
      use_bias: whether to calculate the bias/intercept for this model.
        If set to False, no bias/intercept will be used in calculations, e.g., the data is already centered.
      kernel_regularizer: Regularizer instance to use on kernel matrix.
      bias_regularizer: Regularizer instance to use on bias vector.
    """
    pass

  def compile(self, optimizer="sgd", loss="categorical_crossentropy", metrics=None):
    pass

*Only binary classification is supported for sdca.
```

### DNN Models
The model is not exposed as public tf endpoints, but simply implementation that users can import.

#### DNN Regression
For dnn regression models, mean squared error loss is used by default. Metrics can be customized.
```python
DNNRegressor(Model):

  def __init__(self, 
               hidden_units,
               output_dim=1,
               activation="relu", 
               dropout=None, 
               batch_norm=False,
               *args, **kwargs):
  """
    Args:
      hidden_units: List of hidden units.
      output_dim: Dimensionality of output vector, scalar by default.
      activation: Activation for each dense layer.
      batch_norm: Bool to indicate whether add batch norm after each activation layer.
      dropout: dropout layer to add after batch norm layer.
  """

  def compile(self, optimizer="sgd", loss="mse", metrics=None)
    pass
```
#### DNN Classification
For dnn classification models, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
DNNClassifier(Model):

  def __init__(self, 
               hidden_units, 
               num_classes=2, 
               activation="relu", 
               dropout=None, 
               batch_norm=False, 
               *args, **kwargs):
  """
    Args:
      hidden_units: List of hidden units.
      num_classes: Number of classes.
      activation: Activation for each dense layer.
      batch_norm: Bool to indicate whether add batch norm after each activation layer.
      dropout: dropout layer to add after batch norm layer.
  """

  def compile(self, optimizer="sgd", loss="categorical_crossentropy", metrics=None)
    pass
```

### Wide and Deep Models
#### Wide-Deep Regression

Mean squared error loss is used by default. The FTRL optimizer is used by default for the linear branch, and the Adagrad optimizer is used by default for the deep branch. Metrics can be customized.
```python
DNNLinearRegressor(Model):

  def __init__(self, linear_model, dnn_model, *args, **kwargs):
    """`tf.keras.premade.DNNLinearRegressor`
    Args:
      linear_model: Uncompiled instance of a Sequential model starting with a `SparseFeatures` layer and followed by `LinearRegressor`.
      dnn_model: Uncompiled instance of a Sequential model starting with a `DenseFeatures` layer, its outputs must match the output of the linear model.
    """
    pass

  def compile(self, linear_optimizer="ftrl", dnn_optimizer="adagrad", loss="mse", 
              metrics=None)
    pass
```
#### Wide-Deep Classification
```python
DNNLinearClassifier(DNNLinearModel):

  def __init__(self, linear_model, dnn_model, *args, **kwargs):
    """`tf.keras.premade.DNNLinearClassifier`
    Args:
      linear_model: Uncompiled instance of a Sequential model starting with a `SparseFeatures` layer and followed by `LinearRegressor`.
      dnn_model: Uncompiled instance of a Sequential model starting with a `DenseFeatures` layer, its outputs must match the output of the linear model.
    """
    pass

  def compile(self, linear_optimizer="ftrl", dnn_optimizer="adagrad", loss="categorical_crossentropy", 
              metrics=None)
    pass
```

### BoostedTrees
#### BoostedTrees Regression
For GDBT regressor, mean squared error loss is used by default. Metrics can be customized.
```python
BoostedTreesRegressor(Model):

  def __init__(self,
               n_batches_per_update, 
               output_dim=1, 
               num_trees=10, 
               learning_rate=0.1, 
               regularizer=None,
               min_node_weight=0., 
               max_depth=None, 
               min_samples_split=2, 
               min_samples_leaf=1):
    """tf.keras.premade.BoostedTreesRegressor.

    Args:
      n_batches_per_update: the number of batches to run per tree update.
      output_dim: Dimensionality of the output vector (scalar regression is the default).
      num_trees: The number of boosted trees to perform.
      learning_rate: Shrinkage parameter to be used when a tree added to the model.
      regularizer: Regularizer instance.
      min_node_weight: Minimum sum hessian per leaf.
      max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
      min_samples_split: The minimum number of samples required to split an internal nodel.
      min_samples_leaf: The minimum number of samples required to be at a leaf node.
    """
    pass

  def compile(self, loss=None, loss="mse", metrics=None)
    pass
```

#### BoostedTrees Classification

For GDBT classifier, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
BoostedTreesRegressor(Model):

  def __init__(self,
               n_batches_per_update, 
               num_classes=1, 
               num_trees=10, 
               learning_rate=0.1, 
               regularizer=None,
               min_node_weight=0., 
               max_depth=None, 
               min_samples_split=2, 
               min_samples_leaf=1):
    """tf.keras.premade.BoostedTreesRegressor.

    Args:
      n_batches_per_update: the number of batches to run per tree update.
      num_classes: Number of classes.
      num_trees: The number of boosted trees to perform.
      learning_rate: Shrinkage parameter to be used when a tree added to the model.
      regularizer: Regularizer instance.
      min_node_weight: Minimum sum hessian per leaf.
      max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
      min_samples_split: The minimum number of samples required to split an internal nodel.
      min_samples_leaf: The minimum number of samples required to be at a leaf node.
    """
    pass

  def compile(self, loss=None, loss="categorical_crossentropy", metrics=None)
    pass
```

## Using Premade models with Keras Feature layers and Processing layers
```python
// usage for feature columns.
age = tf.feature_column.numeric_column('age')
occupation = tf.feature_column.categorical_column_with_vocab_list('occupation', ['doctor', 'teacher', 'engineer'])
bucketized_age = tf.feature_column.bucketized_column(age, num_buckets=5)
feature_columns = [bucketized_age, occupation]
sparse_feature_layer = tf.keras.layers.SparseFeatures(feature_columns)
model = tf.keras.Sequential()
model.add(sparse_feature_layer)
model.add(tf.keras.premade.LinearRegressor())

dftrain = pd.read_csv('storage.googleapis.com/tf-datasets/titanic/train.csv')
y_train = dftrain.pop('survived')
ds = tf.data.Dataset.from_tensor_slices(dict(dftrain), y_train)
model.train(ds, epochs=10)

// usage for processing layers.
age = tf.keras.Input(shape=(1,), dtype=tf.float32, name='age')
bucketized_age = tf.keras.Discretize(age, num_buckets=5)
occupation = tf.keras.Input(shape=(None,), dtype=tf.string, name='occupation')
occupation_id = tf.keras.VocabLookup(vocabulary_list)(occupation)
processing_stage = tf.keras.ProcessingStage(
    inputs=[age, occupation], outputs=[bucketized_age, occupation_id])

premade_model = tf.keras.premade.LinearClassifier()
output = premade_model(processing_stage.outputs)
model = tf.keras.Model(inputs=processing_stage.inputs, outputs=[output])

dftrain = pd.read_csv('storage.googleapis.com/tf-datasets/titanic/train.csv')
y_train = dftrain.pop('survived')
model.train(ds, epochs=10)
```