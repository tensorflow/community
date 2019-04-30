# Keras pre-made models

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com)|
| **Sponsor**   | Francois Chollet (fchollet@google.com), Alexandre Passos (apassos@google.com) |
| **Updated**   | 2019-04-29                                           |

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

Users can use [feature_columns](https://www.tensorflow.org/guide/feature_columns) with pre-made models for feature engineering. We use **pre-made** and **canned** interchangeably in this doc.

Several side notes: This document should serve as a foundation for other pre-made models that are not included in this proposal, such as Kmeans, random forest, collaborate filtering, which are yet to come; Detailed design regarding implementation details for BoostedTree, such as distributed training, saving & restoring, acceleration, supported feature columns should live in separate docs. 

## Design Proposal

### Challenges

Many canned models, including BoostedTrees, KMeans (as well as WideDeep and many others more in the future) are highly complicated and do not follow the simple foward & backprop workflow, an assumption heavily relied on by Keras. Building such models, while not compromising the basic composability of Keras (i.e., compose layers on top of layers), is the main challenge we need to resolve in this document. Another challenge is that these models can have multiple training phases (such as collaborative filtering models). We propose the following two approaches to address them:

### Proposal 1: Customized training function & composability

We propose to let each subclassed pre-made model override the training function. It is optional to provide a special subclass `CannedModel` if other methods such as `compile` and `fit` needs to be overriden as well. In traditional Keras models, such training function is dominated by autodiff - given the forward pass of the model, the gradients for each Operation is generated and backprop computation is automatically laid out for the entire model. However such assumption is only valid for neural network based supervised learning architecture. For many other scenarios, we would need to break this assumption:

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
class BoostedTreesClassifier(Model):

def __init__(self, feature_columns, num_classes=2, num_trees=10, learning_rate=0.1,
             max_depth=None, min_samples_split=2, min_samples_leaf=1, partial_data=1.)
  self._quantile_accumulator = QuantileAccumulator()
  self._stats_accumulator = StatsAccumulator()
  self._tree_ensemble = TreeEnsemble(num_classes, num_trees, max_depth)
  self._input_layer = tf.keras.layers.DenseFeatures(feature_columns)
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

This section describes the API signatures of the canned models that we propose.

### Linear Models

We provide linear models for classification and regression, where users can customize the loss and metrics. These linear models support mini-batch training. For large models, model parallelism needs to be performed via sharding the model properly, see [partition strategy in 2.0](https://github.com/tensorflow/community/blob/master/rfcs/20190116-embedding-partitioned-variable.md). 

`LinearRegressor` and `LinearClassifier` are different from `tf.keras.layer.Dense`. It accepts feature columns and creates variables separately for each feature column provided. The user is able to get the trained weights for each column.


#### Linear Regression

For linear regression models, mean squared error loss is used by default. Metrics can be customized.

```python
LinearRegressor(Model):

  def __init__(self,
               output_dim=1,
               use_bias=True,
               kernel_regularizer=None,
               bias_regularizer=None,
               input_dim=None,
               feature_columns=None,
               merge_mode="sum",
               *args, **kwargs):
    """tf.keras.canned.LinearRegressor

    Args:
      output_dim: Dimensionality of the output vector (scalar regression is the default).
      use_bias: whether to calculate the bias/intercept for this model.
        If set to False, no bias/intercept will be used in calculations, e.g., the data is already centered.
      kernel_regularizer: Regularizer instance to use on kernel matrix.
      bias_regularizer: Regularizer instance to use on bias vector.
      input_dim: Optional. Mutually exclusive with `feature_columns`. Dimensionality of the inputs.
      feature_columns: Optional. An iterable containing the feature columns used as inputs of the model.
        All items in the set should be instances of classes derived from `FeatureColumn`.
      merge_mode: String. This argument is only valid if "feature_columns" is not None.
        Specifies how to combine multiple features together.
        One of 'sum', 'mul', 'avg'.
    """
    pass

  def compile(self, optimizer="rmsprop", loss="mse", metrics=None):
    pass

  def get_weights(self, feature_columns=None)
    """Get weight values for the linear model.
    """
    pass

  def set_weights(self, weights, feature_columns=None)
    """Set weight values for the linear model.
    """
    pass
```

#### Logistic Regression

For logistic regression models, binary or categorical cross entropy loss is used by default. Metrics can be customized.

```python
class LinearClassifier(Model):

  def __init__(self,
               num_classes=2,
               use_bias=True,
               kernel_regularizer=None,
               bias_regularizer=None,
               input_dim=None,
               feature_columns=None,
               merge_mode="sum",
               *args, **kwargs):
    """tf.keras.canned.LinearClassifier

    Args:
      num_classes: Number of output classes (>=2).
      use_bias: whether to calculate the bias/intercept for this model.
        If set to False, no bias/intercept will be used in calculations, e.g., the data is already centered.
      kernel_regularizer: Regularizer instance to use on kernel matrix.
      bias_regularizer: Regularizer instance to use on bias vector.
      input_dim: Optional. Mutually exclusive with `feature_columns`. Dimensionality of the inputs.
      feature_columns: Optional. An iterable containing the feature columns used as inputs of the model.
        All items in the set should be instances of classes derived from `FeatureColumn`.
      merge_mode: String. This argument is only valid if "feature_columns" is not None.
        Specifies how to combine multiple features together.
        One of 'sum', 'mul', 'avg'.
    """
    pass

  def compile(self, optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=None):
    pass

  def get_weights(self, feature_columns=None)
    """Get weight values for the linear model.
    """
    pass

  def set_weights(self, weights, feature_columns=None)
    """Set weight values for the linear model.
    """
    pass
```

### Wide and Deep Models

#### Wide-Deep Regression

Mean squared error loss is used by default. The FTRL optimizer is used by default for the linear branch, and the Adagrad optimizer is used by default for the deep branch. Metrics can be customized.

```python
class WideDeepRegressor(Model):
  """`tf.keras.canned.WideDeepRegressor`

  Args:
    linear_model: Uncompiled instance of `LinearRegressor`.
    deep_model: Uncompiled instance of a Sequential model starting starting with a `DenseFeatures` layer.
      Its output must match the output of the linear model.
  """

  def __init__(self, linear_model, deep_model, *args, **kwargs):
    pass

  def compile(self, linear_optimizer="ftrl", deep_optimizer="adagrad", loss="mse", metrics=None):
    pass
```

#### Wide-Deep Classification

```python
class WideDeepClassifier(Model):
  """`tf.keras.canned.WideDeepClassifier`

  Args:
    linear_model: Uncompiled instance of `LinearRegressor`.
    deep_model: Uncompiled instance of a Sequential model starting starting with a `DenseFeatures` layer.
      Its output must match the output of the linear model.
  """

  def __init__(self, linear_model, deep_model, *args, **kwargs):
    pass

  def compile(self, linear_optimizer="ftrl", deep_optimizer="adagrad", loss="sparse_categorical_crossentropy", metrics=None):
    pass
```

### BoostedTrees

#### BoostedTrees Regression

For GDBT regressor, mean squared error loss is used by default. Metrics can be customized.

```python
class BoostedTreesRegressor(Model):

  def __init__(self, output_dim=1,
               num_trees=10,
               learning_rate=0.1,
               regularizer=None,
               min_node_weight=0.,
               max_depth=None,
               min_split_samples=2,
               min_leaf_samples=1,
               feature_columns=None):
    """tf.keras.canned.BoostedTreesRegressor.

    Args:
      output_dim: Dimensionality of the output vector (scalar regression is the default).
      num_trees: The number of boosting trees to perform.
      learning_rate: Shrinkage parameter to be used when a tree added to the model.
      regularizer: Regularizer instance.
      min_node_weight: Minimum sum hessian per leaf.
      max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
      min_split_samples: The minimum number of samples required to split an internal nodel.
      min_leaf_samples: The minimum number of samples required to be at a leaf node.
      feature_columns: An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
    """
    pass

  def compile(self, loss="mse", metrics=None):
    pass
```

#### BoostedTrees Classification

For GDBT classifier, binary or categorical cross entropy loss is used by default. Metrics can be customized.

```python
class BoostedTreesClassifier(Model):

  def __init__(self, num_classes=2,
               num_trees=10,
               learning_rate=0.1,
               regularizer=None,
               min_node_weight=0.,
               max_depth=None,
               min_split_samples=2,
               min_leaf_samples=1,
               feature_columns=None):
    """tf.keras.canned.BoostedTreesRegressor.

    Args:
      num_classes: Number of output classes (>=2).
      num_trees: The number of boosting trees to perform.
      learning_rate: Shrinkage parameter to be used when a tree added to the model.
      regularizer: Regularizer instance.
      min_node_weight: Minimum sum hessian per leaf.
      max_depth: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
      min_split_samples: The minimum number of samples required to split an internal nodel.
      min_leaf_samples: The minimum number of samples required to be at a leaf node.
      feature_columns: An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
    """
    pass

  def compile(self, loss="categorical_crossentropy", metrics=None):
    pass
```

## CannedEstimators

These are set as non goals but nice to have.
1. Compatibility: We propose to provde backward checkpoint compatibility with CannedEstimators so that users can restore a Keras CannedModel from existing checkpoints.
2. Reusability: We propose to reuse CannedModel.call() in CannedEstimator model_fn, and reuse CannedModel.make_train_function() in CannedEstimator Head.create_estimator_spec_train_op


## Questions and Discussion Topics

### Discussion with TFX

1. TFX requires getting weights using feature column.
Status: Supported in the doc.
2. Should we have a `DNNClassifier`/`DNNRegressor` class? This is normally supported in Keras via a Sequential model that starts with a `DenseFeatures` layer.
