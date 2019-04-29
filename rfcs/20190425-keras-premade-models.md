# Keras pre-made models

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Zhenyu Tan (tanzheny@google.com)|
| **Sponsor**   | Francois Chollet (fchollet@google.com), Alexandre Passos (apassos@google.com) |
| **Updated**   | 2019-04-25                                           |

## Objective

This document proposes several pre-made Keras models that would allow users to:
* build basic machine learning models easily
* compose them with other keras Layers
* replace [Canned Estimators](https://www.tensorflow.org/guide/premade_estimators) in TF 2.0.

## Motivation

Scikit-learn is the de facto framework for creating basic machine learning models. From tensorflow perspective,
it is beneficial to provide users several important basic models to possibly support the following features:
* distributed mini-batch training
* accelerated training (GPU & TPU)
* enable AutoML to select different pre-made models
* enable users to freely compose pre-made models with other Keras layers and models.

Currently most pre-made models from tensorflow have been exposed through [CannedEstimator](https://www.tensorflow.org/guide/premade_estimators). However estimator
* relies on continuous graph rebuilding and checkpoints reloading which slows down training
* relies on global collections and not TF 2.0 friendly.
* makes many advanced features such as meta/transfer learning difficult
* enforces user to create input functions when not necessary

Given the status, we propose the following pre-made models as a start:
1. Linear models including Linear Regression and Logistic Regression
2. DNN models including DNN Regression and DNN Classification
3. Wide-Deep models including Wide-Deep Regression and Wide-Deep Classification
4. BoostedTree models including BoostedTree Regression and BoostedTree Classification

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
BoostedTreesClassifier(Model):

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
We provide a generalized linear model for user to customize loss and metrics. The linear models inherently support mini-batch training. For large models, model parallelism needs to be performed via sharding the layer properly, see [partition strategy in 2.0](https://github.com/tensorflow/community/blob/master/rfcs/20190116-embedding-partitioned-variable.md). 
LinearModel is different than `tf.keras.layer.Dense`. It accepts feature columns and creates variables separately for each column. The user is able to get the trained weight for each column.
```python
`tf.keras.canned.LinearModel`
LinearModel(Model):

def __init__(self, use_bias=True, l1=0., l2=0., feature_columns=None, 
             sparse_combiner="sum", *args, **kwargs)
"use_bias": whether to calculate the bias/intercept for this model. If set to False, no bias/intercept will be used in calculations, e.g., the data is already centered.
"l1": L1 regularization strength for both kernel and bias.
"l2": L2 regularization strength for both kernel and bias.
"feature_columns": An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
"sparse_combiner": A string specifying how to reduce if a categorical column is multivalent. One of "mean", "sqrtn", "sum" regularization strength. This argument is only valid if "feature_columns" is not None.

def compile(self, optimizer="sgd", loss=None, metrics=None)
"optimizer": which optimizer to use. choose between sgd, momentum, adam, adadelta, adagrad, adamax, ftrl, sdca
"loss": String, objective function or `tf.losses.Loss` instance.
"metrics": which List of metrics to be evaluated by the model during training and testing. Typically use `metrics=[`accuracy`].

def get_weights(self, feature_columns=None)
get weight for the linear model. If feature_columns is present, then get the weights for the subset of feature_columns

def set_weights(self, weights, feature_columns=None)
set weight for the linear model. If feature_columns is present, then set the weights for the subset of feature_columns, `weights` are required to be dict in this case.
```

#### Linear Regression
For linear regression models, mean squared error loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.LinearRegressor`
LinearRegressor(LinearModel):

def __init__(self, use_bias=True, l1=0., l2=1.0, feature_columns=None, 
             sparse_combiner="sum", label_dimensions=1, *args, **kwargs)
"label_dimensions": number of regression targets per example.
others same as LinearModel.

def compile(self, optimizer="sgd", loss="mse", metrics=None)
same as LinearModel, with built-in MSE loss.
```

#### Logistic Regression
For logistic regression models, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.LinearClassifier`
LinearClassifier(LinearModel):

def __init__(self, num_classes=2, use_bias=True, l1=0.0, l2=0.0, feature_columns=None, 
             sparse_combiner="sum", label_vocabulary=None, *args, **kwargs)
"num_classes": Number of classes.
"label_vocabulary": A list of strings represents possible label values. If given, labels must be string type and have any value in `label_vocabulary`. If it is not given, that means labels are already encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`.
others same as LinearModel.

def compile(self, optimizer="sgd", loss="categorical_crossentropy", metrics=None)
same as LinearModel, with built-in binary/categorical crossentropy loss.

*Only binary classification is supported for sdca.
```
### DNN Models
We provide a generalized dnn model for user to customize loss and metrics. The dnn models inherently support mini-batch training.
```python
`tf.keras.canned.DNNModel`
DNNModel(Model):

def __init__(self, hidden_units, activation="relu", dropout=None, batch_norm=False, 
             feature_columns=None, *args, **kwargs)
"hidden_units": Iterable of number of hidden units per layer, including the last layer. All layers are fully connected.
"activation": Activation function applied after each layer.
"dropout": Fraction of the inputs to drop.
"batch_norm": Whether to use batch normalization after each layer.
"feature_columns": An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.

def compile(self, optimizer="sgd", loss=None, metrics=None)
"optimizer": which optimizer to use. choose between sgd, momentum, adam, adadelta, adagrad, adamax, ftrl. SDCA is not supported.
"loss": String, objective function or `tf.losses.Loss` instance.
"metrics": which List of metrics to be evaluated by the model during training and testing. Typically use `metrics=[`accuracy`].

def get_weights(self, feature_columns=None)
get weight for the dnn model. If feature_columns is present, then get the weights for the subset of feature_columns

def set_weights(self, feature_columns=None)
set weight for the dnn model. If feature_columns is present, then set the weights for the subset of feature_columns
```

#### DNN Regression
For dnn regression models, mean squared error loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.DNNRegressor`
DNNRegressor(DNNModel):

def __init__(self, hidden_units, activation="relu", dropout=None, batch_norm=False, 
             feature_columns=None, label_dimensions=1, *args, **kwargs)
"label_dimensions": number of regression targets per example.
others same as DNNModel.

def compile(self, optimizer="sgd", loss="mse", metrics=None)
same as DNNModel, with loss built-in MSE loss.
```

#### DNN Classification
For dnn classification models, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.DNNClassifier`

DNNClassifier(DNNModel):

def __init__(self, hidden_units, num_classes=2, activation="relu", dropout=None, 
             batch_norm=False, feature_columns=None, label_vocabulary=None, *args, 
             **kwargs)
"num_classes": Number of classes.
"label_vocabulary": A list of strings represents possible label values. If given, labels must be string type and have any value in `label_vocabulary`. If it is not given, that means labels are already encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`.
others same as DNNModel.

def compile(self, optimizer="sgd", loss="categorical_crossentropy", metrics=None)
same as DNNModel, with loss built-in binary/categorical crossentropy loss.
```
### Wide-Deep Models
We provide generalized wide deep model for user to customize loss and metrics.
```python
`tf.keras.canned.DNNLinearModel`
DNNLinearModel(Model):

def __init__(self, l1=0., l2=0., linear_use_bias=True, linear_feature_columns=None, 
             linear_sparse_combiner="sum", dnn_hidden_units, dnn_activation="relu", 
             dnn_dropout=None, dnn_batch_norm=False, dnn_feature_columns=None, *args, 
             **kwargs)
"l1": L1 regularization strength for both linear and dnn models.
"l2": L2 regularization strength for both linear and dnn models.
"linear_use_bias, linear_feature_columns, linear_sparse_combiner": same as LinearModel
"dnn_hidden_unites, dnn_activation, dnn_dropout, dnn_batch_norm, dnn_feature_columns": same as DNNModel

def compile(self, linear_optimizer="ftrl", dnn_optimizer="adagrad", loss=None, 
            metrics=None)
"linear_optimizer": which linear optimizer to use. choose between sgd, momentum, adam, adadelta, adagrad, adamax, ftrl, sdca
"dnn_optimizer": which linear optimizer to use. choose between sgd, momentum, adam, adadelta, adagrad, adamax, ftrl. SDCA is not supported.
"loss": String, objective function or `tf.losses.Loss` instance.
"metrics": which List of metrics to be evaluated by the model during training and testing. Typically use `metrics=[`accuracy`].

def get_linear_model(self):
Get the linear model built from wide-deep model.

def get_dnn_model(self):
Get the dnn model built from wide-deep model.

def get_weights(self, feature_columns=None)
get weight for the wide-deep model. If feature_columns is present, then get the weights for the subset of feature_columns. feature_columns can consist of either linear and dnn feature columns.

def set_weights(self, weights, feature_columns=None)
set weight for the wide-deep model. If feature_columns is present, then set the weights for the subset of feature_columns, `weights` are required to be dict in this case.
```

#### Wide-Deep Regression
For DNNLinear regressor, mean squared error loss is used by default, ftrl optimizer is used for linear path by default, and adagrad optimizer is used for dnn path by default. Metrics can be customized.
```python
`tf.keras.canned.DNNLinearRegressor`
DNNLinearRegressor(DNNLinearModel):

def __init__(self, l1=0., l2=0., label_dimensions=1, linear_use_bias=True, 
             linear_feature_columns=None, linear_sparse_combiner="sum", dnn_hidden_units, 
             dnn_activation="relu", dnn_dropout=None, dnn_batch_norm=False, 
             dnn_feature_columns=None, *args, **kwargs)
"label_dimensions": number of regression targets per example.
others same as DNNLinearModel.

def compile(self, linear_optimizer="ftrl", dnn_optimizer="adagrad", loss="mse", 
            metrics=None)
same as DNNLinearModel, with built-in MSE loss.
```

#### Wide-Deep Classification
For DNNLinear classifier, binary or categorical cross entropy loss is used by default, ftrl optimizer is used for linear path by default, and adagrad optimizer is used for dnn path by default. Metrics can be customized.
```python
`tf.keras.canned.DNNLinearClassifier`
DNNLinearClassifier(DNNLinearModel):

def __init__(self, l1=0., l2=0., num_classes=2, label_vocabulary=None, 
             linear_use_bias=True, linear_feature_columns=None,       
             linear_sparse_combiner="sum", dnn_hidden_units, dnn_activation="relu", 
             dnn_dropout=None, dnn_batch_norm=False, dnn_feature_columns=None, *args, 
             **kwargs)
"num_classes": Number of classes.
"label_vocabulary": A list of strings represents possible label values. If given, labels must be string type and have any value in `label_vocabulary`. If it is not given, that means labels are already encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`.
others same as DNNLinearModel.

def compile(self, linear_optimizer="ftrl", dnn_optimizer="adagrad", 
            loss="categorical_crossentropy", metrics=None)
same as DNNLinearModel, with built-in binary/categorical crossentropy loss.
```

### BoostedTrees
We provide a general gradient descent boosted tree model to customize loss and metrics. No optimizer is needed for this model.
```python
`tf.keras.canned.BoostedTreesModel`
BoostedTreesModel(Model):

def __init__(self, num_trees=10, learning_rate=0.1, l1=0., l2=0., min_node_weight=0., 
             max_depth=None, min_samples_split=2, min_samples_leaf=1, 
             feature_columns=None, partial_data=1.)
"num_trees": The number of boosting trees to perform.
"learning_rate": Shrinkage parameter to be used when a tree added to the model.
"l1": l1 regularization.
"l2": l2 regularization.
"min_node_weight": Minimum sum hessian per leaf.
"max_depth": The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
"min_samples_split": The minimum number of samples required to split an internal nodel.
"min_samples_leaf": The minimum number of samples required to be at a leaf node.
"feature_columns": An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
"partial_data": For mini batch training. The fractional partial dataset required to train a tree. By default it is 1.0 for full batch training dataset.


compile(self, loss=None, metrics=None)
"loss": String, objective function or `tf.losses.Loss` instance.
"metrics": which List of metrics to be evaluated by the model during training and testing. Typically use `metrics=[`accuracy`].
```

#### BoostedTrees Regression
For GDBT regressor, mean squared error loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.BoostedTreesRegressor`
BoostedTreesRegressor(BoostedTreesModel):

def __init__(self, label_dimensions=1, num_trees=10, learning_rate=0.1, l1=0., l2=0.,
             min_node_weight=0., max_depth=None, min_samples_split=2, min_samples_leaf=1, 
             feature_columns=None, partial_data=1.)
"label_dimensions": number of regression targets per example.
others same as BoostedTreesModel.

compile(self, loss=None, loss="mse", metrics=None)
same as BoostedTreesModel, with loss built-in mean squared error loss.
```

#### BoostedTrees Classification
For GDBT classifier, binary or categorical cross entropy loss is used by default. Metrics can be customized.
```python
`tf.keras.canned.BoostedTreesClassifier`
BoostedTreesClassifier(BoostedTreesModel):

def __init__(self, num_classes=2, num_trees=10, learning_rate=0.1, l1=0., l2=0.,
             min_node_weight=0., max_depth=None, min_samples_split=2, min_samples_leaf=1, 
             feature_columns=None, label_vocabulary=None, partial_data=1.)
"num_classes": The number of classes in the classification.
"label_vocabulary": A list of strings represents possible label values. If given, labels must be string type and have any value in `label_vocabulary`. If it is not given, that means labels are already encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`.
others same as BoostedTreesModel.

compile(self, loss=None, loss="categorical_crossentropy", metrics=None)
same as BoostedTreesModel, with loss built-in binary/categorical crossentropy loss.
```

## CannedEstimators
These are set as non goals but nice to have.
1. Compatibility: We propose to provde backward checkpoint compatibility with CannedEstimators so that users can restore a Keras CannedModel from existing checkpoints.
2. Reusability: We propose to reuse CannedModel.call() in CannedEstimator model_fn, and reuse CannedModel.make_train_function() in CannedEstimator Head.create_estimator_spec_train_op


## Questions and Discussion Topics

### Discussion with TFX
1. TFX requires getting weights using feature column.
Status: Supported in the doc.
