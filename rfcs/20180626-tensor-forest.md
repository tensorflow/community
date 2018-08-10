# TensorForest Estimator

| Status        | Accepted      |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Peng Yu(yupbank@gmail.com) |
| **Sponsor**   | Natalia P (Google)                 |
| **Updated**   | 2018-08-09                                           |

## Objective

### Goals

* Provide state of the art (in terms of model quality) online random forest implementation as a canned estimator in tf.estimator module.
* Design interface for the random forest estimator that is intuitive and easy to experiment with.
* Simplify the design of the current contrib implementation, making code cleaner and easier to maintain.
* Use all the new APIs for the new implementation, including supporting new feature columns and new Estimator interface
* Provide value for both
    - Users with small data, which fits into memory, by having a fast local version
    - Provide a distributed version that requires minimum configuration and works well out of the box.

### Non-Goals

* Provide an implementation with all the features available in [contrib version](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest)


## Motivation

Tree based algorithms have been very popular in the last decade. Random Forest by Breiman [1](https://books.google.ca/books/about/Classification_and_Regression_Trees.html?id=JwQx-WOmSyQC&redir_esc=y) is among one of the most widely used tree-based algorithms so far. Numerous empirical benchmarks demonstrate its remarkable performance on small to medium size datasets, high dimensional datasets and in Kaggle competitions([2](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf) , [3](http://icml2008.cs.helsinki.fi/papers/632.pdf), [4](https://www.kaggle.com/dansbecker/random-forests), [5](https://www.kaggle.com/sshadylov/titanic-solution-using-random-forest-classifier), [6](https://www.kaggle.com/thierryherrie/house-prices-random-forest)). Random Forest are champions in industry adoption, with numerous implementations (like scikit learn [7](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), Spark [8](https://spark.apache.org/docs/2.2.0/mllib-ensembles.html), Mahout [9](https://hub.packtpub.com/learning-random-forest-using-mahout/) and other) and tutorials ([10](https://medium.com/rants-on-machine-learning/the-unreasonable-effectiveness-of-random-forests-f33c3ce28883), [11](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd) etc. ) available online. Random Forests also remain popular in academic community, as demonstrated by a number of papers each year in venues like ICML, NIPS and JMLR ([12](http://proceedings.mlr.press/v37/nan15.html), [13](http://proceedings.mlr.press/v32/denil14.html), [14](https://icml.cc/Conferences/2018/Schedule?showEvent=3238),  [15](https://www.icml.cc/Conferences/2018/Schedule?showEvent=3181) and others).

Widespread adoption of random forests is due to the advantages of tree-based algorithms and power of ensembling.

__Tree-based algorithms are__

- **Easy to use:**
    - Tree models are invariant to inputs scale, so no preprocessing of numerical features is required (e.g. no normalization that is a must for gradient-descent based methods)
    - They work well out of the box and are not as sensitive to hyperparameters as neural nets, thus making them easier to tune
    - They are able to filter out irrelevant features
- **Easier to understand:**
    - Trees can be used to obtain feature importances
    - They are arguably easier to reason about, for example, tools like described [here](http://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/) are able to explain even the whole ensembles.

__Ensembling delivers__

- **Excellent performance**: by combining multiple predictors, ensembling obtains better predictive performance than that of using a single predictor [16](https://en.wikipedia.org/wiki/Ensemble_learning)
- **Reduced overfitting**: comes from the fact that individual predictors don't have to be very strong themselves (which can be obtained, for example, by using bagging or feature sampling techniques). Weaker predictors have fewer possibilities to overfit.
- **Fast inference time**:  Each predictor is independent from others, so inference time can be easily parallelized across different predictors

On top of all the above, random forests
- Provide a way to estimate the generalization performance without having a validation set: random forests use bagging, that allows to obtain out-of-bag estimates, that have been shown to be good metrics for generalization performance.

Compared with gradient boosted trees, which is another popular algorithm that uses both ensembling and tree based learners, random forests have the following
- __Advantages__:
    - **Much faster to train**: due to the fact that each tree is independent from another in a random forest, parallelization during inference time is trivial. Boosted trees, on the other hand, is an iterative algorithm that relies on predictors built so far to obtain the next predictor.
    - **Might be less prone to overfitting** since trees are less correlated with each other
    - **More "robust"** to different values of hyperparameters. They require less tuning and a default configuration works well most of the time, with tuning allowing usually just marginal improvements
- __Disadvantages__:
    - Need much larger ensembles due to the fact that trees are independent
    - Can't easily handle custom losses like ranking
    - Can have worse performance than boosted trees for a very complicated decision boundary

## Algorithm

[Extremely Randomized Forest](http://www.montefiore.ulg.ac.be/~ernst/uploads/news/id63/extremely-randomized-trees.pdf) is an online training algorithm, that makes quick split decisions.
In contrast with a classic random forest, in extremely randomized forests:

- Split candidates are generated after seeing only a number of samples from the data (as opposed to seeing the full dataset or a large portion, determined by the bagging fraction, of it.
- Splits quality is evaluated over the samples of the data, as opposed to full or a large portion of the dataset.

Those modifications allow to make quick decisions and forest can be grown in online fashion, and experiments demonstrate the ERF provide similar performance to that of classical forests (at the cost of having to potentially build much deeper trees).

At the start of training, the tree structure is initialized to a root node, and the leaf and growing statistics for it are both empty. Then, for each batch `{(x_i, y_i)}` of training data, the following steps are performed:

1. Given the current tree structure, each instance `x_i` is used to find the leaf assignment `l_i` where this instance falls into.
2. `Y_i` (the label of `x_i`) is used to update the leaf statistics of leaf `l_i`.
3. If the growing statistics for the leaf `l_i` do not yet contain `num_splits_to_consider` splits, `x_i` is used to generate another split. Specifically, a random feature value is chosen, and `x_i`'s value at that feature is used for the split's threshold.
4. Otherwise, `(x_i, y_i)` is used to update the statistics of every split in the growing statistics of leaf `l_i`. If leaf `l_i` has now seen `split_after_samples` data points since creating all of its potential splits, the split with the best score is chosen, and the tree structure is grown.


## Benchmark

Comparing with Scikit-learn ExtraTrees. Both using 100 trees with 10k nodes. And Scikit-learn ExtraTrees is a batch algorithm while TensorForest is a streaming/online algorithm [16](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxtbHN5c25pcHMyMDE2fGd4OjFlNTRiOWU2OGM2YzA4MjE).

|Data Set| #Examples | #Features| #Classes| TensorForest Accuracy(%)/R^2 Score| Scikit-learn ExtraTrees Accuracy(%)/R^2 Score|
|-------|:---------:|:---------:|:---------:|:---------:|---------:|
| Iris| 150| 4| 3| 95.6| 94.6|
|Diabetes| 442| 10| Regression| 0.462| 0.461|
|Boston| 506| 13| Regression| 0.793| 0.872|
|Digits| 1797| 64| 10| 96.7| 97.6|
|Sensit(Comb.)| 78k| 100| 3| 81.0| 83.1|
|Aloi| 108K| 128| 1000| 89.8| 91.7|
|rcv1| 518k| 47,236| 53| 78.7| 81.5|
|Covertype| 581k| 54| 7| 83.0| 85.0|
|HiGGS| 11M| 28| 2| 70.9| 71.7|

With single machine training, TensorForest finishes much faster on big dataset like HIGGS, takes about one percent of the time scikit-lean required [17](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxtbHN5c25pcHMyMDE2fGd4OjFlNTRiOWU2OGM2YzA4MjE).

## Design Proposal

### Interface
### TensorForestClassifier

```python
feature_1 = numeric_column('feature_1')
feature_2 = numeric_column('feature_2')

classifier = estimator.TensorForestClassifier(feature_columns=[feature_1, feature_2],
						model_dir=None,
						n_classes=2,
						label_vocabulary=None,
						head=None,
						n_trees=100,
						max_nodes=1000,
						num_splits_to_consider=10,
						split_after_samples=250,
						config=None)


def input_fn_train():
  ...
  return dataset

classifier.train(input_fn=input_fn_train)

def input_fn_predict():
  ...
  return dataset

classifier.predict(input_fn=input_fn_predict)

def input_fn_eval():
  ...
  return dataset

metrics = classifier.evaluate(input_fn=input_fn_eval)
```

Here are some explained details for the classifier parameters:

- **feature_columns**: An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from FeatureColumn.
- **n_classes**: Defaults to 2. The number of classes in a classification problem.
- **model_dir**: Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into an estimator to continue training a previously saved model.
- **label_vocabulary**: A list of strings representing all possible label values. If provided, labels must be of string type and their values must be present in label_vocabulary list. If label_vocabulary is omitted, it is assumed that the labels are already encoded as integer values within {0, 1} for n_classes=2, or encoded as integer values in {0, 1,..., n_classes-1} for n_classes>2 . If vocabulary is not provided and labels are of string, an error will be generated.
- **head**: .A `head_lib._Head` instance, the loss would be calculated for metrics purpose and not being used for training. If not provided, one will be automatically created based on params
- **n_trees**: The number of trees to create. Defaults to 100. There usually isn't any accuracy gain from using higher values (assuming deep enough trees are built).
- **max_nodes**: Defaults to 10k. No tree is allowed to grow beyond max_nodes nodes, and training stops when all trees in the forest are this large.
- **num_splits_to_consider**: Defaults to sqrt(num_features). In the extremely randomized tree training algorithm, only this many potential splits are evaluated for each tree node.
- **split_after_samples**: Defaults to 250. In our online version of extremely randomized tree training, we pick a split for a node after it has accumulated this many training samples.
- **config**: RunConfig object to configure the runtime settings.



### TensorForestRegressor

```python
feature_1 = numeric_column('feature_1')
feature_2 = numeric_column('feature_2')

regressor = estimator.TensorForestRegressor(feature_columns=[feature_1, feature_2],
						model_dir=None,
						label_dimension=1,
						head=None,
						n_trees=100,
						max_nodes=1000,
						num_splits_to_consider=10,
						split_after_samples=250,
						config=None)


def input_fn_train():
  ...
  return dataset

regressor.train(input_fn=input_fn_train)

def input_fn_predict():
  ...
  return dataset

regressor.predict(input_fn=input_fn_predict)

def input_fn_eval():
  ...
  return dataset

metrics = regressor.evaluate(input_fn=input_fn_eval)
```

Here are some explained details for the regressor parameters:

- **feature_columns:** An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
- **model_dir:** Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.
- **label_dimension:** Defaults to 1. Number of regression targets per example.
- **head**: .A `head_lib._Head` instance, the loss would be calculated for metrics purpose and not being used for training. If not provided, one will be automatically created based on params
- **n_trees:** The number of trees to create. Defaults to 100. There usually isn't any accuracy gain from using higher values.
- **max_nodes:**  Defaults to 10,000. No tree is allowed to grow beyond max_nodes nodes, and training stops when all trees in the forest are this large.
- **num_splits_to_consider:** Defaults to `sqrt(num_features)`. In the extremely randomized tree training algorithm, only this many potential splits are evaluated for each tree node.
- **split_after_samples:** Defaults to 250. In our online version of extremely randomized tree training, we pick a split for a node after it has accumulated this many training samples.
- **config:** `RunConfig` object to configure the runtime settings.

### First version supported features

The first version will only:

- Support dense numeric features. Categorical features would need to be imported as one-hot encoding
- No sample weight is supported
- No feature importances will be provided


## High Level Design

Each tree in the forest is trained independently and in parallel.

For each tree, we maintain the following data:

1. Tree Resource over the DecisionTree [proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/decision_trees/proto/generic_tree_model.proto#L73). It contains information about the tree structure and has statistics for:

    - Non-leaf nodes: namely two children of each non-leaf node and the split used to route data between them. Each split looks at a single input feature and compares it to a threshold value. Right now only numeric features will be supported. Categorical features should be encoded as 1-hot.
    - Leaf nodes ([proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/decision_trees/proto/generic_tree_model.proto#L137)). Each leaf needs to gather statistics, and those statistics have the property that at the end of training, they can be turned into predictions. For classification problems, the statistics are class counts, and for regression problems they are the vector sum of the values seen at the leaf, along with a count of those values (which can be turned into the mean for the final prediction).

2. Fertile Stats resource over Growing statistics. Each leaf needs to gather data that will potentially allow it to grow into a non-leaf parent node. That data usually consists of

    - A list of potential splits, which is an array in a [proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensor_forest/proto/fertile_stats.proto#L80)
    - Split Statistics for each of those splits. Split statistics in turn consist of leaf statistics for their left and right branches, along with some other information that allows us to assess the quality of the split. For classification problems, that's usually the [gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) of the split, while for regression problems it's the mean-squared error.


During training, every tree is being trained completely independently. For each tree, for every batch of the data, we

  - First, pass through the tree structure to obtain the leaf ids.
  - Then we update the leaf statistics
  - Update the growing statistics
  - Pick the leaf to grow
  - And finally, grow the tree.

During inference, for every batch of data, we pass through the tree structure and obtain the predictions from all the trees and then we average over all the predictions.

## Distributed version

Since the trees are independent, for the distributed version, we would distribute the number of trees required to train evenly among all the available workers. For every tree, they would have two tf.resources available for training.

## Differences from the latest contrib version

- Simplified code with only limited subset of features (obviously, excluding all the experimental ones)
- New estimator interface, support for new feature columns and losses
- We will try to reuse as much code from canned boosted trees as possible (proto, inference etc)

## Future Work

Add sample importance, right now we don’t support sample importance, which it’s a widely used [feature](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier.fit).
## Alternatives Considered

## Questions and Discussion Topics

[Google Groups](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/yreM9FRiBs4)

[Github Pull Request Discussion](https://github.com/tensorflow/community/pull/3)
