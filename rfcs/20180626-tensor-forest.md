# TensorForest Estimator

| Status        | Proposed      |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Peng Yu(yupbank@gmail.com) |
| **Sponsor**   | Natalia P (Google)                 |
| **Updated**   | 2018-06-26                                           |

## Objective

In this doc, we discuss the TensorForest Estimator API, which enable user create
[Extremely Randomized Forest](http://www.montefiore.ulg.ac.be/~ernst/uploads/news/id63/extremely-randomized-trees.pdf)
Classifier and Regressor.
And by inheriting from the `Estimator` class, all the corresponding interfaces will be supported

## Motivation

Since tree algorithm is one of the most popular algorithm used in kaggle competition
and we already have a contrib project [tensor_forest](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest) and people like them. If would be beneficial to move them inside of canned estimators.

## Design Proposal

### TensorForestClassifier

```
bucketized_feature_1 = bucketized_column(
	numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
bucketized_feature_2 = bucketized_column(
	numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

classifier = estimator.TensorForestClassifier(feature_columns=[bucketized_feature_1, bucketized_feature_2],
						model_dir=None,
						n_classes=2,
						label_vocabulary=None,
						n_trees=100,
						max_nodes=1000,
						num_trainers=1,
						num_splits_to_consider=10,
						split_after_samples=250,
						bagging_fraction=1.0,
						feature_bagging_fraction=1.0,
						base_random_seed=0,
						config=None)


def input_fn_train():
  ...
  return dataset

classifier.train(input_fn=input_fn_train)

def input_fn_eval():
  ...
  return dataset

metrics = classifier.evaluate(input_fn=input_fn_eval)
```

Here are some explained details for the classifier parameters:

*   **feature_columns:** An iterable containing all the feature columns used by the model.
  All items in the set should be instances of classes derived from `FeatureColumn`.
*   **n_classes:** Defaults to 2. The number of classes in a classification problem.
*   **model_dir:** Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.
*   **label_vocabulary:**  A list of strings represents possible label values. If given, labels must be string type and have any value in `label_vocabulary`. If it is not given, that means labels are already encoded as integer or float within [0, 1] for `n_classes=2` and encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .  Also there will be errors if vocabulary is not provided and labels are string.
*   **n_trees:** The number of trees to create. Defaults to 100. There usually isn't any accuracy gain from using higher values.
*   **max_nodes:**  Defaults to 10,000. No tree is allowed to grow beyond max_nodes nodes, and training stops when all trees in the forest are this large.
*   **num_splits_to_consider:** Defaults to `sqrt(num_features)` capped to be between 10 and 1000. In the extremely randomized tree training algorithm, only this many potential splits are evaluated for each tree node.
*   **split_after_samples:** Defaults to 250. In our online version of extremely randomized tree training, we pick a split for a node after it has accumulated this many training samples.
*   **bagging_fraction:** If less than 1.0, then each tree sees only a different, random sampled (without replacement), bagging_fraction sized subset of the training data. Defaults to 1.0 (no bagging) because it fails to give any accuracy improvement our experiments so far.
*   **feature_bagging_fraction:** If less than 1.0, then each tree sees only a different feature_bagging_fraction * num_features sized subset of the input features. Defaults to 1.0 (no feature bagging).
*   **base_random_seed:** By default (base_random_seed = 0), the random number generator for each tree is seeded by a 64-bit random value when each tree is first created. Using a non-zero value causes tree training to be deterministic, in that the i-th tree's random number generator is seeded with the value base_random_seed + i.
*   **config:** `RunConfig` object to configure the runtime settings.

### TensorForestRegressor

```
bucketized_feature_1 = bucketized_column(
	numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
bucketized_feature_2 = bucketized_column(
	numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

regressor = estimator.TensorForestRegressor(feature_columns=[bucketized_feature_1, bucketized_feature_2],
						model_dir=None,
						label_dimension=1,
						n_trees=100,
						max_nodes=1000,
						num_trainers=1,
						num_splits_to_consider=10,
						split_after_samples=250,
						bagging_fraction=1.0,
						feature_bagging_fraction=1.0,
						base_random_seed=0,
						config=None)


def input_fn_train():
  ...
  return dataset

regressor.train(input_fn=input_fn_train)

def input_fn_eval():
  ...
  return dataset

metrics = regressor.evaluate(input_fn=input_fn_eval)
```

Here are some explained details for the regressor parameters:

*   **feature_columns:** An iterable containing all the feature columns used by the model. All items in the set should be instances of classes derived from `FeatureColumn`.
*   **model_dir:** Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.
*   **label_dimension:** Defaults to 1. Number of regression targets per example.
*   **n_trees:** The number of trees to create. Defaults to 100. There usually isn't any accuracy gain from using higher values.
*   **max_nodes:**  Defaults to 10,000. No tree is allowed to grow beyond max_nodes nodes, and training stops when all trees in the forest are this large.
*   **num_splits_to_consider:** Defaults to `sqrt(num_features)` capped to be between 10 and 1000. In the extremely randomized tree training algorithm, only this many potential splits are evaluated for each tree node.
*   **split_after_samples:** Defaults to 250. In our online version of extremely randomized tree training, we pick a split for a node after it has accumulated this many training samples.
*   **bagging_fraction:** If less than 1.0, then each tree sees only a different, random sampled (without replacement), bagging_fraction sized subset of the training data. Defaults to 1.0 (no bagging) because it fails to give any accuracy improvement our experiments so far.
*   **feature_bagging_fraction:** If less than 1.0, then each tree sees only a different feature_bagging_fraction * num_features sized subset of the input features. Defaults to 1.0 (no feature bagging).
*   **base_random_seed:** By default (base_random_seed = 0), the random number generator for each tree is seeded by a 64-bit random value when each tree is first created. Using a non-zero value causes tree training to be deterministic, in that the i-th tree's random number generator is seeded with the value base_random_seed + i.
*   **config:** `RunConfig` object to configure the runtime settings.

## Questions and Discussion Topics

TBD
