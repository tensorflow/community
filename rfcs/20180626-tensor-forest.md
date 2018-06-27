# TensorForest Estimator

| Status        | Proposed      |
:---------------|:-----------------------------------------------------|
| **Author(s)** | Peng Yu(yupbank@gmail.com) |
| **Sponsor**   | Natalia P (Google)                 |
| **Updated**   | 2018-06-26                                           |

## Objective

In this doc, we discuss the TensorForest Estimator API, which enable user create
Random Forest models.

## Motivation

Since tree algorithm is one of the most popular algorithm used in kaggle competition
and we already have a contrib project tensor_forest and people like them. If would be 
beneficial to move them inside of canned estimators.

## Design Proposal

### Examples

```
classifier = random_forest.TensorForestEstimator(feature_columns, 
						n_classes,
						model_dir=None,
						weight_column=None,
						label_vocabulary=None,
  						n_trees=50, 
						max_nodes=1000,
						num_trainers=1,
						num_splits_to_consider=10,
						split_after_samples=250,
						bagging_fraction=1.0,
						feature_bagging_fraction=1.0,
						base_random_seed=0)

def input_fn_train():
  ...
  return dataset

classifier.train(input_fn=input_fn_train)

def input_fn_eval():
  ...
  return dataset

metrics = classifier.evaluate(input_fn=input_fn_eval)
```
## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.

