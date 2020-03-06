# TensorFlow Official Model Garden Redesign

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Jing Li (jingli@google.com), Hongkun Yu (hongkuny@google.com), Xiaodan Song (xiaodansong@google.com) |
| **Sponsor**   | Edd Wilder-James (ewj@google.com)                 |
| **Updated**   | 2019-08-02                                           |

## Objective

This document presents a proposal to redesign TensorFlow official model garden.
We aim to provide a central and reliable place to contain popular examples,
state-of-the-art models and tutorials to demonstrate the best practice in TF2.0
and illustrate real-world use cases.

## Motivation

The current [TF official model garden](https://github.com/tensorflow/models/tree/master/official)
mainly has ad hoc support. Example models are implemented using mixed TensorFlow
APIs in different coding styles and some of them have convergence and/or
performance regression. With TensorFlow 2.0 launch, there’s a great desire to
provide tensorflow users a clear and central place to showcase reliable TF2.0
models with the best practices to follow.

We want to take this opportunity to substantially improve the state of the
official model garden, and provide seamlessly end-to-end training and inference
user experience on a wide range of accelerators and mobile device chips. We hope
to encourage community to contribute innovations and improve TensorFlow
efficiency and usability.

## User Benefit

We aim to provide the best modeling experience via this revamp effort:

*   Usability and reliability
    *   keep official models well-maintained and tested for both performance and
        convergence.
    *   provide accessible model distribution via [TensorFlow Hub](https://www.tensorflow.org/hub) and share state-of-the-art research accomplishments.
    *   make training on both GPU and TPU an easy switch.
    *   provide reusable components for research and production.
*   End-to-end solutions
    *   provide seamless end-to-end training and inference solutions, where inference covers serving on TPU, GPU, mobile and edge devices.
    *   provide hyper parameter sets to tune models for various resource constraints.
    *   provide solutions with hyper parameters to scale model training to TPU pods or multi-worker GPUs.
    *   provide variants derived from standard models to tackle various practical tasks.

## Design Proposal

### Official model directory reorgnization

We are going to reorganize the official model directory to provide:

*   common libraries, mainly two types:
    *   Common training util library in TF2.0, model configuration and
        hyperparameter definition in a consistent style.
    *   Model category related common library, e.g. primitives as basic building
        block for NLP models, or common networks like resnet, mobilenet. We will follow the fundamental design of Keras
        layer/network/model to define and utilize model building blocks.
        
        **NOTE:** we are still figuring out what level of building block extraction would be the most useful and sharable
        during refactoring. Once we confirm the implementation is really useful, we will move it tensorflow/addons and/or tf.text.

*   popular state-of-the-art (SOTA) models for end users as a product.
*   reference models for performance benchmark testing.
    *   For models provided as SOTA models, we will share the network and
        modeling code, but have separate *main* modules. The main
        module for benchmark testing will have addtional flags and setups for
        performance testing.

The following table shows the detailed view of proposed model directory
structure. The SOTA model list will be updated to cover more categories.

| Directory     | Subdirectories       |   | Explainations                   |
:-------------- |:---------------------|:--|:------------------------------ |
| nlp           |                      |   | models/tasks for Natural Language Processing |
|               | modeling             |   | NLP modeling library |
|               | BERT                 |   |  |
|               | ALBERT                    |  |  |
|               | XLNET                |    |   |
|               | Transformer          |    |   |
|               | ...                  |    |   |
| vision        |                      |    | models/tasks for Computer Vision |
|               | image_classification |    | e.g. resnet, EfficientNet, ... |
|               | detection            |    | e.g. RetinaNet, Mask-RCNN, ... |
|               | ...                  |    |  |
| recommendation|                      |     |   |
|               | NCF                  |     |   |
| utils         |                      |   | Miscellaneous Utilities.          |
|               | ...                  |   |                                 |
| benchmarks    |                      |   | benchmark testing and reference models to validate tensorflow |
| staging       |                      |     | Utilities not in TF core yet, and not suitable for tf addons  |
| r1            |                      |     | tf1.x models and utils  |
|               | utils                |     |   |
|               | resnet50             |     |   |
|               | transformer          |     |   |
|               | wide_deep            |     |   |
|               | boosted_trees        |     |   |

### Pretrained model repository

We are going to provide the pretrained models for research exploration and
real-world application development. The plan is to integrate with [TensorFlow Hub](https://www.tensorflow.org/hub),
where users can access the Hub modules and SavedModel for pretrained checkpoints and links to the code in the model
garden.

### Convergence and Performance Testing

We have a benchmark testing framework to execute continuous performance and
accuracy tests for TensorFlow on different types of accelerators. All official
TF2.0 models are required to provide accuracy tests and these tests will be
automatically expanded to performance tests for continuous regression testing
and monitoring.

## Model Garden Sustainability

### Model Launch Criteria
To ensure that official models are well-maintained and tested, we are going to enforce the following criteria for launching a new model in the official model garden, except for staging folder:

*   Follow the best practice guideline for each model category.
*   Unit tests to verify the basics of the model.
*   Integrate the model to benchmark testing to ensure model’s accuracy should be on par with the original paper / SOTA results.
*   README with commands and procedures to reproduce the SOTA results, including:
    *   Input data generation if necessary
    *   Model execution, including all hyperparameters.

### Community contribution and staging

Due to fast ML development, we can’t possibly support all best-in-class models
up to date on our own. We highly encourage users to contribute to the official
model garden. After model garden refactoring (Phase 1), we plan to provide
a full list of wanted models to tensorflow community and encourage tensorflow
users to claim and contribute the models to the model garden.

We have different requirements from unifying interface, supporting all the chips
and platforms and enabling benchmarks for reference models. Thus, we could have
different stages of models. As we may have immediate needs to add some quick
models for benchmark and debugging, we will provide a staging folder to host
some drafts of SOTA or popular models. Once the staging models can converge and
support major functionalities of standard official models, we can judge whether
they meet the launch standard and migrate to official models or migrate them to
benchmark references.

### Maintenance and Deprecation

Given the nature of this repository, old models may become less and less
useful to the community as time goes on. In order to keep the repository
sustainable, we will be performing bi-annual reviews of our models to ensure
everything still belongs to the repo. For models to be retired, the current plan
is to move them to the archive directory and these models won't run regression
tests to ensure the quality and convergence.

The following details the policy for models in mature and staging phases:

*   Models graduated from staging subdirectory

    The models will be maintained by the model garden team. After we start to
    accept community contributions, we will put the contributors as model owners.

    These models will have continuous convergence and performance testing to
    make sure no regression. In general, we won’t deprecate these models unless:
    *   the model isn’t compatible with the TF APIs any more and have to be replaced by a new version
    *   a strictly better model shows up and the old model isn't needed by the community/market.

*   Models in staging:
    The model garden team will do quarterly review to check the status with the
    model contributors, such as:
    *   model convergence
    *   unit tests
    *   convergence tests
    *   coding style meets the TF2.0 best practice.
    If there’s no further commitment to improve the status in next 90 days, we
    will mark the model as deprecated, which is subject to be deleted.

### Official Model Releases
We will do release for the model garden starting from TF 2.0. Unit tests and
regression tests need to pass against the TF release. Deprecated models will be
removed from the release branch.

We will also create pip package per release version.

## Milestones

| Phases    | Milestones       | Notes                  |
|:--------  |:-----------------| :----------------------|
| Phase_1   | 1. Finished directory reorganization. 2. Add common modeling library. 3. Have 2-3 SOTA models for both NLP and Vision. | Not accepting community contributions during refactorization.|
| Phase_2   | Expand repository to cover more model types| Will accept community contributions on the solicited model list.|
