# TensorFlow Integration Testing

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **Author**    | amitpatankar@google.com 								  |
| **Sponsor**   | gunan@google.com                                        |
| **Updated**   | 2018-10-24                                              |

## Objective

This document proposes the official way to test projects and repositories downstream from TensorFlow. With TensorFlow becoming more and more modularized, libraries that sit on top of core TensorFlow need to be tested. Unfortunately we cannot wait for any adjustments made to core TensorFlow to propagate through to a formal release and we need a way to have a reliable way of getting the latest stable TensorFlow to test any new changes to the external repositories. A great example is the estimator library which is moving out of TensorFlow, but is still heavily dependent on core TensorFlow changes.

## Motivation

There are three potential possibilities to test TensorFlow dependent libraries:

 * Test with the latest official release.
 * Test by building TensorFlow from source at HEAD on the maste branch.
 * Test with the old `tf-nightly`.
 
|Approach:                     |TF-Release|TF-Head  |Old `tf-nightly`|
|------------------------------|----------|---------|----------------|
|TensorFlow update latency     |Poor      |Excellent|Average         |
|Test setup overhead           |Excellent |Poor     |Excellent       |
|Stability                     |Excellent |Poor     |Poor            |
|Test dependencies immediately |Poor      |Excellent|Poor            |

None of these solutions are ideal for testing projects downstream from TensorFlow.

## Design Proposal

### New Testing Approach

The [renovated `tf-nightly` approach](https://github.com/tensorflow/community/blob/master/rfcs/20181026-tf-nightly.md) will combat the two issues that plague option 3 for testing TensorFlow dependent packages.

|Approach:                    |New `tf-nightly`  |
|-----------------------------|------------------|
|TensorFlow update latency    |Excellent         |
|Test setup overhead          |Excellent         |
|Stability                    |Excellent         |
|Test dependencies immediately|Excellent         |

#### Stability
Sometimes the `tf-nightly` packages were created but failed immediately when attempting `import tensorflow`. 

#### Test dependencies immediately
Sometimes `tf-nightly` packages are behind since there are infrastructure issues or the hash they build off of at midnight does not build. With the guaranteed latest green postsubmit, your test is guaranteed to be run against the latest stable TensorFlow code possibly from the previous day.


### Example testing strategy
Here is a quick example that shows how TensorFlow can work with Tensorboard. This example uses a virtualenv with Python3 to run a simple test that theoretically depends on the latest code from TensorFlow.

##### Create the virtual environment

```bash
$ virtualenv -p python3 tf
$ source tf/bin/activate
(tf)$ pip install --upgrade pip
```

#####  Install and check `tf-nightly` or `tf-nightly-gpu`

```bash
(tf)$ pip install --upgrade tf-nightly
Successfully installed tf-nightly-1.13.0.dev20181023
(tf)$ python -c 'import tensorflow as tf; print(tf.__version__)'
1.13.0-dev20181023
```

#####  Clone and test the dependent project

```bash
(tf)$ git clone git@github.com:tensorflow/tensorboard.git
Cloning into 'tensorboard'...
remote: Counting objects: 20684, done.
remote: Total 20684 (delta 0), reused 0 (delta 0), pack-reused 20683
Receiving objects: 100% (20684/20684), 12.17 MiB | 8.89 MiB/s, done.
Resolving deltas: 100% (15053/15053), done.
(tf)$ cd tensorboard
(tf)$ bazel run //tensorboard/plugins/scalar:scalars_demo
```


