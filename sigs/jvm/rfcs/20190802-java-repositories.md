# Java Repositories
| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author**    | Karl Lessard (karl.lessard@gmail.com) |
| **Sponsor**   | James Ring (Google) |
| **Updated**   | 2019-08-02                                           |

## Objective

Create new repositories under the `github.com/tensorflow` organization to host the code supported by SIG JVM, including the
actual Java client found in TensorFlow core repository.

## Motivation

In the spirit of TensorFlow modularization, one main goal of SIG JVM is to migrate [TensorFlow Java client](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/java)
to its own repository so it can evolve and be released independently from TensorFlow core libraries (internally, the SIG calls this migration the *Jexit*, which is self-explanatory).

Additionally, some repositories are also requested to distribute high-level abstractions of TensorFlow in Java that will also evolve independently
from the client and have their own release cycles.

## User Benefit

Having repositories outside the TensorFlow core will help in the development of some major changes to the client architecture, including
that might include the whole replacement of its native binding layer. Doing such experimentations in the main repository is certainly not advised.

Also, having distinct repositories should allow the SIG to take part in the code review process that could unblock
more quickly new features developed by its members and distribute them as soon as the communitiy agrees.

It is also important to note the *Jexit* is a good candidate to start TensorFlow modularization because it already relies heavily
on the C ABI for its interaction with TensorFlow core libraries.

## Design Proposal

The current request focuses on the creation of the two following repositories:

### /tensorflow/java

This is the main repository for hosting TF Java code. It will consist of multiple modules that will be all released altogether and build with Maven.

Right now, the list of modules that will take place in this repository is:

#### core

All artifacts composing the actual Java client, including the Java code, its native layer and different generators used to create Java classes dynamically at compile time, including TF operations wrappers. Each of these components will be also released as seperate modules
  
#### nio

A self-contained Java library that provides advanced support for large buffers I/O operations (exceeding 2<sup>32</sup> - 1 bytes) and for n-dimensional data structures

At some point, the Java client core will be also based on this library to improve I/O performances and usage. The `nio` 
name comes from the similarities between this library and the [`java.nio`](https://docs.oracle.com/javase/8/docs/api/java/nio/package-summary.html) 
package found in the JDK, that is unfortunately lacking the support of 64-bits indexation.
  
#### model-framework

A proper abstraction API (e.g. GraphRunner) that hides the raw tensors and so can be used by non-machine learning experts.
In the future those libraries will allow using the models in a transfer learning setting with TensorFlow Java as well.

More details in the next section.

#### keras

An adaptation of the Keras library to Java, that will serve as the main API for training on TF Java.
  
### /tensorflow/java-models

The java-models will contain Java inference libraries for various pre-trained TensorFlow models, based on the Java 
TF model framework. 

This repository hosts a set of Java libraries for loading and inferring various pre-trained TensorFlow models. 
It provides a quick reference integrating for some of the popular TensorFlow models such as object detection, pose estimation, face detection and alike.

The java-models will provide OOTB utilities for Java developers to jump start using various pre-trained models, archived locally and hosted online. 
For example they can use any of the object-detection models in https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md by 
just passing in the URI.

We will try to add models that complement the existing set of models and can be used as building blocks in other apps.
