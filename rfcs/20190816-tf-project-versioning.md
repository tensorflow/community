# Project versioning in the TensorFlow organization

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Edd Wilder-James (ewj@google.com), Martin Wicke (wicke@google.com) |
| **Sponsor**   | Kevin Haas (khaas@google.com)                 |
| **Updated**   | 2019-08-16                                           |

## Objective

This document describes best practices for numbering versions of projects
that form part of the TensorFlow suite of projects. This practice is required for dependent
projects hosted under the [TensorFlow organization](https://github.com/tensorflow) on
GitHub, and advisory for dependent projects hosted elsewhere.

## Definitions

"TensorFlow" in this document refers to the core TensorFlow project, as developed in
GitHub `tensorflow/tensorflow`.

## Motivation

As the number of projects dependent on TensorFlow increases, such as those shipped by
SIG Addons or IO, it is helpful to maintainers to understand the constraints on how 
to number their releases.

## Versioning Policy

All projects must follow [semantic versioning](https://semver.org/). 

Until a project reaches 1.0, it does not have to make any backward compatibility guarantees.

Projects should not try to track major TensorFlow versioning to indicate compatibility
with particular TensorFlow releases. Instead, compatibility must be signalled
by the use of dependencies in `pip`, or whichever package manager is being used by the project.

Within the constraints of semantic versioning, project maintainers should feel free to do
whatever is best for their projects and users.

## Review Feedback

Included as advisory but not binding.

* Jason Zaman: It might be a good idea to also mention stability guarantees, and things that are excluded from them. eg TensorFlow itself says anything that's `tf.foo.experimental.bar` is not stable and is allowed to change at anytime, and other projects should think about having a similar mechanism if needed.
