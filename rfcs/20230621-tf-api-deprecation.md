# TensorFlow API Deprecation Charter

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) |
| **Author(s)** | Ayush Jain (ayh@google.com)                          |
| **Sponsor**   | Rohan Jain (rohanj@google.com)                       |
| **Updated**   | 2023-06-21                                           |

## Objective

This proposal outlines an explicit policy for long-term code health, specific to TensorFlow.

Because of TensorFlow’s position in both the Google-internal and third-party ML development worlds, we take special care when deprecating or removing public APIs. 

Note: As a primarily process RFC, there are a few sections of the RFC template not included in this draft proposal.

## Motivation

The TensorFlow code base has grown continuously over many years, increasing in complexity to support multiple hardware backends, supporting eager and graph modes, and more. TensorFlow has followed the Semantic Versioning policy, with [strict backward compatibility guarantees](https://www.tensorflow.org/guide/versions) that only allows for backward incompatible changes to the public Python API in a major version release. 

TensorFlow had only one major version revision, and this strict adherence to backward compatibility has come with a high maintenance burden as well as a complicated and occasionally redundant API surface for users. By establishing a mechanism to clean up unnecessary and/or unused behaviors, TensorFlow's users, contributors, and developers alike can build with greater simplicity and speed.

## Proposal

### Eligibility

An API is eligible for deprecation whenever there exists a better way to fulfill its purpose, making alternatives obsolete. “Better” and “obsolete” are terms subject to maintainer discretion - the API may either be part of an unsupported and since-replaced suite, exhibit inferior performance, not comply with Function-based execution, or otherwise. As defined, an eligible API must then have a functional, non-experimental equivalent available that users can be directed upon deprecation. 

If there are fewer than fewer than 500 dependent non-TF-fork repositories on GitHub, said API is deemed suitable for public deprecation. If it does not meet this requirement, it is ineligible and the process stops here. If usage diminishes over time, it can be reevaluated at a later date.

Note that while we use API as a blanket term throughout this document, this policy also explicitly covers TensorFlow ops that have fallen out of favor. For an op to be eligible for deprecation, it must be expressible through a graph rewrite using other existing ops.

Under this policy, deprecation is a precursor to removal. For this reason, the TensorFlow API Owners group gives ultimate approval to deprecate a nominated API when said nomination is contentious.

### Policy

To preserve developer velocity and ensure a maintainable codebase, TensorFlow reserves the flexibility to make exceptions to the strict guidelines imposed by [SemVer 2.0](https://semver.org/#spec-item-8). Namely, TensorFlow lays out here an explicit procedure for **the** **inclusion of backwards incompatible changes in non-major (i.e. TF2.X) version releases**.

Consider the sunsetting of some `tf.foo`, proposed during TF 2.X:

* Add deprecation messages for `tf.foo` in TF 2.(X+1). 
    * If the deprecated API’s replacement is not already publicly available, introduce it alongside the deprecation messaging during TF 2.(X+1).
* Adjust defaults wherever applicable to use updated equivalents in TF 2.(X+1).
* Remove `tf.foo` with the TF release nearest after 12 months after initial warnings were added.

This process takes place over **a full year**, giving users substantial time to adapt. If users are unable to migrate in that period, they may pin to TF 2.X and move when they deem it suitable or necessary.

Subsequent sections go into more depth on the execution of this process.

### Deprecation

Deprecation in TensorFlow serves as a warning to users to act and avoid future friction upon removal of a deprecated API. It is a way to communicate useful information about superior alternatives to unsupported behaviors and design patterns, directly to the users who need it. It is intended to incentivize expeditious migration and emphasize its benefits by making the impending end-of-support clear. 

A TensorFlow API is considered fully deprecated when:

* Runtime warnings appear when a user attempts development with deprecated APIs.
    * Runtime warnings can be added by wrapping a deprecated pattern with `@deprecation(...)` as defined in [deprecation.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/deprecation.py), or, if of greater scale, by adding a distinct deprecated export wrapper in [tf_export.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/tf_export.py) (and using said new wrapper).
    * **Internal usage of deprecated APIs should not raise warnings**. Any warnings should only come from a user-facing call of the deprecated API. 
        * Ex: if `tf.bar` uses deprecated `tf.foo`, calling `tf.bar` should not trigger a warning.
* As appropriate, examples (e.g. experimental Colab notebooks) are created demonstrating the replacement of any deprecated modules or methods.
* The API is covered by a publicly available _strict mode_ which users can optionally enable to convert warnings to failures, with error messages that guide them to replacements.
    * Strict mode is a globally applied state, switched on with `tf.exprimental.enable_strict_mode()`. 
    * Once enabled, ignorable runtime warnings for deprecated APIs will instead be replaced with errors, detailing the deprecation and suggested alternatives.
* Alongside release in TF Nightly, messaging is sent through appropriate communication channels to inform of the upcoming deprecation, rationale, and available substitutes.
    * Depending on the nature and scope of deprecation, this may be done at the individual API level or in interrelated block(s).
    * The primary channel for this will be the TensorFlow release notes. Each release shall have a dedicated, prominent Deprecation section (à la [NumPy](https://numpy.org/doc/stable/release/1.24.0-notes.html#deprecations)) where announcements are made.

### Removal

Removal is substantial, but straightforward. Code is deleted. Should the appropriate precursor steps be taken as specified, the groundwork is set for an as-smooth-as-possible introduction of backwards incompatibility.

### Compatibility

These are relevant for API removal.

#### SavedModels

SavedModels written with one version of TensorFlow can be loaded and evaluated with a later version of TensorFlow within the same major release. This is an end-to-end guarantee: the model can be loaded, the functions registered, …, all the way through runtime execution.

#### GraphDefs and Checkpoints

GraphDef currently has versioning and a given release of TensorFlow supports a range of versions of the GraphDef.  Note that the GraphDef version mechanism is separate from the TensorFlow version. 

TensorFlow will provide a **1 year** backward compatibility guarantee for GraphDefs. This is to support restoring graphs created with the immediately previous version of TensorFlow. 

Forward compatibility support will also be guaranteed for 3 weeks, which implies creating a GraphDef in a newer version of TensorFlow and loading it into an older version of TensorFlow [standing policy].

All the same applies to checkpoints. 

### Engineering Impact
* Package size and import time should gradually decrease as APIs removed, although likely not substantially.

### User Impact
Users will begin to see runtime warnings for usage of deprecated APIs, and, upon removal ~12 months from deprecation, will be expected to either:
* Replace its usage with the recommended alternative, OR 
* Pin to the last version still including the API to preserve its functionality.

## Questions and Discussion Topics

* Should we consider a more aggressive removal timeline than 1 year post-deprecation (e.g. 6 months)?
