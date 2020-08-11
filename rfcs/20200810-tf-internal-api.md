# tf.internal API namespace

| Status        | Proposed |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Qianli Zhu (scottzhu@google.com) |
| **Sponsor**   | Martin Wicke (wicke@google.com), Alex Apassos (apassos@google.com)|
| **Updated**   | 2020-08-10                                           |
| **Intended audience**| tf-api-owners, tf-addons, keras-team, deepmind/sonnet|

## Objective

Adding a new "internal" API namespace in TF to host APIs for framework building/testing, etc. The API will have looser contracts compared to core TF API.

## Motivation

TensorFlow has a strong contract to ensure the stability of its API. The core 
API, once added, can't be updated with backward incompatible change or removed. 
TF API owners will pay extra attention to any new API proposal due to this 
restriction, based on the cost we have to bear to maintain the API 
compatibility. See more details about API review 
[here](https://github.com/tensorflow/community/blob/master/governance/api-reviews.md).

Keras team is trying to split its code into a [standalone Github repository](https://github.com/tensorflow/community/blob/master/rfcs/20200205-standalone-keras-repository.md). 
One of the key actions before the repository split is that Keras code has to 
rely on the public TF API only, to avoid any future breakages due to private 
method/class change.

Historically, Keras was an internal component of TF, and relies heavily on a lot
of private TF utilities and functionalities that are not part of the public API. 
Exposing those functions into public APIs will be very confusing to the end 
user, since those functions are useful to framework-builders, but not typical 
TensorFlow users. In addition, these utilities would be overly restricted by the 
standard TensorFlow API contract, as we would like to change/remove them more 
often than the existing core API.


## Design Proposal

Add a new namespace "tensorflow.internal" to host the framework building/testing 
related APIs. <b>(See alternative naming in the sections below, the naming of 
the namespace will be discussed in the design meeting.)</b>

### Backward compatibility
The contract for all the APIs under this namespace is looser compared to core TF
API. It will remain backward compatible for at least 1 minor TF release. 

This means for any API that is added in 2.x release, it will remain the same in 
2.x+1. If we choose to remove the same API, we will mark it as deprecated in 
2.x+1, and delete it at 2.x+2 release. TensorFlow is released every 3 months, 
and this will give enough time for all the clients to change to the new 
API/alternative.

Any deprecation and backward incompatible API change will be explicitly called 
out in TF release notes. This applies to both core API and "internal" API.

### Acceptance Criteria
The candidate of the "internal" API should:

1. Does NOT fit for core TF API, otherwise it should be exposed as core TF API.
1. Are effectively required to build/test/deploy a framework or library on top 
   of TensorFlow, or to integrate such a framework into TensorFlow's API.
1. Mature enough and won't change behavior/interface for every release. 
1. Non-trivial, otherwise the function should be copied to the client side.
1. Widely used in the implementation of TF public APIs (i.e. new functionality 
   isn’t immediately added to the tf.internal namespace)
1. Has at least two downstream libraries which are known to need it.

TF API owners will review the new API proposal, and follow the existing review 
process for core TF API.


### Documentation and Docstring
The "internal" API should have the same style and standard as the core 
TensorFlow API, which is documented [here](https://github.com/tensorflow/community/blob/master/governance/api-reviews.md#docstrings). 
We should explicitly list out the difference between "internal" API and core
API, and also choose a different place on tensorflow.org so that existing user 
are not confused.

### Naming and sub namespace
Similar "internal" APIs should be grouped together as sub namespaces, e.g., test 
related APIs should live under "tf.internal.test". This is aligned with the 
existing TF naming conversion.

Try not to export experimental APIs since the "internal" API should be mature 
enough.

### Current candidate
The following list is created from the private TF method usage within Keras, 
when we were trying to convert Keras to use the public TF API only. This is by 
no means the full list, but will serve as the first wave of review requests we 
send to the API owner for review. We don't expect all of them to be approved, 
and will discuss with the API owner on a case to case basis.

|Symbol location  |API Name  | 
:-------------- |:---------------------------------------------------- |
|python.framework.func_graph.FuncGraph |tf.internal.FuncGraph     |
|python.framework.combinations.*| tf.internal.test.combinations.* |
|python.distribute.strategy_combinations.* |tf.internal.test.combinations.* |
|python.training.tracking.base.no_automatic_dependency_tracking|tf.internal.tracking.no_automatic_dependency_tracking |
|python.util.object_identity.*|tf.internal.object_identity.* |
|python.util.tf_decorator.*|tf.internal.decorator.* |
|python.util.tf_inspect.*|tf.internal.inspect.* |

### Alternative Names
1. <b>"tf.internal"</b>: It gives the user the impression that this is not a 
   standard user facing API (good), and contains some private/implementation 
   details that are internal to TF. 
1. <b>"tf.infra"</b>: infrastructure is aligned with "building blocks" and low 
   level functionalities, like file system/network etc. So far, the APIs we want
   to add are still high level APIs and utility functions. 
1. By Martin <b>"tf._internal"</b>: the extra "_" emphasis in the pythonic way
   that this is for private usage.


## Questions and Discussion Topics

1. Naming of the API namespace.

