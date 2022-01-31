# tf._internal API namespace

| Status        | Accepted |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [278](https://github.com/tensorflow/community/pull/278)|
| **Author(s)** | Qianli Zhu (scottzhu@google.com) |
| **Sponsor**   | Martin Wicke (wicke@google.com), Alex Passos (apassos@google.com)|
| **Updated**   | 2020-08-10                                           |
| **Intended audience**| tf-api-owners, keras-team|

## Objective

Adding a new "_internal" API namespace in TF to host APIs for framework building
/testing, etc. The API namespace will serve as a whitelist for client libraries 
to gradually migrate off the usage tf private API symbol.


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
of private TF utilities and functionalities that are not part of the public 
API. Exposing those functions into public APIs will be very confusing to the 
end user who uses tf to build/train models, since those functions are useful to
framework-builders, but not typical TensorFlow users.


## Design Proposal

Add a new namespace "tensorflow._internal" to host the framework building/
testing related APIs that are currently used by other high level API in TF 
(Keras, and estimator, etc.). This name space will be treated as a protected 
API, which shouldn't be used by the end users. It will have no API contract 
compared to core TF API. TF side must be aware of the usage from client lib, 
so that it can properly test the change to any of those internal API before TF 
side making any backward incompatible change.

### Backward compatibility
There is no API contract for all the functions under this namespace. TF side 
can make any backward incompatible change at any time. To prevent any breakage 
caused on the client side, TF must be aware who is using this API, so that any 
changes to those API will be verified against the client code.

### Acceptance Criteria
The candidate of the "_internal" API should:

1. Does NOT fit for core TF API, otherwise it should be exposed as core TF API.
1. Are currently used by other high level API to build/test/deploy a framework 
   or library on top of TensorFlow, or to integrate such a framework into 
   TensorFlow's API.
1. Non-trivial, otherwise the function should be copied to the client side.
1. Widely used in the implementation of TF public APIs or test TF related 
   features. (i.e. new functionality isn’t immediately added to the tf._internal
   namespace)

TF API owners will review the new API proposal, and follow the existing review 
process for core TF API.


### Lifespan of the internal API
The internal API should serve as a temporary solution for the client side, and 
they should gradually migrate to alternative public TF core API, or other 
approaches. Once all the usage for a certain internal API has been removed, it 
will be deleted from the internal API namespace.


### Documentation and Docstring
The "internal" API should have the same style and standard as the core 
TensorFlow API, which is documented [here](https://github.com/tensorflow/community/blob/master/governance/api-reviews.md#docstrings). 
We will hide this from the documentation site, since the end users are the 
target audience for this API.

### Naming and sub namespace
Similar "internal" APIs should be grouped together as sub namespaces, e.g., test 
related APIs should live under "tf.internal.test". This is aligned with the 
existing TF naming conversion.


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
|python.util.object_identity.*|tf.internal.object_identity.* |
|python.util.tf_decorator.TFDecorator|tf.internal.decorator.TFDecorator |
|python.util.tf_decorator.unwrap|tf.internal.inspect.unwrap |

### Alternative Names
1. <b>"tf.internal"</b>: It gives the user the impression that this is not a 
   standard user facing API (good), and contains some private/implementation 
   details that are internal to TF. 
1. <b>"tf.infra"</b>: infrastructure is aligned with "building blocks" and low 
   level functionalities, like file system/network etc. So far, the APIs we want
   to add are still high level APIs and utility functions. 
1. By Martin <b>"tf.\_internal"</b>: the extra "_" emphasis in the pythonic way
   that this is for private usage.
