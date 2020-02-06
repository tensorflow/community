# Standalone Keras Repository

| Status        | Proposed |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Qianli Zhu (scottzhu@google.com), Francois Chollet (fchollet@google.com) |
| **Sponsor**   | Francois Chollet (fchollet@google.com), Karmel Allison (karmel@google.com) |
| **Updated**   | 2020-02-05                                           |

## Objective

Split the Keras code from Tensorflow main Github repository to its own
repository, and use Tensorflow as a public dependency.

Currently any contribution to Keras code will require building entire
Tensorflow, which is quite expensive to do with normal hardware and toolset 
setting. Having a separate repository will allow Keras to just build its own 
code without building Tensorflow. This should greatly improve the external 
developer velocity when they contribute to Keras code.

On the TensorFlow side, the split should increase maintainability and greater
modularity by ensuring that the TensorFlow high-level API can be built entirely
on top of public TensorFlow low-level APIs.

## Motivation

Building open source Tensorflow project end to end is an extensive exercise. 
With a standard GCP instance, it might take more than 1 hour to finish the whole
build process, it might take longer with a mac laptop. Although the local build 
cache might help speed up the follow up builds, the initial time cost is too 
high for normal software development. Internally Google has a distributed build 
and caching service, which we heavily rely on, and can finish all Keras tests 
within 5 mins, sadly we can't expose this to external contributors.

This lead to a few issues:

* Discourage contribution since external developers can't test their change and 
make sure it is correct.
* External developers send unverified PR and Google reviewers spend time back 
and forth, fixing the PR. Sometimes PR is just not moving forward because of the
lengthy feedback loop.

There are other side benefits if we split the repository. Keras, as the high 
level API, probably should have similar access level as end user for low level 
API. When splitting the repository, Keras will have to import Tensorflow and 
rely on TF public APIs. If Keras ends up using many TF private functions, it 
might be an indication of tight coupling of implementation details, or if some 
certain functions is used extensively, we might want to consider exposing them 
as public low level API.

This design is also aligned with design for [Modular Tensorflow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md), 
which splits the Tensorflow into smaller components, and not tightly coupled 
together.

## User Benefit

External contributors should experience much shorter turn-around time when 
building/testing Keras since they don't need to build TF anymore. This should 
have a positive effect for developer engagement.

## Design Proposal

### New location of the code

Github: the code will live at [keras-team/keras](https://github.com/keras-team/keras), 
joining the other Keras SIG projects and replacing the current external Keras 
codebase. tf.Keras will also replace Keras on PyPI.

Also considered: tensorflow/keras.

| keras-team/keras   | tensorflow/keras |
:------------------- |:------------------------------------------- |
|Under the umbrella of Keras SIG, which hosts all other Keras related projects like keras-application, KerasTuner etc.|Under the umbrella of tensorflow, which also hosts other TF related projects.|
|Lots of existing followers on keras-team, who may not be easily migrated to TF project.|No cross org repo management cost on Github. Could rely on a lot of existing setup in Tensorflow.|
|Can't easily delete keras project, which already have tons of stars and incoming reference links. Continued existence of external Keras code will create confusion ("why is there tensorflow/keras AND keras-team/keras?")|Issue/PR under the same org can be transferred easily, but not cross the different org. See here|

### Source of Truth
Tensorflow uses Google internal code repository as the source of truth. Every PR
is converted to Google internal change first, submitted in internal system, and 
then copied to Github as commits. At the same time, PR is marked as merged with 
the corresponding commit hash.

For Keras, since we are trying to promote user engagement, we hope it will use 
Github as a source of truth. This will have the following implications:

* We expect the majority of the code development/contribution from Github
and the devtools/tests/scripts should focus on the Github development use
case. See more details below.
* Keras CI/presubmit build for Github repo is targeting tf-nightly pip
package as dependency. This means any change to TF will take at most 24
hours to be reflected to Keras.
* Code will be mirrored to Google internal code repository via Google internal 
tools within a very short time window. The Google internal CI tests will run on 
HEAD for both Keras and TF code.
* CI build for Github might break when it sees a new version of tf-nightly, if 
certain behavior has been changed and didn't caught by unit tests. We have 
observed a few similar cases on [tf/addons]
(https://github.com/tensorflow/addons). We hope this can be reduced by stronger
unit test coverage by Google internel sysmte, when both TF and Keras code are 
tested on HEAD.
* PIP package management. Keras will now follow the tf-estimator approach. 
"pip install tensorflow" should also install keras as well. There are more
details for the PIP package in the [Improved pip package structure]
(https://github.com/tensorflow/community/pull/182).

### Dependency Cleanup
As the high level API, Keras should have the direct dependency to TF low level 
API, but not the other way around. Unfortunately there is some existing reverse 
dependency in the TF code that relies on Keras, which we should update/remove 
when we split the repository.

The current usage of Keras from Tensorflow are:
* Unit tests, which should be converted to integration tests, or port the tests
to Keras repository.
* feature_column.
* Legacy tf.layers in v1 API.
* legacy RNN cells.
* TPU support code for optimizer_v2.
* SavedModel.
* TF Lite.

All the imports can be changed to use dynamic import like below:

```python
try:
   from tensorflow.python.keras.engine import base_layer
except ImportError:
   tf.logging.error('keras is not installed, please pip install keras')
   base_layer = None
```

### Update Keras to use public TF API symbol

The current Keras code will still work if they still do:
```python
from tensorflow.python.ops import array_ops

ones = array_ops.ones([2, 3])
```

On the other hand, since Keras is a separate repository, having it only use TF 
public API symbol will heavily reduce the chance of breakage caused by relying
on private methods or implementation details. We think this is a item that is
critial to the health of the project. This also allows TF to change internal 
implementation without worrying breaking Keras.

The converted code should look like:

```python
import tensorflow as tf

ones = tf.ones([2, 3])
```

During this conversion, we might notice that certain functions used in Keras are
not public API. Decision should be made on a case by case base for whether:

* Copy the functionality from TF to Keras.
* Replace the usage with other alternative TF public API.
* Make the functionality a new TF public API.

<b>Note that this work can be contributed by the open source community.</b>

### Two stage change process

For any change that is changing both Tensorflow and Keras, they will need to be 
split into two, one as a PR to TF, and the other PR to Keras. Here is 
some common scenario to split the change.

1. Adding a new behavior to Tensorflow, and let Keras rely on it. Note that the 
TF change needs to be submitted first, and keras PR needs to wait for the new TF
nightly PIP. Also note that the rollback of TF PR will cause Keras to break, the
rollback sequence should be PR 33333 and then PR 22222. The Google internal test
for TF should catch the error if the rollback sequence is not correct.

```python
# Existing scenario.
# PR 11111 (2 files updated)
# +++ tensorflow/python/ops/array_ops.py
def some_new_function(inputs):
   ...

# +++ tensorflow/python/keras/layers/core.py

class new_layer(Layer):

  def call(inputs):
     array_ops.some_new_function(inputs)
     ...
```

```python
# New scenario.
# PR 22222 (1 file updated)
# +++ tensorflow/python/ops/array_ops.py
@tf.export('some_new_function')
def some_new_function(inputs):
   ...

==================================
# PR 33333 (1 file updated)
# +++ tensorflow/python/keras/layers/core.py

class new_layer(Layer):

  def call(inputs):
     tf.some_new_function(inputs)
     ...
```

2. Changing an existing behavior of TF function signature.
Note that the PR 22222 needs to submit with both new and old function since 
Google internal CI is still testing from HEAD. The existing_function can be 
deleted after PR 33333 is submitted. Also note that this issue is caused by 
Keras not using public TF API but the implementation details. Moving towards 
public API should reduce the chance of this kind of change.

```python
# Existing scenario.
# PR 11111 (2 files updated)
# tensorflow/python/ops/array_ops.py
<<<
def existing_function(inputs):
    ...
>>>
def new_function(inputs, knob1=False, knob2=1):
    ...
# tensorflow/python/keras/layers/core.py

class existing_layer(Layer):

  def call(inputs):
<<<
    array_ops.existing_function(inputs)
>>>
    array_ops.new_function(
        inputs, 
        knob1=True,
        knob2=3)
```

```python
# New scenario.
# PR 22222 (1 file updated)
# tensorflow/python/ops/array_ops.py
<<<
def existing_function(inputs):
   ...
>>>
def existing_function(inputs):
  return new_function(
    inputs, 
    knob1=False,
    knob2=1)

def new_function(inputs, knob1, knob2=1):
    ...

==================================
# PR 33333 (1 file updated)
# tensorflow/python/keras/layers/core.py
class existing_layer(Layer):

  def call(inputs):
<<<
    array_ops.existing_function(inputs)
     ...
>>>
    array_ops.new_function(
        inputs, 
        knob1=True,
        knob2=3)
```


### Performance Implications
Probably will have some performance implications on python if we change to use
public APIs. Need some benchmark to ensure the status quo.

### Dependencies
Tensorflow PIP package will auto install keras package, which shouldn't cause 
any difference on end user side. Under the hood, Keras will be a different 
package imported by tf_core, same as we do for TF estimator.

### Engineering Impact
* The build and test time locally should be greatly reduced, since compiling TF
is no longer needed, and also Keras is pure python which doesn't need any 
complication.
* The cross boundary change will require some extra handling since the change 
needs to be split into two or more. Same as the rollback.
* Tooling on Github side is not as good as the Google internal tool. This will 
impact the engineer velocity for existing Keras team members.

### Best Practices, Tutorials and Examples
* The new Keras repository should have a new contribution guide about how to 
setup a local test environment and iterate based on that. Similar one in 
tf/addons can be used as an example.
* The existing TF doc needs to be updated to advocate that Keras code now lives
in a different repository, and the new process of sending PR etc.
* When filing an issue, people might need to consider where to send the issue,
eg if it is a Keras issue or an issue caused by TF but surfaced by Keras. The 
different ownership of the repository will also cause difficulties for 
transferring the issue.

### User Impact
* No end user facing change for current TF user, only affecting the developer, 
eg in flight PR during the transition period.
* For current Keras PIP package users, they will get the new TF keras package 
when they update their PIP, which should have more features than the current 
Keras-team version.

## Questions and Discussion Topics

1. Tools for issue tracking: We can't rely on Google internal bug tracking tool 
since it's not publicly visible, also if managing Github issues across the ong 
is hard, we might need to find some alternatives for tracking bugs/features etc.
2. OSS test for TPU related code. Since the TPU is not available during local
test, The verification will have to be done when the PR is mirror to Google 
internal.
3. Transition period for moving the Keras code from tensorflow/tensorflow to
keras-team/keras. All the inflight PR/issue will be affected, either they need
to be copied to keras-team/keras, and if they also touch tensorflow, then they
need to split into two.