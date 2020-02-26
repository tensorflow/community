# Standalone Keras Repository

| Status        | Proposed |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [202](https://github.com/tensorflow/community/pull/202) |
| **Author(s)** | Qianli Zhu (scottzhu@google.com), Francois Chollet (fchollet@google.com) |
| **Sponsor**   | Karmel Allison (karmel@google.com) |
| **Updated**   | 2020-02-05                         |

## Objective

Move the Keras code from the TensorFlow main GitHub repository to its own
repository, with TensorFlow as a dependency.

## Motivation

### TensorFlow API modularity

Currently, Keras has to rely on a number of private TensorFlow APIs. However, a 
litmus test of the quality of the public TensorFlow low-level APIs is that they 
should be strictly sufficient to a higher-level API like Keras.
After splitting the repository, Keras will have to import TensorFlow and 
rely exclusively on public APIs. If Keras still ends up using TensorFlow
private features, it  might be an indication of tight coupling of
implementation details. If certain private features are extensively used,
we might want to consider exposing them  as public low level API.

This design is also aligned with the design for
[Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md), 
which splits the TensorFlow project into smaller components that are not
tightly coupled together.

### Build times

Building the open-source TensorFlow project end-to-end is an extensive exercise. 
With a standard GCP instance, it might take more than one hour to finish the 
whole build process (it might take longer with a Mac laptop). Although the local 
build cache might help speed up the follow-up builds, the initial time cost is 
too high for regular software development workflows. Internally, Google has a
distributed build and caching service, which Googlers heavily rely on,
that can build TensorFlow and run all Keras tests within 5 mins. Sadly,
we can't expose this to external contributors.

Currently, any contribution to Keras code will require building all of
TensorFlow c++ binary, which is quite expensive to do for average users.
Having a separate repository will allow the Keras package to be built
without building TensorFlow. This should greatly improve the 
velocity of open-source developers when they contribute to Keras code.

### Community Benefit

The difficulty of building TensorFlow from scratch in order to make a PR
to Keras code has been a significant source of issues:

* It discouraged contributions, since many external developers couldn't test
their changes and make sure they were correct.
* External developers would send unverified PRs, and Google reviewers spend time 
back and forth, fixing the PR. Sometimes PR is just not moving forward because 
of the lengthy feedback loop.

With the new standalone Keras repository, external contributors should 
experience much shorter turn-around time when building/testing Keras, since they 
don't need to build TensorFlow anymore.
This should have a positive impact on building a vibrant open-source
developer community.

In addition, by getting the Keras team at Google to start developing Keras
using the same public tools and infrastructure as third-party developers,
we make the development process more transparent and more community-oriented.
In the meantime, some of the workload for repository management can be shared
with community so that Keras team member within Google won't be the bottleneck
for all the issues.


## Design Proposal

### New location of the code

GitHub: the code will live at [keras-team/keras](https://github.com/keras-team/keras), 
joining the other Keras SIG projects and replacing the current external Keras 
codebase. `tf.Keras` will also replace Keras on PyPI.

Also considered: `tensorflow/keras`.

Pros:
1. Under the umbrella of Keras SIG, which hosts all other Keras related projects
like keras-application, KerasTuner etc.
1. Lots of existing followers on keras-team, who may not be easily migrated to 
TF project.
1. Can't easily delete keras project, which already have tons of stars and 
incoming reference links. Continued existence of external Keras code will create
confusion ("why is there tensorflow/keras AND keras-team/keras?").

Cons:
1. The repo isn't under the same organization as tensorflow, which makes it hard
to manage issues/PRs and references across the organization.
1. Existing issue/PR under the same org can be transferred easily, but not cross the different org. See [here](https://help.github.com/en/github/managing-your-work-on-github/transferring-an-issue-to-another-repository).

### Source of Truth

TensorFlow uses a Google-internal code repository as its source of truth. Every PR
submitted though GitHub is converted to a Google-internal change first,
submitted through the internal system, and then copied to GitHub as commits.
At the same time, PR is marked as merged with the corresponding commit hash.

Likewise, issue tracking and code review takes place through Google-internal tools.

For Keras, since we are trying to promote community engagement, we hope to use 
GitHub as source of truth. This will have the following implications:

* We expect the majority of the code development/contribution from GitHub
and the dev tools / tests / scripts should focus on the GitHub development use
case. See below for more details.
* Keras CI/presubmit build for the GitHub repo should target a stable PIP 
version of tensorflow package as dependency. It could either be (preferably in
this order):
  * a stable version
  * a release candidate version
  * a `tf-nightly` with explicit version.
Using a nightly version for testing should be motivated by the usage of a API
feature not present in the stable or pre-release version.
Depend on a floating `tf-nightly` could cause CI build to be instable, which has
been observed in other repository 
[like tf-addons](https://github.com/tensorflow/addons/pull/912).
* The Keras code will be mirrored to a Google-internal code repository via
Google-internal tools within a very short time window after each change.
The Google-internal CI tests will run on HEAD for both Keras and TF code.
* The CI build for the repository on GitHub might break when it points to a
new version of `tf-nightly`, if certain behavior has been changed and wasn't
caught by unit tests. We have observed a few similar cases with
[tf/addons](https://github.com/tensorflow/addons).
We hope this can be reduced by stronger unit test coverage by Google internel
systems, when both TF and Keras code are tested at HEAD.
* pip package management. Keras will now follow the `tf-estimator` approach. 
"pip install tensorflow" should also install Keras (from PyPI) as well.
There are more details for the pip package in the
[Improved pip package structure](https://github.com/tensorflow/community/pull/182) RFC.

### Dependency Cleanup

As the high-level API of TensorFlow, Keras should have a direct dependency on
TF low-level APIs, but not the other way around. Unfortunately, there is some
existing reverse logic in the TF code that relies on Keras, which we should 
update/remove when we split the repository.

So far there are about 120 usages for Keras within Tensorflow, the current usage
are:
* Unit tests, which relies on Keras to verify certain behavior of TF, like
distribution strategy, tf.function, and eager context. They should either be 
converted to integration tests, or port the tests to Keras repository.
* `feature_column`, which uses Keras base layer and model.
* Legacy `tf.layers` in v1 API, which uses Keras base layer as base class.
* legacy RNN cells, which uses Keras serialization and deserialization.
* TPU support code does a isinstance() check for `optimizer_v2`.
* TF Lite for keras model saving utils.
* Aliases from tf.losses/metrics/initializers/optimizers in tf.compat.v1.
* Keras symbolic tensor check in the ops library for tf.function.

For usage like tf.layers to keras.layers, it can't be removed due to the API
contract and guarantee. We should use LazyLoader to walk around the cyclic
dependency issue.

```python
from tensorflow.python.util.lazy_loader import LazyLoader

BaseLayer = LazyLoader(
  'BaseLayer', globals(), 'keras.layers.Layer')
if not BaseLayer:
  raise ImportError('Keras is not installed, please pip install keras.')
```

Other dependency should be removed as much as possible, eg move the util/code 
from Keras to TF, or rework the implementation detail. For any of the
dependencies that have to stay, <b>need to use public Keras API only</b>. A
check will also be added to TF to make sure there isn't any dependencies being
added in future for need of Keras.

**Note that this is a key point to prevent Keras accidentally break Tensorflow.**


### Update Keras to only use public TF APIs

The current Keras code will still work if we do e.g.:
```python
from tensorflow.python.ops import array_ops

ones = array_ops.ones([2, 3])
```

However, since Keras is a separate repository, having it only use TF 
public APIs will heavily reduce the chance of breakage caused by relying
on private methods or implementation details. We think this point is
critial to the health of the project. This also allows TF to change internal 
implementation details without worrying about breaking Keras.

The converted code should look like e.g.:

```python
import tensorflow as tf

ones = tf.ones([2, 3])
```

During this conversion, we might notice that certain TF features used in Keras
are not public. A decision should be made on a case-by-case basis:

* Copy the functionality from TF to Keras.
* Replace the usage with another alternative TF public API.
* Make the functionality a new TF public API.

**Note that the open-source community is encouraged to contribute to this effort.**

### Two-stage change process

For any change that is affecting both TensorFlow and Keras, the change
will need to be split into two, one as a PR to the TF repo,
and the other as a PR to the Keras repo. This will introduce overhead and slow
down the change for area's like distribution stragey, and other areas that
might under active development.

With the internal change history between 2019-01-01 and 2020-01-01:
1. There are <b>6756</b> changes submitted to tensorflow/python
2. There are <b>5115</b> changes submitted to tensorflow/python but not 
tensorflow/python/keras.
3. Among the <b>1641</b> changes submitted to tensorflow/keras, <b>1338</b> of 
them change Keras only without touching tensorflow, and 303 of them change both 
Keras and TF.

This means about <b>18.5%</b> change that change Keras will change TF, and 
<b>4.4%</b> change that change TF will touch Keras in the meantime.

Here are some common scenarios:

1. Adding a new feature to TensorFlow, and having Keras rely on it. Note that 
the TF change needs to be submitted first, and the Keras PR needs to wait for 
the new TF nightly to become available on PyPI.

Also note that any rollback of the TF PR will cause Keras to break, the
rollback sequence should be PR 33333 and then PR 22222 (see example below).
The Google-internal test for TF should catch the error if the rollback sequence
is not correct.

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

2. Changing the behavior of an existing TF API.

Note that the PR 22222 needs to be submitted with both the new and old
function since Google internal CI is still testing from HEAD.
The previous function can be 
deleted after PR 33333 is submitted. Also note that this issue is caused by 
Keras not using exclusively public TF API, but relying on TF implementation details.
Moving towards only using public APIs should reduce the likelihood of this kind of issue.

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

### Continuous integration and presubmit test
Due to the fact that Keras code is also being used within Google, apart from 
the normal Github CI (action) tests, We will also run the same tests internally
against HEAD.
1. Github CI and presubmit test will use a stable version of TF binary during
test.
2. Google CI and presubmit test will run against HEAD for both TF and Keras
code. Note that we won't allow submiting Keras code directly to Google 
internal code repo, engineers within Google are still allowed to create changes
internally and run test for it.

The gap between the HEAD version and TF used by Keras should be
as close as possible. Large gap is expect to cause issue for debugging and code
tracing.

There are a few common cases that either CI could break:
1. Github CI could break when the version of TF it depend on is changed. We
think this can be mitigated by pinning Keras to a explicit version of TF, rather
than a floating version like `tf-nightly`. The presubmit test when changing the
verison nubmer should catch this. In the case that a new stable version is
breaking some Keras test, we should

    * Disable the failed tests and move forward to minimize the gap between
    TF HEAD and Keras used version. Report the issue TF team for fix.
    
    * In the case of major breakage, Keras will stay with old version, report
    to TF team and get the issue fixed.

    We hope the second case should be minimized since same tests are running on
    Google CI as well. Any change that might break Keras should be caught
    with internal presubmits.

2. Google CI could break when a submitted PR for Keras is mirrored into Google 
code base. We can't foresee these breakage since we don't run global presumbit
internally for every CL. In the case of breakage, since external contributor
won't notice this, Keras team in Google will:

    * Rollback the original Keras PR if the fault is at Keras side (miss test
    coverage, or bad code interface).
    
    * Update the internal tests to correctly rely on Keras public contract, or
    disable the failed test for the moment.

    We hope both case can be minimized with the internal dependency cleanup as
    well as only relying on public TF API described above.


### Github Repository Migration

* For any open Github PR/issue in Keras-team/keras, it need to be copied to 
Tensorflow if the content is still relevant in Tensorflow. Otherwise it will
be closed as obsolete. We intend to have a clean keras-team/keras repository
before we copy any issue or PR from TF side.
* For any opening PR in Tensorflow for Keras, team will try to merge them as
much as possible before the migration. For any open PR that hasn't been merged,
we will check if it is still relevant/active, and will be copied to 
keras-team/keras.
* The permission of keras-team/keras need to be updated as the codebase is new.
The access level for the repository need to be reestablished.
From least access to most access, the permission levels for an organization 
repository are:

  * Read: Recommended for non-code contributors who want to view or discuss the
  project.
  * Triage: Recommended for contributors who need to proactively manage issues
  and pull requests without write access.
  * Write: Recommended for contributors who actively push to your project.
  * Maintain: Recommended for project managers who need to manage the repository without access to sensitive or destructive actions.
  * Admin: Recommended for people who need full access to the project, including sensitive and destructive actions like managing security or deleting a 
  repository.

Any existing Keras-team active member should get `Triage` level for now, and
more permission will be granted once we identified active contributers. In the
meantime, Keras team in Google will manage the repository initially, and will
share more permissions with the community member.

See more details about the project permission in https://help.github.com/en/github/setting-up-and-managing-organizations-and-teams/repository-permission-levels-for-an-organization.

### Alternative Considered

Split the Tensorflow python and c++ code into separate pip package, eg 
tf-core and tf-python, and tf-python will use a stable version of tf-core 
package to build Tensorflow python. We have to maintain the compatiblity 
between c++ layer and python layer, which is currently quite stable.

    Pros:
    * This should allow us to enjoy speed up of build time for OSS build, since 
    build/test TF won't require building all the c kernel, which is the majority
    of the build time. Internal CI won't be affected since it will always run
    against HEAD.
    * All the python code still lives in one repository, so we don't need to
    split the change into 2 if it changes Keras and TF python at the same time.
    
    Cons:
    * The change that touch both c kernel and TF python code will need to do the
    two stage commit process, if the python change relies on c kernel change.
    * Less motivated to cleanup the cross dependency between TF and Keras since
    it is no longer a required task.
    * With Google internel code repo as source of turth, most of the workflow/
    tools will still be Google centric instead of Github centric.
    * Keras-team/keras code base will still be there if we don't move new TF
    code to it. Having a staled version out there is not ideal, and we should
    really merge them (code/issue/community member) together.

### Performance Implications

There may be some performance implications as we move towards only using
public TF APIs. We need to maintain a benchmark to ensure that there
is no performance regression.

### Dependencies

The TensorFlow pip package will auto-install the Keras package, which shouldn't
make any difference on the end-user side. Under the hood, Keras will be a 
different package imported by `tf_core`, like what we do for TF estimator.

### Developer experience Impact

* The local build and test times should be greatly reduced, since compiling TF
is no longer needed, and Keras is so far a pure-Python project (this might 
change in future when custom c ops are added to Keras).
* Cross-boundary changes will require some extra handling since such changes 
needs to be split into two or more PRs. Same for rollbacks.
* Tooling on the GitHub side (for code review, etc.) is not as good as
Google-internal tools. This may impact the develoment velocity for
Keras team members at Google.

### Best Practices, Tutorials and Examples

* The new Keras repository should have a new contribution guide about how to 
setup a local test environment and iterate based on that. A similar one in 
tf/addons can be used as an example.
* The existing TF docs needs to be updated to highlight that Keras code now lives
in a different repository, with a new process for sending PRs, etc.
* When filing an issue, people might need to consider where to send the issue,
e.g. is it a Keras issue or an issue caused by TF but surfaced by Keras. The 
different ownership of the repository will also cause difficulties for 
transferring the issue.

### User Impact

* No end-user facing change for current TF users; the split would only affect
developers, e.g. in-flight PRs during the transition period.
* For current Keras pip package users, they will get the new TF keras package 
when they update their pip, which should have more features than the current 
Keras-team version.

## Questions and Discussion Topics

1. Tools for issue tracking: we can't rely on Google-internal bug tracking tool 
since it's not publicly visible, also if managing GitHub issues across the orgs 
is hard, we might need to find some alternatives for tracking bugs/features etc.
2. OSS tests for TPU-related code. Since TPUs are not available during local
testing, the verification will have to be done when the PR is mirrored to Google's 
internal systems.
3. Transition period for moving the Keras code from `tensorflow/tensorflow` to
`keras-team/keras`. All in-flight PRs / issues will be affected: they need
to be copied to `keras-team/keras`, or if they also touch TensorFlow, then they
need to split into two.
