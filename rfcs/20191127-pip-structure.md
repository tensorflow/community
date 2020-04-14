# Improved pip package structure

| Status        | [Implemented](https://github.com/tensorflow/tensorflow/commit/5c00e793c61860bbf26778cd4704313e867645be)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [182](https://github.com/tensorflow/community/pull/182)|
| **Author(s)** | Anna Revinskaya (annarev@google.com)                 |
| **Sponsor**   | Alex Passos (apassos@tensorflow.org)                 |
| **Updated**   | 2020-02-04                                           |

## Objective

We propose to simplify TensorFlow pip package structure to enable IDE features such as autocomplete, jump-to-definition and quick-documentation.

## Motivation

### Current package structure
TensorFlow package structure has grown quite complex over time as we started to support multiple versions (1.x and 2.x) and import external sub-packages (such as tensorflow\_estimator and tensorboard). This complexity is expected to grow if we split out more components into separate pip packages.

Sources of complexity:

* Versioning: tensorflow\_core API lives under *_api/v1* or *_api/v2* directory depending on the version.
* Virtual pip package: Installing TensorFlow actually installs 2 directories: *tensorflow/* and *tensorflow\_core/* under *site-packages/*. TensorFlow code lives under *tensorflow\_core/*. TensorFlow uses lazy loading to import everything from *tensorflow\_core/* to *tensorflow/*. Two-directory structure helps work-around circular imports caused by tensorflow\_estimator.

Outline of the current structure:
```
tensorflow
    __init__.py (contains "from tensorflow_core import *")

tensorflow_core
    python/...
    lite/...
    _api/v2
        __init__.py
        audio/__init__.py
        autograph/__init__.py
        ...
```

### Rationale behind current package structure
#### Multiple version support
To prepare for TensorFlow 2.0 launch, we added a way to build two versions: 1.x and 2.x. Each version has its own respective genrule that outputs file for 1.x or 2.x since API modules are different (for e.g. *tensorflow/manip/\_\_init\_\_.py* only exists in 1.x and not 2.x API). Now, bazel does not allow two genrules to output files to the same directory. Therefore, we have *_api/v1/* and *_api/v2/* subdirectories.

Note that we could still place the API directly under *tensorflow/* in the pip package since a pip package contains a single version of TensorFlow. This option became out of reach when *tensorflow/contrib/lite/* was migrated to *tensorflow/lite/*. Now *tensorflow/lite/* API directory would conflict with *tensorflow/lite/* source directory if the API was under *tensorflow/* instead of *_api/vN/*.

#### Circular dependencies
Estimator depends on TensorFlow. At the same time, TensorFlow includes estimator as a part of its API. This creates a cycle.

![alt_text](https://github.com/annarev/community/blob/pip_structure_rfc/rfcs/20191127-pip-structure/circular_dependency.png "Circular dependency
between TensorFlow and Estimator.")

#### Metapackage vs base package plans
[Modular TensorFlow
RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md) proposes to keep two pip packages:
tensorflow-base would only contain core TensorFlow (for e.g. no estimator).
TensorFlow Metapackage would be a thin package defining composition of TensorFlow which includes base, estimator, keras and tensorboard.
Note that this 2-package approach is not implemented yet. However, its proposal demonstrates how keeping a virtual pip package could be beneficial in the future.

![alt_text](https://github.com/annarev/community/blob/pip_structure_rfc/rfcs/20191127-pip-structure/modular_structure.png "Proposed modular TensorFlow structure.")

Current structure looks more like this (except *tensorflow/* and *tensorflow\_core/* are directories as opposed to separate pip packages) and meant to be the first step towards structure above:

![alt_text](https://github.com/annarev/community/blob/pip_structure_rfc/rfcs/20191127-pip-structure/current_structure.png "Current TensorFlow structure.")

### Current state of IDE code features

#### PyCharm 2019.1.1

* Autocomplete:
  * Works in most cases after switching to use relative imports.
  * Doesn’t work for tf.compat.v1.keras and tf.compat.v2.keras.
  * Doesn’t work for keras if importing it using from import (i.e. `from tensorflow import keras`).
* Jump-to-definition doesn’t work.
* Quick documentation doesn’t work.

#### PyCharms with 2019.3 EAP build 193.3793.14
Latest version of PyCharms added [custom handling for tensorflow](https://github.com/JetBrains/intellij-community/blob/0a08f8212351ee84d602cdc5547f038ce0df79fd/python/src/com/jetbrains/tensorFlow/PyTensorFlow.kt)
* Autocomplete works in most cases.
* Doesn’t work for keras if importing it using from import (i.e. `from tensorflow import keras`).
* Jump-to-definition works.
* Quick documentation works.

#### VS Code 1.40 (October 2019 release)
* Autocomplete:
  * Works in most cases.
  * Doesn’t work for `tf.estimator` or `tf.keras`.
  * Doesn’t work for `tf.compat.v1.keras` and `tf.compat.v2.keras`.
  * Doesn’t work for keras if importing it using from import (i.e. `from tensorflow import keras`).
* Jump-to-definition doesn’t work.
* Quick documentation doesn’t work.


## User Benefit

TensorFlow package structure creates difficulties for those who use IDEs.
Autocomplete, quick documentation and jump-to-definition features often rely on
module structure matching directory structure. For example, TensorFlow code uses
`from tensorflow.foo` imports but lives under tensorflow\_core package. Simplifying
package structure would improve productivity for TensorFlow users.

## Design Proposal

The best way I can think of to fix the autocomplete issues is to make our package structure as clean as possible. In this case, autocomplete will work out of the box.

### Short term: Remove virtual pip package

Primary purpose of keeping the virtual pip package is to workaround circular
estimator imports. Alternatively, we can resolve this issue by lazy loading
estimator.

Estimator import in root *\_\_init\_\_.py* file:
```python
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader
estimator = _LazyLoader(
    "estimator", globals(),
    "tensorflow_estimator.python.estimator.api._v2.estimator")
setattr(_current_module, "estimator", estimator)
```

Lazy loading by itself would mean that we no longer have autocomplete for estimator. As a workaround, we can import estimator without lazy loading if `typing.TYPE_CHECKING` is `True`.

After building a pip package with this change all of the following work in PyCharms (both released and EAP) and VS Code:

* jump-to-definition
* quick documentation
* autocomplete for `compat.v1.keras`, `compat.v2.keras`
* autocomplete for keras when using from tensorflow import keras
* ...basically any import I tested works with autocompletion

To support the TensorFlow Metapackage plans we could add a new pip package that specifies dependencies on tensorflow, tensorflow\_estimator, tensorboard, etc.. Its sole purpose would be to get all dependencies installed.

![alt_text](https://github.com/annarev/community/blob/pip_structure_rfc/rfcs/20191127-pip-structure/new_modular_structure.png "New proposed modular TensorFlow structure.")

### Long term (optional): Import from external package directly
Short term would fix IDE issues, but the package structure is still not as clean as it could be. We resolve cycles with lazy loading but it would be even better not to have this circular structure at all.

Therefore, I propose that we don’t import external packages into TensorFlow 3.0. Users who want to use estimator, tensorboard summaries or keras could import them separately:

Current code that looks like:
```python
import tensorflow as tf

tf.estimator
tf.keras
tf.summary
```

Would be changed to:
```python
import tensorflow as tf
import tensorflow_estimator as estimator
import keras
from tenosorboard import summaries
```

Rationale for this change:

* One way dependencies (estimator depends on tensorflow and not vise-versa).
* Minimal overhead for users. Adding an extra import is easy.

Note that this change cannot be done in TensorFlow 2.x due to API guarantees. Also, accessing these packages from `tf.` would match familiar workflows. Therefore, we can keep `tf.estimator`, `tf.keras` (once it is moved out of TensorFlow), `tf.summary` available as an alternative to importing pip package directly. This would require some work to make sure these packages contain the right API (for e.g. tensorflow\_estimator.estimator currently always contains V1 API).


### Alternatives Considered
Alternatively, we could solve IDE autocomplete issues by changing all imports in
TensorFlow to import from `tensorflow_core` instead of `tensorflow`.

#### Advantages:

* Keep supporting external libraries included as a sub-namespace, for e.g.
`tf.estimator`.

#### Disadvantages:

* This is a more invasive change since it requires updating every Python file in TensorFlow.
It would also mean that external packages such as `tensorflow_estimator` need to
use imports of the form `from tensorflow_core` instead of `from tensorflow`.

The main proposal in this document seems simpler to me (it removes complexity
instead of adding it) and therefore preferred.

### Performance Implications
I am not expecting major performance changes since this is just a package
structure proposal.

### Dependencies
This proposal does not add new dependencies. The rest of the proposal largely
describes how we plan to handle dependencies.

### Engineering Impact
We don't expect changes to binary size / startup time / build time / test time.

### Platforms and Environments
This should work on all platforms and we will test it to make sure.

### Best Practices, Tutorials and Examples
There are no user-visible changes other than fixes to enable IDE features.

### Compatibility
Short term proposal does not have any compatibility concerns. Long term,
however, proposes to remove `tf.estimator`, etc.. which is not a backwards
compatible change. We can only make this change at the next major release.

### User Impact
There are no user-visible changes other than fixes to enable IDE features.
