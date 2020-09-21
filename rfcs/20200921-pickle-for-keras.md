# Support for Pickle, Python's serialization protocol

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [286](https://github.com/tensorflow/community/pull/286) |
| **Author(s)** | Adrian Garcia Badaracco ({firstname}at{firstname}gb.com), Scott Sievert (tf-rfc@stsievert.com)  |
| **Sponsor**   | Mihai Maruseac (mihaimaruseac@google.com)                 |
| **Updated**   | 2020-09-21                                           |

## Objective

Implement support for Pickle, Python's serialization protocol within Keras.

## Motivation

> *Why this is a valuable problem to solve? What background information is
> needed to show how this design addresses the problem?*

The specific motivation for this RFC: we want to use Keras models in Dask-ML's
and Ray's hyperparameter optimization. More generally, support for serialization
with the Pickle protocol will enable:

* Using Keras with other parallelization libraries like Python's
  `multiprocessing`, Dask, Ray or IPython parallel.
* Saving Keras models to disk with custom serialization libraries like Joblib
  or Dill. This is common when using a Keras model as part of a Scikit-Learn
  pipeline or with their hyperparameter searches.
* Copying Keras models with Python's built-in `copy.deepcopy`.

Supporting Pickle will enable wider usage in the Python ecosystem because
Python's ecosystems of libraries depend strongly on the presence of protocols.
Without these protocols, it's necessary for each library to implement a custom
serialization method for every other library. For example, Dask Distributed has
a custom serialization method for Keras at [distributed/protocol/keras.py].
See "[Pickle isn't slow, it's a protocol]" for more detail (notably, this post
focuses on having an efficient Pickle implementation for PyTorch).

[distributed/protocol/keras.py]:https://github.com/dask/distributed/blob/73fa9bd1bd7dcb4ceed72cdbdc6dd4b92f887521/distributed/protocol/keras.py

This request is *not* advocating for use of Pickle while saving or sharing
Keras models. We believe the efficient, secure and stable methods in TF should
be used for that. Instead, we are proposing to add a Pickle implementation to
support wider usage in the Python ecosystem.

[Pickle isn't slow, it's a protocol]:https://blog.dask.org/2018/07/23/protocols-pickle

> *Which users are affected by the problem? Why is it a problem? What data
> supports this? What related work exists?*

Users trying to use distributed systems (e.g, Ray or Dask) with Keras are
affected. In our experience, this is common in hyperparameter optimization.  In
general, having Pickle support means a better experience, especially when using
Keras with other libraries. Briefly, implementation of this RFC will make the
following possible:

* Saving a Scikit-Learn pipeline to disk if it includes a Keras model
* Using custom parallelization like Joblib or Dask.

More use cases and examples are give in "User Benefit."

Related work is in [SciKeras], which brings a Scikit-Learn API
to Keras. Pickle is relevant because Scikit-Learn requires that estimators must be able to be pickled ([source][skp]).
As such, SciKeras has an implementation of `__reduce__`, which is also in
[tensorflow#39609].
  
[dask-ml#534]:https://github.com/dask/dask-ml/issues/534
[SO#51110834]:https://stackoverflow.com/questions/51110834/cannot-pickle-dill-a-keras-object
[SO#54070845]:https://stackoverflow.com/questions/54070845/how-to-pickle-keras-custom-layer
[SO#59872509]:https://stackoverflow.com/questions/59872509/how-to-export-a-model-created-from-kerasclassifier-and-gridsearchcv-using-joblib
[SO#37984304]:https://stackoverflow.com/questions/37984304/how-to-save-a-scikit-learn-pipline-with-keras-regressor-inside-to-disk
[SO#48295661]:https://stackoverflow.com/questions/48295661/how-to-pickle-keras-model
[skper]:https://scikit-learn.org/stable/modules/model_persistence.html#persistence-example
[TF#33204]:https://github.com/tensorflow/tensorflow/issues/33204
[TF#34697]:https://github.com/tensorflow/tensorflow/issues/34697

[tensorflow#39609]:https://github.com/tensorflow/tensorflow/pull/39609
[SciKeras]:https://github.com/adriangb/scikeras
[skp]:https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/estimator_checks.py#L1523-L1524

<!--
StackOverflow questions where `Model.save` would not work:

* [SO#40396042](https://stackoverflow.com/questions/40396042/how-to-save-scikit-learn-keras-model-into-a-persistence-file-pickle-hd5-json-ya)
  
Examples that could be resolved using `Model.save` (but the user tried pickle first):

* [SO #51878627](https://stackoverflow.com/questions/51878627/pickle-keras-ann)
-->

## User Benefit

> How will users (or other contributors) benefit from this work? What would be the headline in the release notes or blog post?

One blog post headline: "Keras models can be used with the advanced
hyperparameter optimization techniques found in Dask-ML and Ray Tune." This has
already been mentioned in "Framework support" of [a Dask blog post][dbp]
comparing Dask-ML's hyperparameter optimization with Ray's tune-sklearn.

[dbp]:https://blog.dask.org/2020/08/06/ray-tune#framework-support

Users will also benefit with easier usage; they won't run into any of these
errors:

* People try to save Scikit-Learn meta-estimators with Keras components using
  the serialization libraries Joblib or Dill. 
  This fails because Keras models can not be serialized without a custom
  method. Examples include [SO#59872509], [SO#37984304] and
  [SO#48295661], and [SO#51110834].
* Using custom parallelization strategies requires serialization support through
  Pickle; however, many parallelization libraries don't
  special case Keras models (e.g, Joblib). Relevant errors are most common in hyperparameter
  optimization with Scikit-Learn's parallelization through Joblib
  ([TF#33204] and [TF#34697]) or parallelization through Dask ([dask-ml#534]).
* Lack of Pickle support can complicate saving training history like in
  (the poorly asked) [SO#54070845].

This RFC would resolve these issues.

## Design Proposal

We propose implementing the Pickle protocol using the existing Keras
saving functionality as a backend. For example, adding pickle support to TF Metrics
is as simple as the following:

``` python
# tensorflow/python/keras/metrics.py

@keras_export('keras.metrics.Metric')  # line 80
@six.add_metaclass(abc.ABCMeta)
class Metric(base_layer.Layer):
    ...

  def __reduce__(self, protocol):
    # the deserialized/serialize functions are defined in this file
    return deserialize, (serialize(self),)
```

This implementation adds support for the Pickle protocol, which supports serialization
to arbitrary IO, either memory or disk. The `__reduce__` special method can return
the string that would have been written to disk and the function to load that string into memory ([docs][reduce_docs]).

[reduce_docs]:https://docs.python.org/3/library/pickle.html#object.__reduce__

For `tf.keras.Model`, we can use `SaveModel` as the backend for `__reduce__`:

``` python
# tensorflow/python/keras/engine/training.py
...
from tesorflow.python.keras.models import load_model

class Model(base_layer.Layer, version_utils.ModelVersionSelector):  # line 131
  ...

  def __reduce__(self, protocol):
    temp_ram_location = f"ram://tmp/saving/{id(self)}"
    self.save(temp_ram_location)
    b = tf.io.gfile.read_folder(temp_ram_location)
    return self._reconstruct_pickle, (np.asarray(memoryview(b)), )

  @classmethod
  def _reconstruct_pickle(cls, obj):
    temp_ram_location = f"ram://tmp/saving/{id(obj)}"
    tf.io.gfile.write_folder(temp_ram_location, b)
    return load_model(temp_ram_location)
```

This almost exactly mirrors the PyTorch
implementation of Pickle support in [pytorch#9184]
as mentioned in "[Pickle isn't slow, it's a protocol]."
In addition, small augmentations to TensorFlow's IO module will be required (as discussed in [tensorflow#39609]).

By wrapping the pickled object within a Numpy array, pickling will support
pickle protocol 5 for zero-copy pickling. This provides an immediate
performance improvement for many use cases.

[pytorch#9184]:https://github.com/pytorch/pytorch/pull/9184

### Alternatives Considered

Of course, one method is to ask users to monkey-patch Keras models themselves.
This would hold for libraries too. Clearly, this is unreasonable. Regardless,
some libraries like Dask Distributed have already implemented custom serialization
protocols ([distributed/protocol/keras.py]).

#### Other pickle implementations

The Pickle protocol supports two features:

1. In-memory copying of live objects: via Python's `copy` module. This falls back to (2) below.
2. Serialization to arbitrary IO (memory or disk): via Python's `pickle` module.

This proposal seeks to take the conservative approach at least initially and
only implement (2) above since (1) can always fall back to (2) and using only
(2) alleviates any concerns around references to freed memory in the C++
portions of TF and other such bugs.

This said, for situations where the user is making an in-memory copy of an object and it might
even be okay to keep around references to non-Python objects, a separate approach that optimizes
(1) would be warranted. This RFC does not seek to address this problem. Hence
this RFC is generally not concerned with:

* Issues arising from C++ references. These cannot be kept around when
  serializing to a binary file stream.
* Performance of the serialization/deserialization.

### Performance Implications

* The performance should be the same as the underlying backend that is already
  implemented in TF.
* For cases where the user was going to pickle anyway, this will be faster
  because it uses TF's methods instead of letting Python deal with it naively.
* Tests will consist of running `new_model = pickle.loads(pickle.dumps(model))`
  and then doing checks on `new_model`.

### Dependencies

> Dependencies: does this proposal add any new dependencies to TensorFlow?

No

> Dependent projects: are there other areas of TensorFlow or things that use
  TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects?

This should not affect those libraries. It will affect libraries
further downstream like Dask-ML and Ray Tune.

### Engineering Impact

> Do you expect changes to binary size / startup time / build time / test
  times?

No

> Who will maintain this code? Is this code in its own buildable unit? Can this
  code be tested in its own? Is visibility suitably restricted to only a small
  API surface for others to use?

This code depends on existing Keras/TF methods. This code will not break
presuming they are maintained (the new API surface area is very small).

### Platforms and Environments

> Platforms: does this work on all platforms supported by TensorFlow? If not,
  why is that ok? Will it work on embedded/mobile? Does it impact automatic
  code generation or mobile stripping tooling? Will it work with transformation
  tools?

Yes, as long as a Python backend is available.

> Execution environments (Cloud services, accelerator hardware): what impact do
  you expect and how will you confirm?

We don't see any impact.

### Best Practices

> Does this proposal change best practices for some aspect of using/developing
  TensorFlow? How will these changes be communicated/enforced?

No

### Tutorials and Examples

There are plenty of examples of how this can and would be used within all of the issues above, in addition to the linked notebook
([link again](https://colab.research.google.com/drive/14ECRN8ZQDa1McKri2dctlV_CaPkE574I?authuser=1#scrollTo=qlXDfJObNXVf)) which has
end to end implementations and tests for all of this.

### Compatibility

> Does the design conform to the backwards & forwards compatibility
  [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?

Yes

> How will this proposal interact with other parts of the TensorFlow Ecosystem?*

It should have no immediate impact on other parts of the TF ecosystem.

> How will it work with TFLite?

N/A

> How will it work with distribution strategies?

This enables use of other serialization libraries, which might enable support for other distribution strategies.

> How will it interact with tf.function?

N/A

> Will this work on GPU/TPU?

N/A

> How will it serialize to a SavedModel?

Not applicable, and almost a circular question.

### User Impact

> What are the user-facing changes? How will this feature be rolled out?

There are no user-facing changes: this is a backend change to private methods.

Rolling out only involves testing. It will not require any documentation
changes to advertise this features: `Model.save` should still be used for users
simply trying to save their model to disk.

## Questions and Discussion Topics

> Seed this with open questions you require feedback on from the RFC process.
