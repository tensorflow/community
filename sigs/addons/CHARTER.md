# SIG Addons

## Objective

TensorFlow natively supports a larger number of operators, layers, metrics, losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot be integrated into core TensorFlow (because they are experimental, or their significance is not yet clear).

This special interest group maintains a repository of bleeding edge contributions that conform to well-established API patterns, but implement new functionality not available in core TensorFlow.

## Scope

This group maintains the [tensorflow/addons](https://github.com/tensorflow/addons) repository. It contains additional functionality which fits the following criteria:

* The functionality is not otherwise available in TensorFlow
* The functionality conforms to an established API pattern in TensorFlow. For instance, it could be an additional subclass of an existing interface (new Layer, Metric, or Optimizer subclasses), or an additional Op or OpKernel implementation.
* Addons have to be compatible with TensorFlow 2.x. 
* The addon conforms to the code and documentation standards defined by the group. These policies are detailed in the project's [README](https://github.com/tensorflow/addons/blob/master/README.md)
* The addon is useful for a large number of users (e.g., an implenentation used in widely cited paper, or a utility with broad applicability)

The group is responsible for reviewing new additions to the repository, including evaluating designs and implementations.

## Membership

Everybody with an interest in helping extend TensorFlow with new types of Ops, Layers, etc. is welcome to join the SIG. To participate, request an invitation to join the mailing list. Maintainer status for the repository will be conferred by consensus of the existing members. Archives of the mailing list are publicly accessible.

## Resources

* SIG Addons mailing list: [addons@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
* Repository maintained by SIG Addons: [github.com/tensorflow/addons](https://github.com/tensorflow/addons)

## Contacts

* Project leads: Sean Morgan [@seanpmorgan](https://github.com/seanpmorgan),
  Yan Facai (颜发才) [@facaiy](https://github.com/facaiy)
* TensorFlow technical contact [@karmel](https://github.com/karmel) - karmel@google.com
* For administrative questions, contact Edd Wilder-James [@ewilderj](https://github.com/ewilderj) - ewj at google

## Archive

* Lead Emeritus: Armando Fandango [@armando-fandango](https://github.com/armando-fandango)

## Code of Conduct

As with all forums and spaces related to TensorFlow, SIG Addons is subject to the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).
