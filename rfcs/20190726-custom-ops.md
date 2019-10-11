# Best practices for custom operations in TensorFlow

| Status        | Accepted                                          |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Alexandre Passos (apassos@google.com |
| **Sponsor** | Karmel Allison (karmel@google.com) |
| **Updated**   | 2019-06-10                                           |

For most of TF’s history, it was very expensive for third-party packages or
libraries to release their own tf operations. This created pressure to put ops
in tf core or in tf contrib, which created some uncertainty around support
stories and backwards compatibility.

Around (but technically not a part of) TF 2.0, however, TensorFlow supports [a
straightforward way for third-party package to build and deploy their own custom
TF ops](https://github.com/tensorflow/custom-op/blob/master/README.md). To
maintain a healthy ecosystem, we recommend the following best practices.

## Experimental ops should live out of tree

Unless some special considerations apply, experimental op development should not
happen inside the core TensorFlow package. Strongly prefer adding experimental
or new operations to libraries and packages downstream from core TensorFlow. Any
op in core TensorFlow is subject to very strict backward and forward
compatibility policies, as TensorFlow is very aggressive about not breaking
existing GraphDefs, and this includes even meant-to-be experimental operations
in the core TensorFlow package.

Once things are no longer experimental, and once the TensorFlow team determines
it is ok with taking responsibility for the code, it’s fine to propose adding a
new version with the final intended interface and implementation to core
TensorFlow. The intermediate states are best explored in another package.

This has many advantages:
 - downstream packages often have a faster release cadence than core TensorFlow
 - each downstream package can choose its own backward and forward compatibility
   processes, allowing fine-grained trade-offs between velocity and stability

## Out-of-tree ops must be namespaced

Since an op’s name uniquely identifies it, different TF packages should ensure
their op names are globally unique across the entire TF ecosystem. To do so,
prepend the package’s name to the op’s name and separate with a ‘>’. An op named
“MatMul” inside the “tensorflow_addons” package should be named “Addons>MatMul”,
for example.

The string used for a package’s component name is any valid op name, but should
be unique to the package. This allows different packages to experiment with ops
without needing a central coordinator to assign unique operation names. Failing
to use unique names will mean two packages are potentially incompatible.

If a third-party-developed operation is to be integrated in TensorFlow core, it
should be renamed to have no prefix, creating a new op name, and removing any
risk of internal and external versions silently diverging.


