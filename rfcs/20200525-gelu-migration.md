# Migrate gelu activation from TensorFlow Addons to TensorFlow Core

| Status      | Proposed (Waiting for Sponsor)                                                                                           |
| :---------- | :------------------------------------------------------------------------------------------------- |
| **RFC #**   | [252](https://github.com/tensorflow/community/pull/252) |                                       |
| **Authors** | Tzu-Wei Sung (@WindQAQ) & Sean Morgan (@seanpmorgan) |
| **Sponsor** | TBD   |
| **Updated** | 2020-06-02 |
| **Sponsorship Deadline** | 2020-07-17 (45 Days after submission)  |

## Rationale for Migration
* [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415.pdf) cited 600+ times
* Used in BERT and other influential architectures
* Multiple approvals from TF side: 
    * https://github.com/tensorflow/tensorflow/pull/33945#issuecomment-617832325
    * https://github.com/tensorflow/tensorflow/issues/32783#issuecomment-537284266

## Historical Information
* Have there been signifiant issues reported to Addons that need to be adressed?
    * Only ABI incompatibilities for the custom-op (not an issue if built along with core TF)
* When was it implemented in Addons?
    * C++ custom-op added **2019-08-2019 (TFA 0.5.0)**
    * Python composite op added **2020-02-26 (TFA 0.9.0)**
* We have [performed benchmarking of the GELU activation](https://colab.research.google.com/drive/1rLb4EuydbFg9PbhboXhCDqopcl6BmphG#scrollTo=0GL2x2S4zxW3) 
which shows it may be beneficial to retain the custom-ops, but the maintence burden has grown 
to much for us to continue to support it in Addons.
* This migration is long over-due but we've struggled with finalizing the migration process.

## Implementation Details
* Link to implementation in Addons:
    * Python: https://github.com/tensorflow/addons/blob/r0.10/tensorflow_addons/activations/gelu.py
    * C++ : https://github.com/tensorflow/addons/blob/r0.10/tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op.h
* Does this include custom-op kernels?
    * Yes, but currently proposing to just migrate the python composite op. This may 
    change with discussion in the RFC.
    * Are they CPU/GPU/TPU compatible?
        * CPU/GPU compatible. No support for TPU.
* What is the pytest coverage of the addon?
    * `tensorflow_addons/activations/gelu.py 89%`
## Changes to Implementation (If Needed)
```
def gelu(x: types.TensorLike, approximate: bool = True) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    if approximate:
        pi = tf.cast(math.pi, x.dtype)
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))
```
The above implementation would only bring over the python composite op. Since there is 
[no way for us to build tfxla kernels](https://github.com/tensorflow/tensorflow/pull/33945#issuecomment-617842977) 
we had no support for TPUs in Addons.  [There were comments](https://github.com/tensorflow/tensorflow/pull/33945#issuecomment-625380208) 
about using a "selector", but we would need guidance on how to implement that.

We may also want to discuss the `approximate` bool and if it should be included in the 
upstream version.


## Transition Plan
* The activation would land in [nn_ops.py](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow//python/ops/nn_ops.py), [keras activations](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/activations.py),
 and possibly in [keras advaced_activation layers](https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/layers/advanced_activations.py)
* No planned changes to the parameter signatures at this time
* Addons would deprecate our activation and make a call to the core functionality.
* After merging to TF Core:
    * Consolidate/remove https://github.com/tensorflow/models/blob/r2.2.0/official/modeling/activations/gelu.py
    * Consolidate/remove https://github.com/tensorflow/models/blob/r2.2.0/official/modeling/activations/gelu_test.py
    * Consolidate/remove https://github.com/tensorflow/models/blob/r2.2.0/official/nlp/xlnet/xlnet_modeling.py#L29

## Relevant GitHub Issues
* https://github.com/tensorflow/tensorflow/pull/33945
* https://github.com/tensorflow/addons/issues/550
* https://github.com/tensorflow/tensorflow/issues/32783

## Questions and Discussion Topics
* Whom from the TF core team would sponsor this migration and ownership of the API?
* Is it worth bringing over the custom-op kernels for CPU/GPU?

## Final Decision
TBD
