# Title of RFC

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Elena Zhelezina (elena.zhelezina@arm.com) |
| **Sponsor**   | David Rim (davidrim@gmail.com)                 |
| **Updated**   | 2021-09-13                                           |
| **Obsoletes** | TF-RFC it replaces, else remove this header          |

## Objective

Indicate that model saved in tflite file has layers that have been pruned with structural sparsity.

## Motivation

The model can be pruned using structural pruning with TensorFlow Model Optimization Toolkit. This functionality has been added by this PR [Structural pruning](https://github.com/tensorflow/model-optimization/pull/743). The model pruned this way is usually saved into .tflite file. When the model is loaded from .tflite file, all tensors need to be checked to identify whether they are structurally pruned. Currently to get benefits of this, every tensor needs to be checked every time once it is loaded.

We propose to add information whether tensors are structurally pruned to tflite file. This allows to speed-up inference time by moving this check to the conversion time once tflite is created.

## User Benefit

The user can identify which layers of the model have been structurally pruned without doing computationally intensive check every time when the model is loaded from .tflite file.

## Design Proposal

We propose to add an optional structure `StructuralSparsityParameters` to the structure [Tensor] (https://github.com/tensorflow/tensorflow/blame/master/tensorflow/lite/schema/schema.fbs#L195) that provides the parameters of structural pruning. If it is present, then the tensor is structurally pruned.

```
table StructuralSparsityParameters {
  // For a tensor that is structurally pruned we save the type of the pruning.
  // (m, n) means that in a block of n elements, m elements are zeros.
  // If this field is NULL, then the tensor is not sparse.
  sparsity_type_m_by_n:[int];
}
```

During the model conversion, we need to indicate whether tensors should be checked.
For this purpose we introduce a new flag `STRUCTURAL_SPARSITY` to the TensorFlow Lite convertor.

```cpp
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = {tf.lite.Optimize.STRUCTURAL_SPARSITY}
tflite_model = converter.convert()
```

When this flag is set, we check every tensor similar to the [check function](https://github.com/tensorflow/model-optimization/blob/0607d0b056b7281933962a8170432ff78c12c52c/tensorflow_model_optimization/python/core/sparsity/keras/pruning_utils.py#L333) introduced to the TensorFlow Model Optimization Toolkit and store this information as described above. This check will be done for all types of the conversion.

Note that this flag does not provide parameters `(m, n)`, which will be found out during the check function.


### Alternatives Considered

We could introduce the more general structure. For example,

```
// Represents a specific sparsity parameters.
union SparsityDetails {
  StructuralSparsity,
}

table StructuralSparsity {
  // For a tensor that is structurally pruned we save the type of the pruning.
  // (m, n) means that in a block of n elements, m elements are zeros.
  // If this field is NULL, then the tensor is not sparse.
  sparsity_type_m_by_n:[int];
}
```

### Performance Implications
When the flag `STRUCTURAL_SPARSITY` is set, then it is expected to have increase in the conversion time.

### Dependencies
Changes are needed only for TensorFlow Lite.


### Engineering Impact
* The impact to binary size / startup time / build time / test times are minimal and only in the case when the flag is set.
* The TensorFlow team will maintain this code.

### Platforms and Environments
* This change to TensorFlow Lite is platform independent.

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples
* We will provide an end-to-end example and tutorial that will be added to TensorFlow Model Optimization Toolkit. They will demonstrate how to prune the model using structural pruning, convert it to tflite file using the proposed flag.

### Compatibility
How will this proposal interact with other parts of the TensorFlow Ecosystem?

*   **TFLite:** This change works for TFLite.
*   **Distribution strategies:** This change should not impede them.
*   **tf.function:** The change would not interact with tf.function.
*   **GPU/TPU:** This change should not affect them.
*   **SavedModel:** The change does not affect serialization to a SavedModel.

### User Impact
* This change could be introduced as an experimental initially.

## Questions and Discussion Topics

* Any comments on this proposal? Any particular testing should be done for the change of schema.fbs?
