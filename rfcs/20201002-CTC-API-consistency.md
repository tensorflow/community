# Improve documentation and change the CTC-related APIs to ensure the consistency of usage

| Status        |  Proposed                                            |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [295](https://github.com/tensorflow/community/pull/295)|
| **Author(s)** | Mingjie Zhou (benjamin_chou@outlook.com), |
| **Sponsor**   | François Chollet (francois.chollet@gmail.com)                 |
| **Updated**   | 2020-10-09                                           |

## Objective
Improve documentation and change the APIs of connectionist temporal classification (CTC) decoders (beam search decoder and greedy decoder) to ensure the consistency with the usage of CTC loss. 
Preserve the same arguments of API as far as possible.


## Motivation
When implementing a model for automatic speech recognition (ASR) and optical character recognition (OCR), one may need to apply CTC at the last layer of their models. 
However, the training behavior is not the same as that of testing.
During training, one needs to apply the CTC loss as a part of the objective function.
During testing, the beam search decoder or greedy decoder will be used to get the labels with the maximum probability. 

The CTC introduces a **blank** label, which one needs to specify the index of it in their model, to separate the different characters. The current APIs of CTC loss and its decoders don't use the same blank index by default and the API documentation doesn't point it out, which may make developers and researchers confused and lead to an invisible mistake.
Besides, the `tf.nn.ctc_loss` doesn't use the same blank index by the internal implementation, which isn't pointed out in the documentation.


## User Benefit
This, apparently, allows developers and researchers to use CTC API easily without making a mistake.

## Design Proposal

### Improve the documentation or improve the internal implementation
[ctc_loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss):
<pre><code>tf.nn.ctc_loss(
    labels, logits, label_length, logit_length, logits_time_major=True,unique=None, 
    blank_index=None, name=None
)</code></pre>
According to API docs, default blank label is 0 rather num_classes - 1, unless overridden by blank_index.
Besides, there is some memory/performance overhead to switching from the default of 0 as an additional shifted copy of the logits may be created.
However, this is not precise description due to the inconsistent internal behavior.

Looking at the original CTCLoss workflow:
1. For sparse tensor, it will use `_ctc_loss_op_standard`(CPU) or `_ctc_loss_op_cudnn`(GPU). The `_ctc_loss_op_standard` by default uses the `num_classes - 1` as the blank index, while the `_ctc_loss_op_cudnn` uses `0` as the blank index. Both implementations have shifted the blank index to the corresponding position before passing arguments to `_ctc_loss_impl`. 

> <pre><code>def _ctc_loss_op_standard(labels, logits, logit_length, logits_time_major,
>                          blank_index):
>part_before = logits[:, :, :blank_index]
>part_after = logits[:, :, blank_index + 1:]
>part_blank = logits[:, :, blank_index:blank_index + 1]
>logits = array_ops.concat([part_before, part_after, part_blank], axis=2)
>...
></code></pre>

><pre><code>def _ctc_loss_op_cudnn(labels, logits, logit_length, logits_time_major,
>                       blank_index):
>  part_before = logits[:, :, :blank_index]
>  part_after = logits[:, :, blank_index + 1:]
>  part_blank = logits[:, :, blank_index:blank_index + 1]
>  logits = array_ops.concat([part_blank, part_before, part_after], axis=2)
>...
></code></pre>

2. For dense tensor, the [dense version](https://github.com/tensorflow/tensorflow/blob/3e63843a2b55c428ab2833b4054c5acfc9827184/tensorflow/python/ops/ctc_ops.py#L939) is implemented by python code with custom gradients.  This dense implementation performs with `blank_index=0` by default. For `blank_index!=0` cases, it would be shifted at the beginning [here](https://github.com/tensorflow/tensorflow/blob/3e63843a2b55c428ab2833b4054c5acfc9827184/tensorflow/python/ops/ctc_ops.py#L1014). 

Here are possible suggestions:
- Improve the `tf.nn.ctc_loss` documentation to make it clear that what kind of situation will use which blank index by the internal behavior. Also, provide some examples.
- Change the internal behavior of `tf.nn.ctc_loss` to make 0 as the blank index of `_ctc_loss_op_standard`(CPU).

### Change the original APIs to the proposed APIs.

**Original APIs**:
Both beam search decoder and greedy decoder perform with `blank index = num_classes - 1` and `logits_time_major=True` by default behavior without an option in the arguments. 
However, the blank index is different from the default in `tf.nn.ctc_loss` and no reminder in the API doc.
This may lead to an invisible mistake when developers use it.


[ctc_beam_search_decoder](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder):
<pre><code>tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1)</code></pre>

[ctc_greedy_decoder](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder): 
<pre><code>tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True)</code></pre>

**Proposed APIs**:

ctc_beam_search_decoder: add arguments (logits_time_major and blank_index)
<pre><code>tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100, top_paths=1, logits_time_major=True, blank_index=None)
</code></pre>


ctc_greedy_decoder:
<pre><code>tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True, logits_time_major=True, blank_index=None)</code></pre>



### Alternatives Considered
Without this RFC, developers and researchers need to either
* set the `blank index = num_classes - 1` in the `tf.nn.ctc_loss`;

or

* roll the row of blank label to the `num_classes - 1`-th row.

What's more, there is no reminder in the API doc that how `tf.nn.ctc_beam_search_decoder` and `tf.nn.ctc_greedy_decoder` perform. 
Developers and researchers could make a mistake unconsciously.


### Performance Implications
* We don’t expect performance impact due to this RFC.

### Dependencies
* This RFC doesn’t add new dependencies to external libraries.


### Engineering Impact
* The impact to binary size / startup time / build time / test times are minimum.
* The TensorFlow team will maintain this code.

### Platforms and Environments
* Platforms: it doesn't change the original support for platforms.

### Best Practices
* This would make the Tensorflow APIs more consistent and easy understanding.

### Tutorials and Examples
Firstly, we import necessary packages.
<pre><code>import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
</code></pre>
Then, we define the `CTCLayer` for training and `CTCDecoder` for prediction.
<pre><code>class CTCLayer(layers.Layer):

    def __init__(self, blank_index, logits_time_major, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.blank_index = blank_index
        self.logits_time_major = logits_time_major

    def call(self,
             labels,
             logits,
             label_length,
             logits_length,
             ):
        ctc_loss = tf.nn.ctc_loss(labels,
                                  logits,
                                  label_length,
                                  logits_length,
                                  logits_time_major=self.logits_time_major,
                                  blank_index=self.blank_index)
        self.add_loss(ctc_loss)

        # Return the logits. This layer only calculates ctc loss.
        return logits


class CTCDecoder(layers.Layer):
    """
    CTC decoder. When beam_width==1, the greedy decoder will be applied.
    """
    def __init__(self, beam_width, blank_index, logits_time_major, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.beam_width = beam_width
        self.blank_index = blank_index
        self.logits_time_major = logits_time_major

    def call(self, logits, logits_length):
        # Here are the proposed APIs in this RFC
        if self.beam_width == 1:
            return tf.nn.ctc_greedy_decoder(logits,
                                            sequence_length=logits_length,
                                            logits_time_major=self.logits_time_major,
                                            blank_index=self.blank_index
                                            )
        else:
            return tf.nn.ctc_beam_search_decoder(logits,
                                                 sequence_length=logits_length,
                                                 beam_width=self.beam_width,
                                                 logits_time_major=self.logits_time_major,
                                                 blank_index=self.blank_index
                                                 )
</code></pre>
Here, we only define a simple model which doesn't include the complicated layers for feature extraction.
<pre><code>def build_training_model(blank_index, num_classes):
    # Suppose we want to build a model for ASR or OCR
    # After going through several layers for feature extraction,
    # we have a sequence of features with shape [batch_size, length, hidden_size]

    features = layers.Input(shape=(None, 64), name='features')  # [batch_size, length, hidden_size]

    # We take dense label as an example. One can use sparse label.
    labels = layers.Input(shape=(None,), name='labels', dtype=tf.int32)

    # For some models, the length can be dynamic and the features is padded with zero.
    # So we have to specify its length as an input
    logits_length = layers.Input(shape=(), name='logits_length', dtype=tf.int32)  # [batch_size]

    # label_length is unnecessary for sparse label
    label_length = layers.Input(shape=(), name='label_length', dtype=tf.int32)  # [batch_size]

    # Only a simple BLSTM model
    outputs = layers.Bidirectional(layers.LSTM(128,
                                               return_sequences=True,
                                               dropout=0.2))(features)
    logits = layers.Dense(num_classes, use_bias=False, name='logits')(outputs)

    ctc_layer = CTCLayer(blank_index=blank_index,
                         logits_time_major=False, name='logtis_with_CTC')
    logits = ctc_layer(labels,
                       logits,
                       label_length,
                       logits_length)

    model = keras.models.Model(inputs=[features, labels, logits_length, label_length],
                               outputs=[logits])

    # In this example, we use batch-major tensor instead of time-major one.
    model.compile(optimizer='Adam')
    model.summary()
    return model


def get_prediction_model(training_model, blank_index, beam_width):
    assert isinstance(training_model, keras.models.Model)

    logits = training_model.get_layer(name='logits').output
    logits_length = training_model.get_layer(name='logits_length').input

    decoder = CTCDecoder(beam_width,
                         blank_index,
                         logits_time_major=False,
                         name='ctc_decoder')

    decoded_outputs = decoder(logits, logits_length)

    predition_model = keras.models.Model(
        inputs=[training_model.get_layer(name='features').input,
                logits_length],
        outputs=[decoded_outputs]
    )

    predition_model.summary()

    return predition_model
</code></pre>

Start to build the models for training and prediction.
<pre><code># set some hyperparameters
NUM_CHARACTERS = 1024
NUM_CLASSES = NUM_CHARACTERS + 1
blank_index = 0
beam_width = 10

# start building model for training and prediction
training_model = build_training_model(blank_index, NUM_CLASSES)

prediction_model = get_prediction_model(training_model, blank_index, beam_width=beam_width)
</code></pre>


### Compatibility
* It shouldn't change any compatibility of what the original APIs have.
