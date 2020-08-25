This RFC will be open for comment until Friday, August 10th, 2018.

# RFC: End-to-end text preprocessing with TF.Text

| Status        | (Proposed)                                                            |
:-------------- |:----------------------------------------------------------------------|
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Terry Huang (Google)                                                  |
| **Sponsor**   | Xiaodan Song(Google), Greg Billock (Google), Mark Omernick (Google)   |
| **Updated**   | 2020-08-24                                                            |

## Objective

Introduce a new suite of text preprocessing APIs to empower the end-to-end transformation of text to model inputs in TensorFlow, expressive enough for pre-training and downstream tasks of Transformer Encoders such as BERT, ALBERT, XLNet, etc.


## Motivation

Text preprocessing is the end-to-end transformation of raw text into a model’s integer inputs. While in the past models may perform a few preprocessing steps (such as tokenization or string normalizations), we have seen an increasing interest to extract and pretrain models with unsupervised tasks such as BERT’s masked language model (MLM), next sentence prediction (NSP), ALBERT’s sentence order prediction (SOP), etc.

NLP models are often accompanied by several hundreds (if not thousands) of lines of Python text preprocessing code. It becomes increasingly challenging to manage the preprocessing logic without introducing training/serving skew bugs, especially when the model is interacted in multiple stages (e.g. pretraining, fine-tuning, evaluation, inference). For example, using different hyperparameters, tokenization, string preprocessing algorithms or simply packaging model inputs inconsistently could yield hard-to-debug, and disastrous effects to the model. Many of the problems can be alleviated by packaging the preprocessing directly with the model.

Additionally, many existing Python methods write out processed outputs to files on disk and construct TF input pipelines to consume said preprocessed data. This incurs an additional read/write cost and is inconvenient for dynamically changing text preprocessing decisions. Perhaps more importantly, it does not align well with exporting a self-contained model to TF Serving that goes from string input to prediction outputs.

## User Benefit

The proposed new set of text preprocessing APIs will allow users to:
- ***Assemble TF input pipelines w/ reusable, well-tested, standard building blocks*** that transform their text datasets into model inputs. Being part of the TF graph also enables users to make preprocessing choices dynamically on the fly.
- ***Drastically simplify their model’s inputs to just text.*** Users will be able to easily expand to new datasets for training, evaluation or inference. Models deployed to TF Serving can start from text inputs and encapsulate the details of preprocessing.
- ***Reduce risks of training/serving skew*** by giving models stronger ownership of the entire preprocessing and postprocessing process.
- ***Reduced complexity and improved input pipeline efficiency*** by removing an extra read & write step to transform their datasets and improved efficiency w/ vectorized mapping by processing inputs in batches.


## Design Proposal

We propose a new suite of text preprocessing APIs to empower the end-to-end transformation of text to model inputs in TensorFlow. We propose to provide:
a suite of Python functions and classes for defining the APIs of reusable building blocks and the actual tensor-to-tensor computations (incorporating the BertTokenizer class, which has existed since 2019).
a parallel suite of Keras Layer classes to make the same transformations immediately usable in Keras. These classes have enough responsibilities of their own (serialization, resource handling) to pull their weight.
All components are compatible with a tf.data input pipeline for training/evaluation and can be packaged into a SavedModel.

In particular, we propose the following API additions to TF.Text:
- `Splitter`
  - `RegexSplitter`
  - `StateBasedSentenceBreaker`
- `Trimmer`
  - `WaterfallTrimmer`
  - `RoundRobinTrimmer`
- `ItemSelector`
  - `RandomItemSelector`
  - `FirstNItemSelector`
- `MaskValuesChooser`
- `mask_language_model()`
- `combine_segments()`
- `pad_model_inputs()`

## Input Pipeline

Say that we have tensorflow.Examples with a single “text” feature, such as:

```
features {
  feature {
    key: "text"
    bytes_list {
        value: ["Shall I tell you a lie? I do despise a liar as I do despise one that is false, or as I despise one that is not true. The knight, Sir John, is there; and, I beseech you, be ruled by your well-willers. I will peat the door for Master Page. Knocks"]
      }
    }
  }
}
```

For training, our goal is to empower users to construct a tf.data pipeline that invokes a custom preprocessing map() function.

```
    feature_spec = {
      "text": tf.FixedLenFeature([1], tf.string),
    }
    dataset = tf.data.experimental_v2.make_batched_features_dataset(
      doc_batch_size, feature_spec, …)
    dataset = dataset.map(
        functools.partial(bert_pretrain_preprocess, vocab_lookup_table))
    # … continue to build the dataset
```

Users will be able to use the APIs as building blocks to construct an end-to-end preprocessing function to transform raw text into the model’s input. The following examples shows a preprocessing function that tokenizes, truncates, combines segments and extracts labels for BERT’s NSP task, and MLM task:

```
import tensorflow_text as text

def bert_pretrain_preprocess(vocab_lookup_table, features):
  # Input is a string Tensor of documents, shape [num_docs].
  input_text = features["text"]

  # Split sentences on new line character.
  # Output has shape [num_docs, (num_sentences_per_doc)].
  sentence_breaker = text.RegexSplitter(split_regex="\n")
  sentences = sentence_breaker.split(input_text)

  # Extract next-sentence prediction labels and segments.
  # Output has two segments of shape [num_sentences] each.
  # The rest of this function keeps using this data-dependent batch size.
  # The output gets rebatched to the fixed training batch size later on.
  segment_b = _get_next_sentence_labels(sentences)
  segments = [sentences, segment_b]

  # Tokenize segments to shape [num_sentences, (num_words)] each.
  tokenizer = text.BertTokenizer(
      vocab_lookup_table,
      token_out_type=dtypes.int64)
  # Flatten [num_sentences, (num_words), (num_wordpieces)] -> [num_sentences, (num_tokens)]
  # to make tokens (not words) the unit of truncation and masking.
  segments = [tokenizer.tokenize(s).merge_dims(-2, -1) for s in segments]

  # Truncate inputs to a maximum length.
  trimmer = text.RoundRobinTrimmer(...)
  trimmed_segments = trimmer.trim(segments)
  
  # Combine segments, get segment ids and add special tokens.
  segments_combined, segment_ids = text.combine_segments(truncated_segments)

  # Apply dynamic masking task.
  masked_input_ids, masked_lm_positions, masked_lm_ids = text.mask_language_model(
    segments_combined,
    RandomItemSelector(...),
    MaskValuesChooser(...)
  )

  model_inputs = {
      "input_word_ids": masked_input_ids,
      "input_type_ids": segment_ids,
      "masked_lm_positions": masked_lm_positions,
      "masked_lm_ids": masked_lm_ids,
  }
  padded_inputs_and_mask = tf.nest.map_structure(
    tf_text.pad_model_inputs, model_inputs, ...)
  model_inputs = {
      k: padded_inputs_and_mask[k][0] for k in padded_inputs_and_mask
  }
  model_inputs["masked_lm_weights"] = (
      padded_inputs_and_mask["masked_lm_ids"][1])
  model_inputs["input_mask"] = padded_inputs_and_mask["input_word_ids"][1]

  return {
    "input_ids": input_ids,
    "input_mask": input_mask,
    "segment_ids": segment_ids,
    "masked_lm_positions": masked_lm_positions,
    "masked_lm_ids": masked_lm_ids,
    "masked_lm_weights": masked_lm_weights
  }
```

The output of the tf.data pipeline is integer inputs transformed from the raw text and can be fed directly to the model (e.g., bert_pretraining model in model_garden):

```
{
'input_ids': [
  [ 101, 14962, 10944, 102, 20299, 10105, 102, 0 ],
  [ 101, 12489, 102, 10105, 31877, 10155, 15687, 102],
],
 'segment_ids': [
  [0, 0, 0, 0, 1, 1, 1, 0]
  [0, 0, 0, 0, 1, 1, 1, 1]
],
'input_mask': [
  [1, 1, 1, 1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1, 1, 1, 1],
],
'masked_lm_positions': [
   [ 7,  8,  0,  0],
   [ 3,  5,  0,  0]
],
'masked_lm_ids': [
   [ 10124,  10271, 0, 0],
   [ 10124, 136, 0, 0],
 ],
 'masked_lm_weights': [
   [ 1,  1, 0, 0],
   [ 1, 1, 0, 0],
],
'is_next_sentence': [0, 1]
}
```

## Reusable SavedModels, Serving, and Transfer learning
The new APIs proposed enable saving task and model-spepcific preprocessing pieces alongside the pre-trained model for use in serving and transfer learning.

For example, a pre-trained Transformer Encoder can be saved into a Reusable SavedModels with multiple entry points: one for the encoder itself, and one with all the end-to-end preprocessing functions and text entry point for a downstream task (e.g. single-segment classification, text-pair classification, etc).


## Class design

### Splitter

The current `Tokenizer` and `TokenizerWithOffsets` base classes from RFC 98 will be deprecated and existing subclasses will be migrated to inherit from Splitter and SplitterWithOffsets.

```
class Splitter(object):
  """An abstract class for splitting text."""

  @abc.abstractmethod
  def split(self, input): # pylint: disable=redefined-builtin
  """Splits `input` into sentences.

    Args:
       input: A string `Tensor` of shape [batch].

    Returns:
       A string `RaggedTensor` of shape [batch, (num_splits)].
  """

class SplitterWithOffsets(Splitter):
  """An abstract base class for splitters that support offsets."""

  @abc.abstractmethod
  def break_sentences_with_offsets(self, input):
    """Splits `input` into substrings and returns the starting & ending offsets.

    Args:
      input: A string `Tensor` of shape [batch].

    Returns:
      A tuple of (substrings, begin_offset, end_offset) where:

      sentences: A string `RaggedTensor` of shape [batch, (num_splits)] with
        each input broken up into its constituent splits.
      begin_offset: A int64 `RaggedTensor` of shape [batch, (num_splits)]
        where each entry is the inclusive beginning byte offset of a substring.
      end_offset: A int64 `RaggedTensor` of shape [batch, (num_splits)]
        where each entry is the exclusive ending byte offset of a substring.
    """
```

Splitter subclasses can implement different algorithms for segmenting strings and can even be a trained TF model. We also introduce two concrete implementations of Splitter: RegexSplitter and StateBasedSentenceBreaker). 


#### RegexSplitter

```
class RegexSplitter(SplitterWithOffsets):
  """A `Splitter` that splits sentences separated by a delimiter regex pattern.

  `RegexSplitter` splits text when a delimiter regex pattern is matched and returns
  the beginning and ending byte offsets of the splits as well.
  """
```

##### Example Usage

```
text_input=[
  b"Hi there.\nWhat time is it?\nIt is gametime.",
  b"Who let the dogs out?\nWho?\nWho?\nWho?",
]

sb = text.RegexSplitter(new_sentence_regex="\n")
sentences =  sb.break_sentences(text_input)

sentences = [
  [b"Hi there.", b"What time is it?", b"It is gametime."],
  [b"Who let the dogs out?", b"Who?", b"Who?", b"Who?"]
]
```

#### StateBasedSentenceBreaker

```
class StateBasedSentenceBreaker(SplitterWithOffsets):
  """A `Splitter` that splits sentences using a state machine to determine sentence breaks.

  `StateBasedSentenceBreaker` splits text into sentences by using a state machine to
  determine when a sequence of characters indicates a potential sentence break.

  The state machine consists of an "initial state", then transitions to a "collecting
  terminal punctuation state" once an acronym, an emoticon, or terminal punctuation
  (ellipsis, question mark, exclamation point, etc.), is encountered.

  It transitions to the "collecting close punctuation state" when a close punctuation
  (close bracket, end quote, etc.) is found.

  If non-punctuation is encountered in the collecting terminal punctuation or collecting
  close punctuation states, then we exit the state machine, returning false, indicating we have    
  moved past the end of a potential sentence fragment.
  """
```

##### Example Usage

```
text = [["Hello. Foo bar!"]]
sb = StateBasedSentenceBreaker()
split = sb.split(text)
split = [b"Hello.", b"Foo bar!"]

text = ["Hello (who are you...) foo bar"]
split = sb.split(text)
split = [b"Hello (who are you...)", b"foo bar"]
```

#### Trimmer
```
class Trimmer(object):
  """Trims a list of segments using a predetermined trimming strategy.
  Removes elements from tensors to ensure that they have a desired maximum size.

  When applied to a single tensor, this will mask values from the tensor to
  ensure that its size along a specified axis is bounded by a specified maximum
  length.  E.g.:

  >>> trimmer = WaterfallTrimmer(max_length=3, axis=1)
  >>> t1 = tf.ragged.constant([[10, 11, 12, 13, 14], [20, 21], [30, 31, 32, 33])
  >>> trimmer.trim(t1)
  <tf.RaggedTensor [[10, 11, 12], [20, 21], [30, 31, 32]]>
  >>> t1.row_lengths().numpy()  # All rows have length <= 3
  array([3, 1, 3])

  When applied to a list of tensors, this will mask values from those tensors to 
  ensure that their *total* length along the specified axis is bounded by a
  specified maximum length.  E.g.:

  >>> trimmer = WaterfallTrimmer(max_length=3, axis=1)
  >>> t1 = tf.ragged.constant([[10, 11, 12, 13, 14], [20, 21], [30, 31, 32, 33])
  >>> t2 = tf.ragged.constant([[100, 101], [200, 202, 203], [204, 205]])
  >>> trimmer.trim([t1, t2])
  [<tf.RaggedTensor [[10, 11, 12], [20, 21], [30, 31, 32]]>,
   <tf.RaggedTensor [[], [200], []]>]
  >>> (t1.row_lengths() + t2.row_lengths()).numpy()  # *total* row length <= 3
  array([3, 3, 3])

  The values that are removed from tensors are selected by the `generate_masks`
  method, which should be defined by concrete subclasses of `Trimmer`.  (Subclasses
  should not override the `trim` method, which simply applies a boolean_mask
  operation using the masks returned by `generate_masks`.)
  """

  def __init__(self, max_length, axis=1):
    """Constructs a new `Trimmer`.

    Args:
      max_length: The maximum total dimension length for the tensors along the 
        indicated axis. This may be a scalar or 1-D Tensor.
      axis: The axis whose size should be limited.
    """

  def trim(self, segments):
    """Trims elements from `segments` to have `max_length` total size.

    Args:
      segments: The `RaggedTensor` (or a list of `RaggedTensors`) to trim.
        If a list of tensors is used, then they must all have the same shape
        for the first `self.axis` dimensions.  (E.g., with the default value
        of `axis=1`), they must all have the same shape in the outermost dimensions.

    Returns:
      A copy of `segments` with elements removed to ensure that the total length
      along dimension `self.axis` is at most `self.max_length`.  Values are
      removed according to the mask returned by `self.generate_masks(segments)`.
    """

  @abc.abstractmethod
  def generate_masks(self, segments):
    """Generates a boolean mask specifying which values from `segments` to drop.

    Args:
      segments: The tensor (or list of tensors) to trim.  If a list of tensors is
        used, then they must all have the same shape for the first `self.axis`
        dimensions.  (E.g., with the default value of `axis=1`), they must all have
        the same shape in the outermost dimensions.

    Returns:
      A boolean mask (or list of masks) with the same shape(s) as `segments`,
      indicating which values to remove.  In particular, a `True` mask values 
      indicates that the corresponding value in `segments` should be kept; and
      a `False` mask value indicates that it should be dropped.  The total number
      of `True` values for all segments across axis `self.axis` must be less than
      or equal to `self.max_length`.
    """

  @property
  def max_length(self): return self._max_length

  @property axis(self): return self._axis
```

Trimmer implementations can choose to differ in how they select items for truncation. Concrete implementations are WaterfallTrimmer (allocates quota to each segment using a waterfall strategy) or a RoundRobinTrimmer (allocates quota using a round robin strategy).

#### WaterfallTrimmer

```
class WaterfallTrimmer(Trimmer):
  """Trims input by allocating quota using a `waterfall` strategy.

  `WaterfallTrimmer` calculates a drop mask given a budget of the
  max number of items for each or all batch row. The allocation of
  the budget is done using a 'waterfall' algorithm. This algorithm
  allocates quota in a left-to-right manner and fill up the buckets
  until we run out of budget.

  For example if the budget of [5] and we have segments of size
  [3, 4, 2], the truncate budget will be allocated as [3, 2, 0].

  The budget can be a scalar, in which case the same budget is broadcasted
  and applied to all batch rows. It can also be a 1D `Tensor` of size
  `batch_size`, in which each batch row i will have a budget corresponding to
  `max_length[i]`.
  """
```

##### Example Usage

```
segment_a =  [
  [b"hello", b"there"],
  [b"name", b"is"],
  [b"what", b"time", b"is", b"it", b"?"]
]

segment_b = [
  [b"whodis", b"?"],
  [b"bond", b",", b"james", b"bond"],
  [b"5:30", b"AM"]
]

trimmer = WaterfallTrimmer(max_length=[1, 3, 4])
trimmed_a, trimmed_b = trimmer.trim([segment_a, segment_b])

# first segment has shape [3, (1, 2, 4)]
trimmed_a = [
  [b"hello"],
  [b"name", b"is"],
  [b"what", b"time", b"is", b"it"]
]

# second segment has shape [3, (0, 1, 0)]
trimmed_b =   [
  [],
  [b"bond"],
  [],
]
```
Note that if trimmed_a and trimmed_b were concatenated along the last dimension, the last dimension would have a shape of [1, 3, 4] which is equal to max_length.

#### RoundRobinTrimmer

```
class RoundRobinTrimmer(Trimmer):
  """Trims input by allocating quota using a `round-robin` strategy.

  `RoundRobinTrimmer` calculates a drop mask given a budget of the
  max number of items for each or all batch row. The allocation of
  the budget is done using a round-robin algorithm. This algorithm
  allocates a single unit of quota to each segment in a left-to-right
  manner repeatedly until it runs out of budget.

  For example if the budget of [5] and we have segments of size
  [3, 4, 2], the truncate budget will be allocated as [2, 2, 1].

  The budget can be a scalar, in which case the same budget is broadcasted
  and applied to all batch rows. It can also be a 1D `Tensor` of size
  `batch_size`, in which each batch row i will have a budget corresponding to
  `max_length[i]`.
  """

trimmer = RoundRobinTrimmer(max_seq_length=[1, 3, 4])
trimmed_a, trimmed_b = trimmer.trim([segment_a, segment_b])

trimmed_a = [
  [b"hello"],
  [b"name", b"is"],
  [b"what", b"time"]
]

trimmed_b =   [
  [],
  [b"bond"],
  [b"5:30", b"AM"],
]
```

#### ItemSelector

```
class ItemSelector(object):
  """A class encapsulating the logic for selecting items.

  `ItemSelector` implementations contain algorithms for selecting items in a
  `RaggedTensor`. Users of `ItemSelector` implementations can call
  `get_selection_mask()` to retrieve a bool `RaggedTensor` mask indicating the
  items that have been selected. For example:

  ```
  inputs = tf.ragged.constant([
    [1, 2, 3, 4],
    [100, 200]
  ])

  selector = RandomItemSelector(...)

  selected = selector.get_selection_mask(inputs)

  #  selected = [
  #    [True, False, False, True],
  #    [True, True],
  #  ]
  ```

  For subclass writers that wish to implement their own custom, selection
  algorithm, please override `get_selection_mask()`.

  A helper function `get_selectable()` is provided to help subclass writers
  filter out undesirable items from selection. The default implementation will
  filter out items listed in `unselectable_ids`. Subclass writers may also
  override `get_selectable()` if they wish to customize the items to filter out
  from the selection algorithm.
  """
  def __init__(self, unselectable_ids=None):
    """Creates an instance of a `ItemSelector`.

    Args:
      unselectable_ids: a list, or `Tensor` of ids that are not selectable.
    """
  
  @property
  def unselectable_ids(self):
    return self._unselectable_ids

  def get_selectable(self, input_ids, axis):
    """Return a boolean mask of items that can be chosen for selection.

    Args:
      input_ids: a `RaggedTensor`.
      axis: axis to apply selection on.

    Returns:
      a `RaggedTensor` with dtype of bool and same shape as `input_ids` up to
      `axis` or `input_ids.shape[:axis]`. Its values are True if the
      corresponding item (or broadcasted subitems) should be considered for
      masking. In the default implementation, all `input_ids` items that are not
      listed in `unselectable_ids` (from the class arg) are considered
      selectable.
    """
  
  def get_selection_mask(self, input_ids, axis):
    """Returns a mask of items that have been selected.

    The default implementation returns all selectable items as selectable.

    Args:
      input_ids: A `RaggedTensor`.

    Returns:
      a `RaggedTensor` with the same shape as `input_ids` up to `axis` or
      `input_ids.shape[:axis]`. Its values are True if the corresponding item
      (or broadcasted subitems) should be considered for masking. The result
      contains bool values which describe if the corresponding value (or
      broadcasted subitem) is (True) or is not (False) selected.
    """
```

#### RandomItemSelector

```
class RandomItemSelector(ItemSelector):
  """An `ItemSelector` implementation that randomly selects items in a batch.

    `RandomItemSelector` randomly selects items in a batch subject to
    restrictions given (max_selections_per_batch, selection_rate and
    unselectable_ids).
  """

  def __init__(self,
               max_selections_per_batch,
               selection_rate,
               unselectable_ids=None,
               shuffle_fn=None):
    """Creates instance of `RandomItemSelector`.

    Args:
      max_selections_per_batch: An int of the max number of items to mask out.
      selection_rate: The rate at which items are randomly selected.
      unselectable_ids: (optional) A list of python ints or 1D `Tensor` of ints
        which are ids that will be not be masked.
      shuffle_fn: (optional) A function that shuffles a 1D `Tensor`. Default
        uses `tf.random.shuffle`.
    """
```

#### FirstNItemSelector

```
class FirstNItemSelector(ItemSelector):
  """An `ItemSelector` that selects the first `n` items in the batch."""

  def __init__(self, num_to_select, unselectable_ids=None):
    """Creates an instance of `FirstNItemSelector`.

    Args:
      num_to_select: An int which is the leading number of items to select.
      unselectable_ids: (optional) A list of int ids that cannot be selected.
        Default is empty list.
    """
```


#### MaskValuesChooser

```
class MaskValuesChooser(object):
  """Assigns values to the items chosen for masking.

  `MaskValuesChooser` encapsulates the logic for deciding the value to assign
  items that where chosen for masking. The following are the behavior in the
  default implementation:

  For `mask_token_rate` of the time, replace the item with the `[MASK]` token:

  ```
  my dog is hairy -> my dog is [MASK]
  ```

  For `random_token_rate` of the time, replace the item with a random word:

  ```
  my dog is hairy -> my dog is apple
  ```

  For `1 - mask_token_rate - random_token_rate` of the time, keep the item
  unchanged:

  ```
  my dog is hairy -> my dog is hairy.
  ```

  The default behavior is consistent with the methodology specified in
  `Masked LM and Masking Procedure` described in `BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding`
  (https://arxiv.org/pdf/1810.04805.pdf).

  Users may further customize this with behavior through subclassing and
  overriding `get_mask_values()`.
  """

  def __init__(self,
               vocab_size,
               mask_token,
               mask_token_rate=0.8,
               random_token_rate=0.1):
    """Creates an instance of `MaskValueChooser`.

    Args:
      vocab_size: size of vocabulary.
      mask_token: The id of the mask token.
      mask_token_rate: (optional) A float between 0 and 1 which indicates how
        often the `mask_token` is substituted for tokens selected for masking.
        Default is 0.8, NOTE: `mask_token_rate` + `random_token_rate` <= 1.
      random_token_rate: A float between 0 and 1 which indicates how often a
        random token is substituted for tokens selected for masking. Default is
        0.1. NOTE: `mask_token_rate` + `random_token_rate` <= 1.
    """
    if mask_token_rate is None:
      raise ValueError("`mask_token_rate` cannot be None")
    if random_token_rate is None:
      raise ValueError("`random_token_rate` cannot be None")
    self._mask_token_rate = mask_token_rate
    self._random_token_rate = random_token_rate
    self._mask_token = mask_token
    self._vocab_size = vocab_size

  @property
  def mask_token(self):
    return self._mask_token

  @property
  def random_token_rate(self):
    return self._random_token_rate

  @property
  def vocab_size(self):
    return self._vocab_size

  def get_mask_values(self, masked_lm_ids):
    """Get the values used for masking, random injection or no-op.

    Args:
      masked_lm_ids: a `RaggedTensor` of n dimensions and dtype int32 or int64
        whose values are the ids of items that have been selected for masking.
    Returns:
      a `RaggedTensor` of the same dtype and shape with `masked_lm_ids` whose
      values contain either the mask token, randomly injected token or original
      value.
    """
```

#### mask_language_model()

```
def mask_language_model(
    input_ids,
    item_selector,
    mask_values_chooser,
    axis=1):
):
"""Applies dynamic language model masking.

  `mask_language_model` implements the `Masked LM and Masking Procedure`
  described in `BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding`  (https://arxiv.org/pdf/1810.04805.pdf).
  `mask_language_model` uses an `ItemSelector` to select the items for masking,
  and a `MaskValuesChooser` to assign the values to the selected items.
  The purpose of this is to bias the representation towards the actual
  observed item.

  Masking is performed on items in an axis. A decision is taken independently at
  random to mask with [MASK], mask with random tokens from the full vocab, or
  not mask at all. Note that the masking decision is broadcasted to the
  sub-dimensions.

  For example, in a RaggedTensor of shape `[batch, (wordpieces)]` and if axis=1,
  each wordpiece independently gets masked (or not).

  Args:
    input_ids: A `RaggedTensor` of n dimensions (where n >= 2) on which
      masking will be applied to items up to dimension 1.
    axis: the axis where items will be treated atomically for masking.
    item_selector: An instance of `ItemSelector` that is used for selecting
      items to be masked.
    mask_values_chooser: An instance of `MaskValuesChooser` which determines the
      values assigned to the ids chosen for masking.
  Returns:
    A tuple of (masked_input_ids, masked_positions, masked_ids) where:

    masked_input_ids: A `RaggedTensor` in the same shape and dtype as
      `input_ids`, but with items in `masked_positions` possibly replaced
      with `mask_token`, random id, or no change.
    masked_positions: A `RaggedTensor` of ints with shape
      [batch, (num_masked)] containing the positions of items selected for
      masking.
    masked_ids: A `RaggedTensor` with shape [batch, (num_masked)] and same
      type as `input_ids` containing the original values before masking
      and thus used as labels for the task.
  """
```
  
#### Example Usage

`mask_language_model()` can mask or randomly insert items to the inputs at different scales, whether individual wordpieces, tokens or any arbitrary span. For example, with the following input:

```
ids = [[b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants" ],
       [b"Bar", b"##ack", b"Ob", b"##ama"],
       [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]]

masked_token_ids, masked_pos, masked_ids = mask_language_model(
  ids,
  item_selector=RandomItemSelector(max_selections_per_batch=2),
  MaskValuesChooser(...))
```
mask_language_model could end up masking individual wordpieces:

```
masked_token_ids = [[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
              [b"Bar", b"##ack", b"[MASK]", b"##ama"],
              [b"[MASK]", b"##vel", b"A", b"##ven", b"##gers"]]

masked_pos = [[0, 4],
              [2],
              [0]]

masked_ids = [["Sp", "##uare"],
              ["Ob"],
              [ "Mar"]]
```

..or with randomly insert wordpieces:

```
masked_token_ids = [[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
                    [b"Bar", b"##ack", b"Sq", b"##ama"],   # random token inserted for 'Ob'
                    [b"Bar", b"##vel", b"A", b"##ven", b"##gers"]]  # random token inserted for 'Mar'
```

mask_language_model() can mask any arbitrary spans that are constructed on the first dimension of a RaggedTensor. For example, if we have an RaggedTensor with shape `[batch, (token), (wordpieces)]`:

```
ids =  [[[b"Sp", "##onge"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
        [[b"Bar", "##ack"], [b"Ob", b"##ama"]],
        [[b"Mar", "##vel"], [b"A", b"##ven", b"##gers"]]]
```

`mask_language_model()` could mask whole spans:

```
masked_token_ids = [[[b"[MASK]", "[MASK]"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
              [[b"Bar", "##ack"], [b"[MASK]", b"[MASK]"]],
              [[b"[MASK]", "[MASK]"], [b"A", b"##ven", b"##gers"]]]
```

or insert randoms items in spans:

```
masked_token_ids = [[[b"Mar", "##ama"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
                    [[b"Bar", "##ack"], [b"##onge", b"##gers"]],
                    [[b"Ob", "Sp"], [b"A", b"##ven", b"##gers"]]]
```

#### combine_segments

```
def combine_segments(self, segments, start_of_sequence_id, end_of_segment_id):
  """Combines `segments`, adds special tokens, and generates segment ids.
 
 `combine_segments` combines the tokens of one or more input segments to a
  single sequence of token values and generates matching segment ids.
  `combine_segments` may be called after the invocation of a `Truncator`, if the
  user seeks to limit segment lengths, and and can be followed up by `pad_model_inputs`
  to pad the inputs for the model.

  See `Detailed Experimental Setup` in `BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding`
  (https://arxiv.org/pdf/1810.04805.pdf) for more examples of combined
  segments.

  `combine_segments` first flattens and combines a list of one or more
  segments (`RaggedTensor`s of n dimensions) together along the 1st axis, then packages
  any special tokens  into a final n dimensional `RaggedTensor`.

  And finally `combine_segments` generates another `RaggedTensor` (with the
  same rank as the final combined `RaggedTensor`) that contains a distinct int id
  for each segment.

  Args:
    segments: A list of `RaggedTensor`s with the tokens of the input segments.
      All elements must have the same dtype (int32 or int64), same rank, and
      same dimension 0 (namely batch size). Slice `segments[i][j, ...]`
      contains the tokens of the i-th input segment to the j-th example in the
      batch.
    start_of_sequence_id: a python int or scalar Tensor containing the id used
      to denote the start of a sequence (e.g. `[CLS]` token in BERT
      terminology).
    end_of_segment_id: a python int or scalar Tensor containing the id used to
      denote end of a segment (e.g. the `[SEP]` token in BERT terminology).

  Returns:
    a tuple of (combined_segments, segment_ids), where:

    combined_segments: A `RaggedTensor` with segments combined and special
      tokens inserted.
    segment_ids:  A `RaggedTensor` w/ the same shape as `combined_segments`
      and containing int ids for each item detailing the segment that they
      correspond to. Note that `start_of_sequence_id` will correspond to the 0th segment
      and `end_of_sequence_id` will correspond to the i-th segment.
  """
```

##### Example usage:

```
  segment_a = [[1, 2],
               [3, 4,],
               [5, 6, 7, 8, 9]]
    
  segment_b = [[10, 20,],
               [30, 40, 50, 60,],
               [70, 80]]
  expected_combined, expected_ids = combine_segments(
    [segment_a, segment_b], start_of_sequence_id=101, end_of_segment_id=102)

 # segment_a and segment_b have been combined w/ special tokens describing
 # the beginning of a sequence and end of a sequence inserted.
 expected_combined=[
   [101, 1, 2, 102, 10, 20, 102],
   [101, 3, 4, 102, 30, 40, 50, 60, 102],
   [101, 5, 6, 7, 8, 9, 102, 70, 80, 102],
 ]

 # ids describing which items belong to which segment.
 expected_ids=[
   [0, 0, 0, 0, 1, 1, 1],
   [0, 0, 0, 0, 1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
 ]
```

#### pad_model_inputs

```
def pad_model_inputs(input, max_seq_length, pad_value=0):
  """Pad model inputs and generate corresponding input masks.

  `pad_model_inputs` performs the final packaging of a model's inputs commonly
  found in text models. This includes padding out (or simply truncating) to a
  fixed-size, 2-dimensional `Tensor` and generating mask `Tensor`s (of the same
  2D shape) with values of 0 if the corresponding item is a pad value and 1 if
  it is part of the original input. Note that a simple truncation strategy
  (drop everything after max sequence length) is used to force the inputs
  to the specified shape. This may be incorrect and users should instead apply
  a `Trimmer` upstream to safely truncate large inputs.

  Args:
    input: A `RaggedTensor`.
    max_seq_length: An int, or scalar `Tensor`. The input `Tensor` will be
        flattened down to 2 dimensions and then have its 2nd dimension either
        padded out or truncated to this size.
    pad_value: An int or scalar `Tensor` specifying the value used for padding.

  Returns:
      A tuple of (padded_input, pad_mask) where:

      padded_input: A `Tensor` corresponding to `inputs` that has been 
        padded/truncated out to a fixed size and flattened to 2 dimensions.
      pad_mask: A `Tensor` corresponding to `padded_input` whose values are
        0 if the corresponding item is a pad value and 1 if it is not.
  """
```

##### Example Usage

```
inputs={
   "input_ids": [
     [101, 1, 2, 102, 10, 20, 102],
     [101, 3, 4, 102, 30, 40, 50, 60],
     [101, 5, 6, 7, 8, 9, 102, 70],
   ],
   "segment_ids": [
     [0, 0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0, 0, 1],
   ],
},

results = tf.nest.map_structure(
  pad_model_inputs, input, max_seq_length=10)
padded = {k: v[0] for k, v in results.items()}
expected_mask = results.values[0]

padded={
   "input_ids": [
     [101, 1, 2, 102, 10, 20, 102, 0, 0, 0],
     [101, 3, 4, 102, 30, 40, 50, 60, 0, 0],
     [101, 5, 6, 7, 8, 9, 102, 70, 0, 0]],
  "segment_ids": [
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  ],
}

expected_mask=[
  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
]
```