# RFC: Tokenization API Conventions & Initial Implementations

Status        | Proposed
:------------ | :-----------------------------------
**Author(s)** | Robby Neale (Google)
**Sponsor**   | Mark Omernick (Google), Greg Billock (Google)
**Updated**   | 2019-04-18

## Objective {#objective}

Establish a set of principles to follow that will unify Tensorflow tokenization
offerings, and apply these conventions to APIs for a set of op-level tokenizers.

## Motivation {#motivation}

There are a number of tokenization methods we wish to make available for
converting text runs into sequenced substrings processed by Tensorflow graphs.
In the past, these steps needed to be performed outside the graph in a data
preprocessing step, or through custom ops. The former had a chance of creating
skew if the preprocessing wasn't performed consistently, and the latter
fragmented Tensorflow NLP usage.

## User Benefit {#user-benefit}

To prevent further fragmentation, and to the benefit of all NLP modelers, we
wish to establish a set of API conventions for tokenization ops to follow that
will make them easy to use, switch between, and compose. There is not one
best tokenizer for all use cases, and it is not a goal to establish this best
tokenizer or a single API for all tokenizers to rigidly follow. [[See appendix A
for further details](#appendix-a-conventions-vs-oo)]

In addition to conventions to follow for tokenization ops, we intend to
discuss new APIs for whitespace split, Unicode script split, and wordpiece.

## Design Proposal {#design-proposal}

The base API for each tokenizer is that it takes a Tensor or
[RaggedTensor](https://www.tensorflow.org/guide/ragged_tensors) of strings (or
optionally integer Unicode code points) as input, and outputs a RaggedTensor of
tokens. Tokenizers can come in the form of ops, subgraphs, TF.Hub modules, or TF
Serving models. Tokens can be strings or integers (frequently as vocabulary
indices). By standardizing on inputs and output formats, we aim to allow for
ease in composability between tokenizers (see example of this in the
custom_tokenizer example below).

Below is a list of guidelines for the tokenization APIs, followed by
explanations regarding the reasoning behind each choice, and finally we list out
specific tokenization APIs that we wish to make available first.

### Conventions {#conventions}

1.  Tokenizers should be able to accept a Tensor or RaggedTensor of UTF-8
    encoded strings as input.
1.  Tokenizers should output a RaggedTensor of tokens that includes an added
    ragged dimension of tokens for each input string, unless the input is a
    scalar, in which case the output will be a vector Tensor.
1.  Tokenizers should also provide a `_with_offsets` variant that includes start
    and limit offsets with the output which will allow alignment of the tokens
    to the original byte offsets of the strings they are tokenized from.
1.  Tokenizers should use a lookup object containing a vocabulary for
    tokenization schemes requiring a vocabulary.

### Details {#details}

#### 1. String inputs - Speed vs Simplicity {#1-string-inputs-speed-vs-simplicity}

The input to each op is a Tensor (or RaggedTensor). For the common case, the
type of these tensors will be UTF-8 strings, but implementers may choose for
their tokenizers to also work on integers. By accepting strings, we wish to make
adoption and usage as easy as possible, even if comes at a slight performance
hit by working only with Unicode codepoints.

#### 2. Ragged outputs {#2-ragged-outputs}

The number of tokens created from tokenizing a string is unknown. For this
reason, it is impossible to fully tokenize and output a normal tensor with a
uniform shape for a batch of varying strings. Thus, it is expected that each
output will be ragged.

To allow the caller to know which groups of tokens belong to each string, the
innermost ragged dimension will be tokens for the originating string. This means
that the shape of the output will have an additional dimension when compared to
the input. Example:

```python
>>> whitespace_tokenize(["This is great!", "Awesome!"])
[["This", "is", "great!"],
 ["Awesome!"]]
```

#### 3. Offset variants {#3-offset-variants}

The tokenizers should include a companion op that returns byte offsets for each
token into the originating string. We use *_with_offsets* function variants (as
opposed to an additional *offsets=True* parameter) with the *start_offsets* &
*limit_offsets* naming to match the style employed by
`tf.strings.unicode_decode_with_offsets` and
`tf.strings.unicode_split_with_offsets`. This also has the benefit of separating
out the heavier alignment producing ops' return signatures in the API by name.

#### 4. Vocabulary Lookups - tf.Hub {#4-vocabulary-lookups-tf-hub}

For ops which require a vocabulary for looking up tokens, they should receive a
lookup object, as opposed to a flat file containing the vocabulary.
This can provide added functionality like out-of-vocab bucketing,
allowing for varying lookup implementations, and better resource handling that a
simple vocabulary file path cannot duplicate.

For these ops requiring external assets, if there is a desire to include a
standard version of the op along which includes the required asset files,
these should be published as a [TF.Hub](https://www.tensorflow.org/hub) module.
These modules can include popular versions of assets to be used for general use
cases. The published TF.Hub module should continue to follow the other
conventions outlined above.

### New Tokenize APIs {#new-tokenize-apis}

Following the above conventions, tokenization ops tend to follow a pattern.
This is intentional, as it gives familiarity across differing implementations,
and again allows for convenient exchanging of implementations.
This is not a base class, but a general pattern that is followed throughout
APIs of tokenization implementations following the above conventions.

```python
def xyz_tokenize(input, **kwargs):
  """
  Args:
    input: An N-dimensional string Tensor or RaggedTensor
    **kwargs: Additional tokenizer-specific options and data
      structures.  See specific implementations for details.
  Returns:
    An N+1-dimensional string or integer Tensor or RaggedTensor. (etc..)
  """
def xyz_tokenize_with_offsets(input, **kwargs):
  """
  Args:
    input: An N-dimensional string Tensor or RaggedTensor
    **kwargs: Additional tokenizer-specific options and data
      structures.  See specific implementations for details.
  Returns:
    A tuple (tokens, start_offsets, limit_offsets):
      * tokens is an N+1-dimensional string or integer Tensor or RaggedTensor. (etc..)
  """
```

#### whitespace_tokenize {#whitespace_tokenize}

A basic tokenization method that splits on International Components for Unicode
(ICU) defined whitespace characters.

```python
whitespace_tokenize
"""
Args:
  input: A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.

Returns:
  A RaggedTensor of tokenized text. The returned shape is the shape of the
    input tensor with an added ragged dimension for tokens of each string.
"""

whitespace_tokenize_with_offsets
"""
Args:
  input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

Returns:
  A tuple of `RaggedTensor`s `tokens`, `offset_starts`, `offset_limits` where:

  `tokens`: A `RaggedTensor` of tokenized text.
  `offset_starts`: A `RaggedTensor` of the tokens' starting byte offset.
  `offset_limits`: A `RaggedTensor` of the tokens' ending byte offset.
"""
```

#### unicode_script_tokenize {#unicode_script_tokenize}

Splits strings based on the script codes of the Unicode code points. Script
codes correspond to ICU UScriptCode values. This means that text may often be
split by language as well as punctuation and whitespace. Similar to the
whitespace tokenizer, whitespace is removed.

```python
unicode_script_tokenize
"""
Args:
  input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

Returns:
  A RaggedTensor of tokenized text. The returned shape is the shape of the
    input tensor with an added ragged dimension for tokens of each string.
"""

unicode_script_tokenize_with_offsets
"""
Args:
  input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

Returns:
  A tuple of `RaggedTensor`s `tokens`, `offset_starts`, `offset_limits` where:

  `tokens`: A `RaggedTensor` of tokenized text.
  `offset_starts`: A `RaggedTensor` of the tokens' starting byte offset.
  `offset_limits`: A `RaggedTensor` of the tokens' ending byte offset.
"""
```

#### wordpiece_tokenize {#wordpiece_tokenize}

Wordpiece is an unsupervised text tokenizer which requires a predetermined
vocabulary for tokenization. It normally also requires a pretokenization step
that splits text into tokens, which wordpiece then splits further into
subwords (prefixes & suffixes).

[BERT](https://github.com/google-research/bert) currently uses Wordpiece.

```python
wordpiece_tokenize
"""
Args:
  input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
  vocab_lookup_table: A lookup table implementing the LookupInterface
    containing the vocabulary of subwords.
  suffix_indicator: (optional) The characters prepended to a wordpiece to
    indicate that it is a suffix to another subword. Default is '##'.
  max_bytes_per_word: (optional) Max size of input token. Default is 100.
  token_out_type: (optional) The type of the token to return. This can be
    `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
  unknown_token: (optional) The value to use when an unknown token is found.
    Default is "[UNK]". If this is set to a string, and `token_out_type` is
    `tf.int64`, the `vocab_lookup_table` is used to convert the `unknown_token`
    to an integer. If this is set to `None`, out-of-vocabulary tokens are left
    as is.

Returns:
  A `RaggedTensor`s `tokens` where `tokens[i1...iN, j]` is the string
        contents, or ID in the vocab_lookup_table representing that string,
        of the `j`th token in `input[i1...iN]`
"""

wordpiece_tokenize_with_offsets
"""
Args:
  input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
  vocab_lookup_table: A lookup table implementing the LookupInterface
    containing the vocabulary of subwords.
  suffix_indicator: (optional) The characters prepended to a wordpiece to
    indicate that it is a suffix to another subword. Default is '##'.
  max_bytes_per_word: (optional) Max size of input token. Default is 100.
  token_out_type: (optional) The type of the token to return. This can be
    `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
  unknown_token: (optional) The value to use when an unknown token is found.
    Default is "[UNK]". If this is set to a string, and `token_out_type` is
    `tf.int64`, the `vocab_lookup_table` is used to convert the `unknown_token`
    to an integer. If this is set to `None`, out-of-vocabulary tokens are left
    as is.

Returns:
  A tuple of `RaggedTensor`s `tokens`, `start_offsets`, `limit_offsets` where:

    `tokens[i1...iN, j]` is the string contents, or ID in the vocab_lookup_table
      representing that string, of the `j`th token in `input[i1...iN]`
    `start_offsets[i1...iN, j]` is the byte offset for the start of the `j`th
      token in `input[i1...iN]`
    `limit_offsets[i1...iN, j]` is the byte offset for the end of the `j`th
      token in `input[i1...iN]`
"""
```

#### a custom_tokenizer example {#a-custom_tokenizer-example}

If all tokenizers follow the same principles, it allows for flexibility in
swapping out tokenization methods, can lend itself to composability, and will be
easy for anybody already familiar with standard tokenization APIs to use. A
non-offsets version of a custom tokenize function is provided below that one
might implement following the same set of guidelines.

```python
def my_tokenize(input):
 """
  A custom tokenizer for string tensors.

  Args:
    input: An N-dimensional string Tensor or RaggedTensor

  Returns:
    An N+1-dimensional string or integer Tensor or RaggedTensor.
  """
  # normalize & strip control characters
  input = tf_text.case_fold_utf8(input)
  input = tf.strings.regex_replace(input, r"\p{Cc}|\p{Cf}", "")

  # tokenize based on unicode_script
  script_tokenized = tf_text.unicode_script_tokenize(input)
  token_codepoints =  tf.strings.unicode_script(
      tf.strings.unicode_decode(script_tokenized.flat_values, "UTF-8"))

  HAN_SCRIPT_ID = 17
  is_han_script = tf.equal(token_codepoints, HAN_SCRIPT_ID)[:, :1].values
  is_emoji = tf_text.wordshape(
      script_tokenized.flat_values, text.WordShape.HAS_EMOJI)

  # Further splitting
  split_cond = is_han_script | is_emoji
  unicode_char_split = tf.strings.unicode_split(script_tokenized, "UTF-8")
  unicode_split_tokens = tf.where(
      split_cond,
      y=tf.expand_dims(script_tokenized.flat_values, 1),
      x=unicode_char_split.values)

  # put back into [batch, (num_tokens), (num_unicode_chars)]
  mix_tokenized = tf.RaggedTensor.from_row_lengths(
      values=unicode_split_tokens, row_lengths=script_tokenized.row_lengths())

  return mix_tokenized
```

## Appendix {#appendix}

### Appendix A - Conventions vs OO {#appendix-a-conventions-vs-oo}

This RFC lays out a series of conventions. This document consciously does not
try to establish a base class or interface that all tokenizers extend or
implement. A base class would overly restrict access to tokenization functions.

### Appendix B - Other tokenizers {#appendix-b-other-tokenizers}

Here we will briefly describe other tokenization methods that could follow the
same set of conventions despite not being Tensorflow ops.

#### Segmentation {#segmentation}

ML models trained to determine tokens within a given text are common solutions
for tokenizating CJKT languages, for example,
[SyntaxNet](https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html).
Since they use NN models internally for tokenizing, they do not package well as
ops, but instead could be built from TF ops and called through [Tensorflow
Serving](https://www.tensorflow.org/tfx/guide/serving)  or
[TF.Hub](https://www.tensorflow.org/hub). The implementation differences should
not prevent segmentation modules from following the same set of API conventions
described in this RFC.

#### SentencePiece {#sentencepiece}

[SentencePiece](https://github.com/google/sentencepiece) is an unsupervised text
tokenizer and detokenizer where the vocabulary size is predetermined prior to
the neural model training. SentencePiece implements subword units (e.g.
byte-pair-encoding (BPE)
[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)) and unigram
language model [Kudo.](https://arxiv.org/abs/1804.10959)) with the extension of
direct training from raw sentences.

