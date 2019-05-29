# RFC: Tokenization API & Initial Implementations

Status        | Proposed
:------------ | :-----------------------------------
**Author(s)** | Robby Neale (Google)
**Sponsor**   | Mark Omernick (Google), Greg Billock (Google)
**Updated**   | 2019-05-29

## Objective {#objective}

Establish common interfaces for Tensorflow tokenizers, and introduce three
concrete op-level tokenizers.

## Motivation {#motivation}

There are a number of tokenization methods we wish to make available for
converting text runs into sequenced substrings processed by Tensorflow graphs.
In the past, these steps needed to be performed outside the graph in a data
preprocessing step, or through custom ops. The former had a chance of creating
skew if the preprocessing wasn't performed consistently, and the latter
fragmented Tensorflow NLP usage.

## User Benefit {#user-benefit}

To prevent further fragmentation, and to the benefit of all NLP modelers, we
wish to establish two tokenizer interfaces for new tokenizers to implement that
will make them easy to use, switch between, and compose. There is not one
best tokenizer for all use cases, and it is not a goal to establish a single
best tokenizer.

In addition to these tokenizer interfaces, we intend to discuss new concrete
subclasses - whitespace split, Unicode script split, and wordpiece.

## Design Proposal {#design-proposal}

We propose a base Tokenizer class that takes a Tensor or
[RaggedTensor](https://www.tensorflow.org/guide/ragged_tensors) of strings (or
optionally integer Unicode code points) as input, and outputs a RaggedTensor of
tokens. Tokens can be strings or integers (frequently as vocabulary
indices), and may differ from the originating text. By accepting strings, we
wish to make adoption and usage as easy as possible. This standardization on
both input and output formats, also allows for ease in composability between
tokenizers (see example of this in the custom_tokenizer example below). Plus,
the use of a base class allows for a single point of initialization in the
constructor, and not having to reinitialize when reusing the tokenizer.

```python
class Tokenizer(tf.Module):
  def tokenize(self, input):
    """
    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) Tensor or
        RaggedTensor.
    Returns:
      An N+1-dimensional UTF-8 string or integer Tensor or RaggedTensor.
    """
```

The number of tokens created from tokenizing a string is unknown. For this
reason, it is impossible to fully tokenize and output a normal tensor with a
uniform shape for a batch of varying strings. Thus, it is expected that each
output will be ragged (except in the vector, rank 1, case when the input is a
string scalar).

To allow the caller to know which groups of tokens belong to each string, the
innermost ragged dimension will be tokens for the originating string. This means
that the shape of the output will have an additional dimension when compared to
the input. Example:

```python
>>> tokenizer.tokenize(["This is great!", "Awesome!"])
[["This", "is", "great!"],
 ["Awesome!"]]
```

Model authors often want to know the alignment between the tokens and
the original string. For these instances, a separate class is available which
has a *tokenize_with_offsets* that returns a tuple containing the resulting
tokens plus a *best effort* of starting and ending offsets for each token into
the originating string. This is similar to the ops
`tf.strings.unicode_decode_with_offsets` and
`tf.strings.unicode_split_with_offsets`. We propose a new base class,
TokenizeWithOffsets, which extends Tokenizer and provides the added
functionality. This makes it clear whether or not the implementing Tokenizers
support the *_with_offsets* variant of tokenization.

```python
def TokenizerWithOffsets(Tokenizer):
  def tokenize_with_offsets(self, input):
    """
    Args:
      input: An N-dimensional UTF-8 string (or optionally integer) Tensor or
        RaggedTensor.
    Returns:
      A tuple (tokens, start_offsets, limit_offsets):
        * tokens is an N+1-dimensional UTF-8 string or integer Tensor or
            RaggedTensor.
        * start_offsets is an N+1-dimensional integer Tensor containing the
            starting indices of each token (byte indices for input strings).
        * limit_offsets is an N+1-dimensional integer Tensor containing the
            exclusive ending indices of each token (byte indices for input
            strings).
    """
```

Here is a basic example of using *tokenize_with_offsets*.

```python
>>> tokenizer.tokenize_with_offsets(["This is great!", "Awesome!"])
([["This", "is", "great!"], ["Awesome!"]],
 [[0, 5, 8], [0]],
 [[4, 7, 14], [8]])
```

Along with these base classes, there are three tokenizers we plan on
introducing - whitespace tokenizer, unicode script tokenizer, and a wordpiece
tokenizer.

### WhitespaceTokenizer {#whitespace_tokenize}

A basic tokenization method that splits on International Components for Unicode
(ICU) defined whitespace characters.

```python
class WhitespaceTokenizer(TokenizerWithOffsets):
  def tokenize(self, input):
    """
    Args:
      input: A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A RaggedTensor of tokenized text. The returned shape is the shape of the
        input tensor with an added ragged dimension for tokens of each string.
    """

  def tokenize_with_offsets(self, input):
    """
    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A tuple of `RaggedTensor`s `tokens`, `start_offsets`, and `limit_offsets`
      where:
        * `tokens`: A `RaggedTensor` of tokenized text.
        * `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
        * `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
    """
```

### UnicodeScriptTokenizer {#unicode_script_tokenize}

Splits strings based on the script codes of the Unicode code points. Script
codes correspond to ICU UScriptCode values. This means that text may often be
split by language as well as punctuation and whitespace. Similar to the
whitespace tokenizer, whitespace is removed.

```python
class UnicodeScriptTokenizer(TokenizerWithOffsets):
  def tokenize(self, input):
    """
    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A RaggedTensor of tokenized text. The returned shape is the shape of the
        input tensor with an added ragged dimension for tokens of each string.
    """

  def tokenize_with_offsets(self, input):
    """
    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A tuple of `RaggedTensor`s `tokens`, `start_offsets`, and `limit_offsets`
      where:
        * `tokens`: A `RaggedTensor` of tokenized text.
        * `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
        * `limit_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
    """
```

#### WordpieceTokenizer {#wordpiece_tokenize}

Wordpiece is an unsupervised text tokenizer which requires a predetermined
vocabulary for tokenization. It normally also requires a pretokenization step
that splits text into tokens, which wordpiece then splits further into
subwords (prefixes & suffixes).

[BERT](https://github.com/google-research/bert) currently uses Wordpiece.

```python
class WordpieceTokenizer(TokenizerWithOffsets):
  def __init__(self, vocab_lookup_table, suffix_indicator='##',
               max_bytes_per_word=100, token_out_type=tf.int64,
               unknown_token='[UNK]'):
    """
    Args:
      vocab_lookup_table: A lookup table implementing the LookupInterface
        containing the vocabulary of subwords.
      suffix_indicator: (optional) The characters prepended to a wordpiece to
        indicate that it is a suffix to another subword. Default is '##'.
      max_bytes_per_word: (optional) Max size of input token. Default is 100.
      token_out_type: (optional) The type of the token to return. This can be
        `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
      unknown_token: (optional) The value to use when an unknown token is found.
        Default is "[UNK]". If this is set to a string, and `token_out_type` is
        `tf.int64`, the `vocab_lookup_table` is used to convert the
        `unknown_token` to an integer. If this is set to `None`,
        out-of-vocabulary tokens are left as is.
    """

  def tokenize(self, input):
    """
    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A `RaggedTensor`s `tokens` where `tokens[i1...iN, j]` is the string
            contents, or ID in the vocab_lookup_table representing that string,
            of the `j`th token in `input[i1...iN]`
    """

  def tokenize_with_offsets(self, input):
    """
    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple of `RaggedTensor`s `tokens`, `start_offsets`, and `limit_offsets`
      where:
        * `tokens[i1...iN, j]` is the string contents, or ID in the
          vocab_lookup_table representing that string, of the `j`th token in
          `input[i1...iN]`
        * `start_offsets[i1...iN, j]` is the byte offset for the start of the
          `j`th token in `input[i1...iN]`
        * `limit_offsets[i1...iN, j]` is the byte offset for the end of the
          `j`th token in `input[i1...iN]`
    """
```

#### a CustomTokenizer example {#a-custom_tokenizer-example}

If all tokenizers follow the same principles, it allows for flexibility in
swapping out tokenization methods, can lend itself to composability, and will be
easy for anybody already familiar with standard tokenization APIs to use. Below
is a custom tokenizer example that extends the Tokenizer base class and thus not
providing a *tokenizer_with_offsets* method.

```python
class MyCustomTokenizer(Tokenizer):
  def tokenize(self, input):
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

    # put back into [batch, (num

update conventions doc -_tokens), (num_unicode_chars)]
    mix_tokenized = tf.RaggedTensor.from_row_lengths(
        values=unicode_split_tokens, row_lengths=script_tokenized.row_lengths())

    return mix_tokenized
```

## Appendix {#appendix}

### Appendix A - TF.Data example {#appendix-a}

A very common use case will be using Tokenizers in the [tf.data
API](https://www.tensorflow.org/guide/datasets). With the recent (in tf-nightly)
support for RaggedTensors in tf.data, this should be straight-forward for
anybody familiar with tf.data and pose no problems. A simple example is provided
below showing how this could look.

```python
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'],
                                           ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = tokenized_docs.make_one_shot_iterator()
tokenized_doc = iterator.get_next()
```

### Appendix B - Keras Preprocessing {#appendix-a}

Keras provides its own set of preprocessing layers, one which tokenizes,
normalizes, and vectorizes the text. An equivalent tokenizer (most likely the
WhitespaceTokenizer described above) will be provided for anybody wanting to
duplicate the tokenization functionality.

Because of the simplified nature of the Keras tokenization and that the
tokenizer API described above is to be included in a TensorFlow library outside
of core, these tokenizers will not be used from within the Keras preprocessing
layers to prevent the extra dependency from within Keras. However, more
full-featured Keras tokenization layers will be provided in the same library as
these tokenizers and use the API internally.

### Appendix C - Other tokenizers {#appendix-c}

Here we will briefly describe other tokenization methods that could extend the
same base classes despite not being Tensorflow ops.

#### Segmentation {#segmentation}

ML models trained to determine tokens within a given text are common solutions
for tokenizating CJKT languages, for example,
[SyntaxNet](https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html).
Since they use NN models internally for tokenizing, they do not package well as
ops, but instead could be built from TF ops and called through [Tensorflow
Serving](https://www.tensorflow.org/tfx/guide/serving)  or
[TF.Hub](https://www.tensorflow.org/hub).

#### SentencePiece {#sentencepiece}

[SentencePiece](https://github.com/google/sentencepiece) is an unsupervised text
tokenizer and detokenizer where the vocabulary size is predetermined prior to
the neural model training. SentencePiece implements subword units (e.g.
byte-pair-encoding (BPE)
[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)) and unigram
language model [Kudo.](https://arxiv.org/abs/1804.10959)) with the extension of
direct training from raw sentences.

