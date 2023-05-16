# DynamicEmbedding layer for Keras

Status        | Accepted
:------------ | :-----------------------------------------------------------
**RFC #**     | 446
**Author(s)** | Divyashree Sreepathihalli(divyasreepat@google.com)
**Sponsor**   | Rick Chao (rchao@google.com)
**Updated**   | 2023-05-16

## Objective
The objective of this proposal is to introduce the DynamicEmbedding layer to
the Keras ecosystem, providing a native solution for handling
colossal-scale problems in recommendation systems. The proposed solution 
facilitates automatic vocabulary building and updates, and dynamic embedding
updates corresponding to evolving input patterns and vocabulary changes.
### Goal
    * Works across accelerators (GPU / TPU)
    * Works with Parameter server strategy (asynchroous distributed training)
    * The solution requires minimum user code changes
    * Works with batched training and streamed training
    * Has performance parity with existing training jobs w/o dynamic embedding
### Extended goals
    * Works with synchronous distributed training

## Motivation
Recommendation systems and search ranking are crucial in powering the largest
revenue streams, such as PCTR/PCVR and video recommendation. However, as
recommendation models have become more complicated, there are three distinct
challenges that need to be addressed. These include 

the difficulty in separating popular and less-popular items or adapting to
the seasonal cycle of popularity, the lack of a cross-platform solution for
handling larger and larger embedding tables the dynamic nature of large
embedding tables due to modeling large unique id-based features and the
crossing features among them.

Currently, there are two ways to handle such limitations in TensorFlow: 
direct hashing without a vocabulary
a pre-computed fixed vocab with out-of-vocabulary hashing.
Neither approximation gives the user a fine grained control over
vocab-embedding mapping.  Hence, the proposal aims to provide a native
solution for handling these challenges by introducing the concept of
DynamicEmbedding.

### Why Keras?
We believe that internal and external users share many common pain points.
To support these features, external users today often need to rebuild an
entire suite of APIs, including optimizers, distributed training logic,
and customized TF kernels, to work around TensorFlow restrictions (that
variables are special-cased). As the middle layer of the TF tech stack,
we believe that we are in the best position to work with upstream 1P and
3P users, consolidate feedback, collaborate to drive a hardware-agnostic
solution.

## User Benefit
This initiative offers several benefits, including:
Providing a unified TensorFlow solution that allows for productive
exploration and potential large model quality gain across different use
cases.
Reducing computation cost and training latency by eliminating the need
for a pre-computed vocab.
Strengthening TensorFlow's advantage for third-party adoption.(Nvidia,
spotify, Tencent/Alibaba
(RFC: https://github.com/tensorflow/recommenders-addons/blob/master/rfcs/20200424-sparse-domain-isolation.md)
- vip.com case study
(https://drive.google.com/file/d/1UEWtixlA_zucLLkXlmgbF-4DAZHHKNmo/view?resourcekey=0-QXC4KOuQ6_RXuaYiyyRfYQ)

Additionally, many external users that rely on TensorFlow have already
adopted this idea, and open-source libraries have been pushing on this
front - TorchRec with a native embedding distribution and a dynamic
embedding solution & HugeCTR (Merlin) with a highly-performant
embedding caching strategy. This makes it essential for TensorFlow to
introduce a native solution to stay competitive in the market.

## Design Proposal
In this design approach, the DynamicEmbedding layer is composed of two
layers: the DynamicLookup layer and the Embedding layer. The
DynamicLookup layer is responsible for the following tasks:
    * Maintaining a vocabulary table using an eviction policy that is 
    updated based on input pattern.
    * Performing vocabulary lookup for the given input and returning
    integer indexes.
    * The index is then passed to the Embedding layer, which looks
    up the embedding vector. The Embedding layer is responsible for
    the following tasks:
        + Looking up the embedding vector for the given integer index.
        + Returning the embedding vector.
The embedding vector is then used by the subsequent layer in the
neural network. The Dynamic Embedding layer is used in conjunction
with UpdateEmbeddingCallback. The callback is triggered at a
predetermined time interval. It aggregates the Dynamic vocabulary
table across all workers and updates the vocabulary that is used
for input lookup across all workers. This ensures that the vocabulary
is always up-to-date and that all workers are using the same vocabulary. 


![DynamicEmbedding](/20230515-DynamicEmbedding/DynamicEmbedding.png)

Here is a deeper look at what is done in DynamicLookup layer and how the
UpdateEmbeddingCallback updates the embeddings and vocabulary
The DynamicEmbedding layer identifies and adds unique keys to the dynamic
vocabulary table for every input passed to it. This table is constantly
updated based on the eviction policy provided, such as TTL, LFU, or LRU.
The table is maintained on each worker when used with distributed
training, and the tables on different workers may be different.
The UpdateEmbeddingCallback is a timed callback that uses a timed
thread to create a callback event when the timer expires. The callback
aggregates the dynamic vocabulary table values across all workers in a
distributed training setup and updates the vocabulary on all workers. 
Update the vocab->index mapping(mutable hash table/ tf.Variable) on
all workers Update/remap the embedding matrix to reflect new
vocabulary-> index mapping
    * Old vocab keys will have the same embedding vector 
    * New vocab keys will have newly initialized embedding vector
This updated vocabulary is used for lookup in the DynamicLookup layer
until the callback event is triggered again after the time interval.

![DynamicLookup](0230515-DynamicEmbedding/DynamicLookup.png)

The image below illustrates the workflow when the parameter server
strategy is used. PSS supports asynchronous training. Each worker
will have a copy of the vocabulary, which will be consistent across
all the workers. Each worker learns the dynamic vocabulary table
independently. At regular intervals, in the update embedding callback,
the vocabulary table is aggregated from values across all the workers.
The top k vocabulary is extracted and the vocabulary lookup is updated
with these values.

![DynamicEmbedding asynchronous training](0230515-DynamicEmbedding/AsyncTraining.png)

## Performance implications
There are two options to have a mutable data structure to maintain the
dynamic vocabulary table:
    * Mutable hash tables
    * Variables with dynamic shapes
Here are some additional details about each option:
Mutable hash tables are a type of data structure that allows for quick
lookups of data. 
Variables with dynamic shapes are a type of data structure that allows
for variables to have different shapes at different times. This can be
useful for storing data that is constantly changing, such as the
vocabulary of a language. Right now, with parameter server strategy
variables cannot be placed on parameter servers. Mutable hash tables
are always placed on the chief, which could have performance
implications for lookups, inserts, and updates to the vocabulary.
However, if we can add support for the TensorFlow distribute side
to allow per-worker variable creation, this performance implication
can be overcome.

## Dependencies
The proposed feature does not introduce any new dependencies. It is
a stand-alone feature that can be used with any existing TensorFlow
workflow. There is no need to modify any existing code or workflows
to use this feature.

## Engineering Impact
This feature can add a small time overhead to update the dynamic
vocabulary table, but this comes with improved performance of models
and less user intervention to update vocabulary and restart training.
Training can be continuous and with real-time data, and the model
would continuously keep updating its vocabulary. This is beneficial
because it allows the model to learn new input patterns, which can
improve its accuracy and performance. Additionally, it reduces the
amount of time and effort required to maintain the model, as the
user does not need to manually update the vocabulary table or
restart training every time new data is available. These benefits
are particularly valuable in an online learning setting

## Platforms and Environments
    * GPU, TPU, CPU
    * Asynchronous distributed training
Synchronous distributed training

## Best Practices
The following are the best practices used so far:
    * The users need to stop training the model and update the
    vocabulary before restarting training.
    * The vocabulary that needs to be provided to the model needs
    to be generated by the user separately.
The DynamicEmbedding layer is a new layer that enables users to
train a model on a dataset with a dynamic vocabulary. This means
that the vocabulary can change over time without the user having
to stop training the model and update the vocabulary. The layer
is simply used as any other Keras layer. The initial vocabulary
can be provided or the layer will learn the whole vocabulary on
its own.

## Tutorials and Examples
```
from keras.layers import DynamicEmbedding  
train_data = np.array([
       ['a', 'j', 'c', 'd', 'e'],
       ['a', 'h', 'i', 'j', 'b'],
       ['i', 'h', 'c', 'j', 'e'],

   ])
   train_labels = np.array([0, 1, 2])
   vocab = tf.constant(['a', 'b', 'c', 'd', 'e'])
   eviction_policy = 'LFU'
   # Define the model
   model = keras.models.Sequential([
       DynamicEmbedding(
           input_dim=5,
           output_dim=2,
           input_length=5,
           eviction_policy=eviction_policy,
           initial_vocabulary=vocab,
       ),
       keras.layers.Flatten(),
       keras.layers.Dense(3, activation='softmax'),
   ])

   # Compile the model
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy'],
   )
   update_embedding_callback = dynamic_embedding.UpdateEmbeddingCallback(
       model.layers[0],
       interval=2,
   )
   with update_embedding_callback:
     result = model.fit(
         train_data,
         train_labels,
         epochs=100,
         batch_size=1,
         callbacks=[update_embedding_callback],
     )
```

## Compatibility
This design is forward and backward compatible. The layer should work with
both synchronous and asynchronous distribution strategies. The model with
DynamicEmbedding can be saved and loaded just like any other keras layer.
The vocabulary will be accessible to users to save and load as well.

## User Impact
Users will be able to access DynamicEmbedding as a new layer in Keras.
An illustration of how to use this layer is shown above.

## Acknowledgement
The [TensorFlow Recommenders Addon project](https://github.com/tensorflow/recommenders-addons/blob/master/docs/api_docs/tfra/dynamic_embedding.md)
maintained by TensorFlow SIG Recommenders is a community-led project that
aims to solve similar issues currently. This RFC is inspired by both
Google internal use cases as well as the TFRA project. We are thankful
for the contributions from TFRA maintainers (in particular, Haidong
Rong from Nvidia) and welcome future collaborations on this RFC.

## Questions and Discussion Topic
