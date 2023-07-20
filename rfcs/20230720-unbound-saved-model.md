# Unbound SavedModel

| Status        | Implemented                                          |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Kathy Wu (<kathywu@google.com>)                        |
| **Updated**   | 2023-05-22                                           |

## Objective

We are proposing a new format for SavedModel and a generic proto-splitting library that resolves the 2GB proto issue, specifically for Adbrain TF2 model graph generation.

## Motivation

The 2GB proto serialization limitation is an ongoing problem that has plagued many projects. We have evaluated many solutions and have decided on splitting the proto as the short-term solution, with a longer term plan to migrate to a common way to represent using MLIR. This doc focuses on splitting the proto message, but see the appendix for all alternatives considered and a summary of ongoing issues involving scalability of the SavedModel format.

- (Appendix) [Alternate Solutions](#TODO)
- (Appendix) [SavedModel / Proto Scaling Issues](#TODO)

### The Adbrain 2GB problem

Adbrain models are generated from Python, then exported to SavedModel (TF2) or GraphDef (TF1). These models are then passed to the TF2 Loading C++ API or the custom training framework for TF1 GraphDefs.

The number of nodes and functions in the GraphDef directly corresponds to the number of slices (devices) used. When there are too many devices, the GraphDef/SavedModel cannot be exported. To work around this, the number of devices is reduced. This workaround, however, significantly slows down step times.

Adbrain would like a way to serialize and deserialize models that exceed this limit. There are existing models currently blocked in the TF2 pilot on the SavedModel size.

Since proto cannot natively handle sizes > 2GB, we must develop a new format for serializing the SavedModel proto.

## User Benefit

As models get larger, the likelihood of encountering the 2GB limit increases. Many users inside and outside Google have reported this issue, and were met with no solution, resulting in hacks and workarounds. With this change, current and future model builders will be unrestricted by the size of the SavedModel proto.

## Design Proposal

### SavedModel Changes

A new option will be introduced to `tf.saved_model.save`, which is capable of saving SavedModels that exceed the regular GraphDef. SavedModels produced with this functionality enabled can only be consistently read from the TF2 C++ / Python APIs. If the SavedModel proto does not exceed 2GB, then the output format will still be a single binary proto profile and remains compatible with TF1 APIs (although this could change in the future).

There are many customers who could benefit from this option, but downstream libraries may require updates to their code.
API Change:

```python
tf.saved_model.SaveOption(..., experimental_image_format=False)
"""
  experimental_image_format: When True, enables exporting SavedModels that 
    would normally error due to exceeding the protobuf 2GB serialization 
    limit. Enabling this option may break compatibility with other 
    SavedModel consumers (e.g. TensorFlow Serving, TPU Inference Converter,
    etc.). SavedModels saved in the image format can be read by the TF2 
    C++ and Python loading APIs. Defaults to False.
""" 
```

File Change:

| Old [SavedModel format]   | New [Image format]                               |
| ------------------------- | ------------------------------------------------ |
| saved_model_dir/<br>├ variables/<br>├ assets/<br>├ fingerprint.pb<br>├ saved_model.pb | saved_model_dir/<br>├ variables/<br>├ assets/<br>├ fingerprint.pb<br>├ saved_model.cpb

### Proto-splitting library

We will add a generic proto-splitting library that can handle splitting, writing, reading, and merging for both Python and C++.

A generic protobuf splitter could handle more situations than a more specialized algorithm but it comes at the cost of being less efficient. Knowing the structure of the message beforehand allows the algorithm to skip several steps of the proto message tree traversal to find which fields to split. For example, in the SavedModel case, the most common causes of large protos are: (1) large embedding constants, and (2) a large number of the repeated NodeDef/Function field. An algorithm optimized for splitting GraphDef would only look at "const"-type nodes, and the repeated NodeDef/Function fields. A generic algorithm would be looking at each field in GraphDef, most of which do not have issues with size.

This library implements generic `Splitter` and `Merger` classes that can handle all message types, and users may register specialized splitters for specific message types.

#### Public C++ API

```c++
namespace proto_splitter {

// Abstract class that defines how to split a proto message.
// Can be specialized for specific message types.
class Splitter {
 public:
  // Splits the proto into smaller chunks.
  virtual StatusOr<pair<vector<MessageBytes>*, ChunkedMessage*>> Split() = 0;

  // Write message to disk.
  virtual Status Write(string file_prefix) = 0;
}

// A Splitter that can be composed with other splitters.
class ComposableSplitter : public Splitter {
 public:
  explicit ComposableSplitter(Message* message);

  explicit ComposableSplitter(Message* message,
                              ComposableSplitter* parent_splitter,
                              vector<FieldType>* fields_in_parent);

  // Splits a proto message.
  // Returns a pair of (
  //   chunks: List of messages/bytes,
  //   ChunkedMessage: Metadata about the chunked fields.)
  // If the message is not split, `chunks` should only contain the original
  // message.
  StatusOr<pair<vector<MessageBytes>*, ChunkedMessage*>>
  Split() override;

  // Serializes a proto to disk.
  // The writer writes all chunks into a Riegeli file. The chunk metadata
  // (ChunkMetadata) is written at the very end.
  //   file_prefix: string prefix of the filepath. The writer will automatically
  //     attach a `.pb` or `.cpb` (chunked pb) suffix depending on whether the
  //     proto is split.
  Status Write(string file_prefix) override;

  // Builds the Splitter object by generating chunks from the proto.
  // Subclasses of `ComposableChunks` should only need to override this method.
  // This method should be called once per Splitter to create the chunks.
  // Users should call the methods `Split` or `Write` instead.
  virtual Status BuildChunks() = 0;

 protected:
  // Adds a new chunk and updates the ChunkedMessage proto. If set, the index
  // indicates where to insert the chunk.
  Status AddChunk(unique_ptr<MessageBytes> chunk,
                  vector<FieldType>* fields, int* index = nullptr);
}


// A static class for merging an on-disk or in-memory chunked proto.
class Merger {
 public:
  // Merges the provided `chunks` into `merged_message` using `chunked_message`.
  static Status Merge(const vector<std::unique_ptr<Message>>& chunks,
                      const ChunkedMessage& chunked_message,
                      Message* merged_message);

  // Reads a proto from `prefix` (must be .pb or .cpb) into `merged_message`.
  // The proto format of `merged_message` must match the format of the proto
  // written to `prefix`.
  static Status Read(string prefix, Message* merged_message);
}

} // namespace proto_splitter
```

#### Public Python API

```python
class Splitter(abc.ABC):
  """An abstract class for splitting and writing protos that are > 2GB."""

  def split(
    self
  ) -> tuple[Sequence[Union[message.Message, bytes]], chunk_pb2.ChunkedMessage]:
    """Splits proto message into a Sequence of protos/bytes."""

  def write(self, file_prefix: str) -> None:
    """Serializes proto to disk.

    Args:
      file_prefix: string prefix of the filepath.
    """

class ComposableSplitter(Splitter):
  """A Splitter that can be composed with other splitters.

  This Splitter writes to the riegeli file format.
  """

  def __init__(
      self,
      proto,
      *,
      proto_as_initial_chunk: bool = True,
      parent_splitter: Optional["ComposableSplitter"] = None,
      fields_in_parent: Optional[util.FieldTypes] = None,
  ):
    """Initializes ComposableSplitter.

    Args:
      proto: Proto message to split.
      proto_as_initial_chunk: Whether to initialize chunks with the
        user-provided proto as the initial chunk.
      parent_splitter: The parent `ComposableSplitter` object.
      fields_in_parent: Fields to access `proto` from the parent splitter's
        proto.
    """

  def build_chunks(self) -> None:
    """Builds the Splitter object by generating chunks from the proto.

    Subclasses of `ComposableChunks` should only need to override this method.

    This method should be called once per Splitter to create the chunks.
    Users should call the methods `split` or `write` instead.
    """

  def split(
      self,
  ) -> tuple[Sequence[Union[message.Message, bytes]], chunk_pb2.ChunkedMessage]:
    """Splits a proto message into a Sequence of protos/bytes."""

  def write(self, file_prefix: str) -> None:
    """Serializes a proto to disk.

    The writer writes all chunks into a riegeli file. The chunk metadata
    (ChunkMetadata) is written at the very end.

    Args:
      file_prefix: string prefix of the filepath. The writer will automatically
        attach a `.pb` or `.cpb` (chunked pb) suffix depending on whether the
        proto is split.
    """

  def add_chunk(
      self,
      chunk: Union[message.Message, bytes],
      field_tags: util.FieldTypes,
      index=None,
  ) -> None:
    """Adds a new chunk and updates the ChunkedMessage proto.

    Args:
      chunk: Proto message or bytes.
      field_tags: Field information about the placement of the chunked data
        within self._proto.
      index: Optional index at which to insert the chunk. The chunk ordering is
        important for merging.
    """
```

The `split.py` and `composable_splitter.h` files provide `ComposableSplitter` classes that are implemented to write to the Riegeli format, and allow combinable implementations of different message splitters.

Recommended steps to subclass `ComposableSplitter`:

1. (required) Override `build_chunks()`. This method sets the values of `self._chunks` and `self._chunked_message` based on the user-passed proto.
2. Update `version_def`. This is important to ensure that users are able to apply the Merger to the chunked proto, or get understandable version errors.
3. If `__init__` is overridden: call `super().__init__(proto, **kwargs)`. This is optional but highly recommended since it sets up basic attributes that may be needed by other splitters.

Users can apply the python `Splitter` implementations by calling:

```python
splitter = MySplitterClass(proto)

# Export the chunks to a file.
splitter.write(file_prefix)

# Access the chunks created in the splitter.
chunks, chunked_message = splitter.split()
```

Once the proto has been split (and possibly written to disk), the `Merger` class can be used to merge it back into its original form. `Merger::Merge` requires the user to already have a collection of chunks in memory, while `Merger::Read` merges a chunked proto directly from disk. The methods can be used like so:

```c++
// Merge from memory
vector<unique_ptr<Message>> my_chunks = GetMyChunks();
ChunkedMessage chunked_message = GetMyChunkedMessage();
my_project::MyProto my_proto;
Merger::Merge(my_chunks, chunked_message, &my_proto);

// Read from disk
my_project::MyOtherProto my_other_proto;
Merger::Read("path/to/my_proto_file", &my_other_proto);
```

#### Chunked File Specification

The chunked proto is split into a sequence of smaller messages or scalars, all serialized into bytes in a Riegeli file (Riegeli also uses the term "chunk" for each entry in the format). We use the [Riegeli](https://github.com/google/riegeli) format for the following reasons:

- Does not silently skip corrupted records (RecordIO does this)
- Handles data packing and compression
- Limits is 2^64 bytes per chunk
- Fast chunk seeking
- Fast writing (parallel encoding)

The last chunk in the `{PREFIX}.cpb` file is always a set of serialized chunks followed by a `ChunkedMetadata` message. In the initial version of the chunker, all chunks will be stored in a single file, but in the future there may be other outputted files that follow the pattern `{PREFIX}.*.cpb`. The main file will always be `{PREFIX}.cpb`.

> NOTE: We do not use the .riegeli suffix to somewhat hide the implementation details about the format. The user should go through official APIs to read and write `SavedModel`. Many dependencies on `SavedModel` are on the `saved_model.pb` file itself. Converters and libraries read/write directly from this file, which makes it difficult to update the format without breaking these use cases.

#### Chunked Proto Specification

```protobuf
// The ChunkMetadata is generated and saved as the last chunk when exported with
// Splitter.write(prefix). This metadata is always located in the {prefix}.cpb
// file.
message ChunkMetadata {
  // The version of the Splitter was used (for information purposes only), and
  // what versions of the Merger can be used to read the chunked proto.
  VersionDef version = 1;
  // Metadata about how/where to read the chunk data.
  repeated ChunkInfo chunks = 2;
  // Recursive structural information of the proto that was passed to the
  // Splitter. Used to merge the chunked data into a single proto.
  ChunkedMessage message = 3;
}

message ChunkInfo {
  // List of types supported by the Merger.
  enum Type {
    UNSET = 0;
    MESSAGE = 1;
    BYTES = 2;
  }
  Type type = 1;
  uint64 size = 2;    // Size of this chunk in bytes.
  uint64 offset = 3;  // Byte offset of this chunk.
}

message ChunkedMessage {
  // Index of the ChunkInfo in ChunkedProto.chunks.
  optional uint64 chunk_index = 1;

  // The same field can appear multiple times in this list. When this
  // happens, the parsed blob data is joined in the order that they appear.
  // The way the field is joined depends on the type of field. e.g. Repeated
  // fields are joined by appending them, bytes are joined concatenated.
  repeated ChunkedField chunked_fields = 2;
}

message ChunkedField {
  // Stores the nested field numbers, which tell us where and how to rejoin
  // this chunk.
  // E.g. If the parent ChunkedMessage contains a GraphDef, this field value
  // could point to: `GraphDef.node[15].attr["value"].tensor.float_value`
  // Field tag = [] (empty list) indicates that the chunked message is the same
  // type as the parent. This is useful for slicing repeated fields.
  repeated FieldIndex field_tag = 1;
  ChunkedMessage message = 3;  // This allows us to build a tree of chunked
                               // messages.
}

message FieldIndex {
  message MapKey {
    oneof type {
      string s = 1;
      bool boolean = 2;
      uint32 ui32 = 3;
      uint64 ui64 = 4;
      int32 i32 = 5;
      int64 i64 = 6;
    }
  }
  oneof kind {
    uint32 field = 1;  // Field number
    MapKey map_key = 2;
    uint64 index = 3;
  }
}
```

We choose to use the proto format here for familiarity. In the event that even `ChunkedMessage` exceeds 2GB, we can (in the future) add recursion to `Splitter.Split()` to chunk the output proto until it is less than 2GB. The version field is there to handle this future feature.

##### Versioning

The outputs of `Splitter.Split()` (`ChunkedProto` and `ChunkedMessage`) have a version field to handle future updates.

###### Hypothetical version updates

- The initial version of `Splitter` writes a single riegeli file, but in a new version it outputs several files.
- The `ChunkedMessage` itself could exceed 2GB. In a new version of `Splitter`, the `ChunkedMessage` can be recursively chunked and recursively joined when read.
- Riegeli is no longer used. `*.chunk.riegli` is kept so that previous versions of the code can still read it. Old code will read the `VersionDef` and see that it can no longer process the data, and report an understandable error.
- The initial version of `Splitter` returns serialized proto chunks, but in a new version, returns a chunk reference that is actually a URL to an external file.

### Alternatives Considered

We decided to explore different serialization formats, even outside of proto. There hasn't been any major changes to `SavedModel` in ages - lots of libraries are built on top of the `saved_model.pb` or a `GraphDef` file, and are parsing that proto directly. Any change we make to the format now is almost equally painful, and since we've decided to do this, we're exploring all options out there.

The conclusion can likely be that re-using proto is the best, as it gives us an easy solution to compatibility and we can focus on other areas (e.g. MLA integration to interface with other frameworks). However, even splitting the proto comes with compatibility problems: What happens when we need to split the proto again for constants? This breaks compatibility -again-. Also in the future, are we anticipating more cross-framework models (JAX2TF is using a workaround by storing StableHLO as an attribute in a custom op)? Why not invest this time in MLIR, which supports multiple dialects and will never hit a file size limit?

#### Proto Splitting: Introduce REF fields to the current schema

An alternative to the `Splitter` described above is introducing a new `Ref` message type that references external data. When the proto must be split, we serialize the split data into a separate file, and reference the file in a new field.

For example, splitting `GraphDef` nodes and constant value:

<table>
<tr>
<td>

```protobuf
message GraphDef {
  repeated NodeDef node = 1;
  FunctionDefLibrary library = 2;
  VersionDef versions = 4;
  Ref node_ref = 5;
}

message Ref {
  string filename = 1;
}
```

</td>
<td>

```protobuf
message AttrValue {
  oneof value {
    bytes s = 2;                 // "string"
    int64 i = 3;                 // "int"
    float f = 4;                 // "float"
    bool b = 5;                  // "bool"
    DataType type = 6;           // "type"
    TensorShapeProto shape = 7;  // "shape"
    TensorProto tensor = 8;      // "tensor"
    ListValue list = 1;          // any "list(...)"
    NameAttrList func = 10;
    string placeholder = 9;
    Ref ref_value = 11;
  }
}
```

</td>
</tr>
</table>

This approach allows existing code to continue parsing the `GraphDef`/`SavedModel` proto with minimal changes, and allows us to continue using the proto library from the box.

##### The problem

Many existing converters and libraries parse the proto directly from file, or utilize the proto as some in-memory representation. When a new ref field is added, the existing code will ignore this field unless they are updated. They will see an incomplete Graph, or a constant node with no specified value. These differences in the representation will bubble up as errors later down the line (for example, when tracing the graph to find connected ops, or trying to create the constant tensor).

#### Alternate Formats

We are exploring written formats for both the image dump and function graphs (which are currently represented as Proto data structures in-memory). The format for the image does not necessarily need to be Proto-compatible, but the serialization of the function graph should be.

In the future, we should consider using other HLO or using native data structures to represent functions and ops in-memory, rather than a proto-backed data structure. Proto recommends not using protobuf when it is not being serialized.

Specific features we are looking for:

- No 2GB limit
- Easy migration for current proto schemas: 
- Random access / Fast parsing
- Compatibility support
- Cross-language support
- Flexible API but somewhat obscure written format

##### *Easy migration for current proto schemas*

We can assume that `GraphDef` is constantly changing, e.g. with the introduction of new types and fields. Even with ~1 major change a year, it would be problematic if the CL author had to jump through hoops to make it serializable with the Image format.

##### *Flexible API but obscure written format*

Many projects write directly to/from the `SavedModel` proto instead of using the intended APIs. This leads to many unintended dependencies on the format that make it difficult for us to make the needed changes.

The format should come with APIs that are flexible enough to handle existing use cases (e.g. inspecting the image, transforming a function from one graph to another, inserting new symbols), but not be a free-for-all transformation that we have no control over.

##### The options

###### Proto

- Has 2GB limitation

###### Convert function Graph to MLIR

- MLIR can be used for interop with JAX

- Serialization format isn't mature - doesn't have strong compatibility guarantees that we support for SavedModels. A good long term solution as long as this issue is ironed out.

###### MessagePack

- Used by PAX/ORBAX checkpoints to serialize checkpoint values and PyTree.
- Good for serializing data in a small and fast format (Like JSON but fast)
- File format itself has no 2GB limitation, and the limits for individual fields is the same as proto. 
- Not great for serializing the TF Graph IR. When starting from a `GraphDef`/`FunctionDef` proto, it takes a lot of time to convert the proto into a dict (~1 min for a 300MB proto). Excluding this conversion time, packing the message into a binary takes 30% more time than proto.
  - Note: We could explore directly serializing proto to the msgpack file format. Some moma-fu shows that there are Go teams that have taken this approach. However, I don't think this can be accomplished in the allotted time frame (Q2 2023) without help.

###### Flatbuffer

- Extra work in translating schema and ensuring that they are in sync.
- Also has a 2GB limitation as it is also based on int32.

###### RecordIO

- Fault-tolerant to a fault (will silently skip records with errors). This is not permissible in model graphs.
- No 2GB limit

###### CapnProto

- Extra work in translating schema and ensuring they are in sync
- Has a data blob limit of 512 MB
- No 2GB limit
- Very fast

###### FRODO (fast read-only data objects) go/frodo/overview

- still has 2GB limit
- More cache efficient
- Compatible with .proto

###### Table-Driven Protobuf Parsing

- Saves a bit of file size but looks like it still has 2GB limit
- Much faster parsing
- Compatible with .proto

###### JSON, XML

- Human-readable
- Slow

###### HDF5

- Not supported on all file systems (CNS)

###### Database or Query-compatible format (SQL, SSTable)

- Could be used for storing `GraphDef`, but not sure if this path is worth exploring

###### Home-brewed format [.tf]

- We have to write our own encoder and decoder
- Efficiency, performance, compatibility are on us
- Homemade format can target versioning. Proto doesn't have versions. Differences in schema are handled automatically by proto encoding/decoding logic, it's easy to edit schema without bumping up any versions [`GraphDef` mitigates this by using a bot that updates the version every day]. Differences in the representation are handled programmatically, leading to legacy code floating around to handle older serialized models.
- Can exceed 2GB as needed

#### Related Discussions and Solutions

##### Official Documentation

From the Protobuf docs:

> Protocol Buffers do not include any built-in support for large data sets because different situations call for different solutions. Sometimes a simple list of records will do while other times you want something more like a database. Each solution should be developed as a separate library, so that only those who need it need pay the costs.

##### MLIR: TFG Dialect

One of the options that was heavily considered was converting the `GraphDef` to a non-proto representation. TFG is an MLIR dialect that has exact roundtrip conversions to/from `GraphDef`, and is fairly mature. The issue here is that the MLIR Bytecode format is not yet mature enough for `SavedModel` - close teams (Foundations and DNA) will be investigating this format for the cross-framework serving use case. Eventually `SavedModel` should integrate with MLA and be able to load functions defined using MLIR, but we are close to this stage yet.

##### Community Suggestions

###### Applicable

- Splitting on a common repeated field (GraphDefs consist of repeated nodes)
- Using `ByteSizeLong` to measure the total size of (Applicable - GraphDefs)
  - `ByteSizeLong` returns the correct size if > 2GB
- Splitting

###### Inapplicable

- Structuring data like rows in a database (Nodes in GraphDefs are interdependent, unlike datasets where rows are independent)
  - From RecordIO Overview: "Why Should You NOT Use RecordIO? RecordIO is designed to be fault-tolerant. Bad records may be skipped without causing read errors. As a result, RecordIO is not ideal for data streams where (silently) skipping one record has an effect on total correctness or the meaning of later records. "
- Streaming (this is a persistent format on disk)

### Performance Implications

#### Splitting Algorithm

Computing the size of protos can take a long time, especially if called repeatedly for each branch of a nested structure. We can use `GetCachedSize` after a call to `ByteSizeLong` to get the cached sizes. `SerializeWithCachedSizesToArray` can be used if messages are directly serialized to bytes before being passed to the Riegeli writer.

#### Riegeli Compression

Riegeli has many compression options, one of which is storing protos transposed. Thus, a future optimization is storing protos in a separate riegeli file from other bytes. In the first prototype, we will simply store everything into a single file as bytes.

#### Lazy Parsing

With the way that the `ChunkedMessage` is defined, we can easily tell which fields have been chunked / not chunked, and which blobs contain the data for the fields. We can use this to implement a lazy proto reader which parses and joins fields when directly requested.

### Dependencies (TODO)
* Dependencies: does this proposal add any new dependencies to TensorFlow?
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

### Engineering Impact (TODO)
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

### Platforms and Environments (SKIP)
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### Best Practices (SKIP)
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples (TODO)
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged. 

### Compatibility (TODO)
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact (TODO)
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design (SKIP)

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics (SKIP)

Seed this with open questions you require feedback on from the RFC process.
