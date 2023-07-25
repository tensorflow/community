# Unbound SavedModel

| Status        | Implemented                                                                                  |
:-------------- |:-------------------------------------------------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Kathy Wu (<kathywu@google.com>), Adam Cogdell (<adamcogdell@google.com>)                     |
| **Sponsor**   | Ivan Petrychenko (<petrychenko@google.com>)                                                  |
| **Updated**   | 2023-07-24                                                                                   |

## Objective

We are proposing a new format for `SavedModel` and a generic proto-splitting library that resolves the 2GB proto issue. The purpose of this RFC is to publicize the design of this new format, which has been implemented (see [tensorflow/tools/proto_splitter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/proto_splitter)), but is open to comments and changes from the open source community.

## Motivation

The 2GB proto serialization limitation is an ongoing problem that has plagued [many projects](https://discuss.tensorflow.org/t/fix-the-notorious-graphdef-2gb-limitation/12572). We have evaluated many solutions and have decided on splitting the proto as the short-term solution.

For SavedModels, the most common causes of large protos are large embedding constants and a large number of the NodeDef/FunctionDefs. Current workarounds for models >2GB result in significantly slower step times, so we have developed a solution: a new format for serializing the `SavedModel` proto.

## User Benefit

As models get larger, the likelihood of encountering the 2GB limit increases. Many users inside and outside Google have reported this issue, and were met with no solution, resulting in hacks and workarounds. With this change, current and future model builders are unrestricted by the size of the SavedModel proto.

## Design Proposal

### SavedModel Changes

A new option has been introduced to `tf.saved_model.save`, which is capable of saving SavedModels that exceed the regular GraphDef size limitation. SavedModels produced with this functionality enabled can only be consistently read from the TF2 C++ / Python APIs. If the SavedModel proto does not exceed 2GB, then the output format will still be a single binary proto profile and remains compatible with TF1 APIs (although this could change in the future).

There are many customers who could benefit from this option, but downstream libraries may require updates to their code.

#### API Change

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

#### File Change

| Old [SavedModel format]   | New [Image format]                               |
| ------------------------- | ------------------------------------------------ |
| saved_model_dir/<br>├ variables/<br>├ assets/<br>├ fingerprint.pb<br>├ saved_model.pb | saved_model_dir/<br>├ variables/<br>├ assets/<br>├ fingerprint.pb<br>├ saved_model.cpb

### Proto-splitting library

Proto-splitting is not specific to TensorFlow, so had to decide how generic our library should be. A pure proto-splitter would simply split each field >2GB into its own chunk, but this approach lacks effeciciency. Knowing the structure of the message beforehand allows the algorithm to skip several steps of the proto message tree traversal to find which fields to split. For example, in the SavedModel case, the most common causes of large protos are: (1) large embedding constants, and (2) a large number of the repeated NodeDef/Function field. An algorithm optimized for splitting GraphDef would only look at "const"-type nodes, and the repeated NodeDef/Function fields. A generic algorithm would be looking at each field in GraphDef, most of which do not have issues with size.

We have added a proto-splitting library that walks the line between generic and specialized, and can handle splitting, writing, reading, and merging for both Python and C++. This library implements a generic `Splitter` class that can handle all message types, which users may use to implement their own specialized splitters for specific message types. The `Merger` class is fully generic, and can merge any chunked proto output by a `Splitter`.

Users can make use of the python `Splitter` implementations by calling:

```python
splitter = MySplitterClass(proto)

# Export the chunks to a file.
splitter.write(file_prefix)

# Access the chunks created in the splitter.
chunks, chunked_message = splitter.split()
```

And in C++:

```c++
MyProto proto = ConstructMyProto();
MySplitterClass splitter = MySplitterClass(&proto);

// Export the chunks to a file.
splitter.Write(file_prefix);

// Access the chunks created in the splitter.
ASSERT_OK_AND_ASSIGN(auto ret, splitter.Split());
vector<MessageBytes>* chunks = ret.first;
ChunkedMessage chunked_message = ret.second;
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

The last chunk in the `{PREFIX}.cpb` file is always a set of serialized chunks followed by a `ChunkedMetadata` message. In the initial version of the splitter, all chunks are stored in a single file, but in the future there may be other outputted files that follow the pattern `{PREFIX}.*.cpb`. The main file will always be `{PREFIX}.cpb`.

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
  // Index of the ChunkInfo in ChunkMetadata.chunks.
  optional uint64 chunk_index = 1;

  // The same field can appear multiple times in this list. When this
  // happens, the parsed chunk data is merged in the order that they appear.
  // The way the field is merged depends on the type of field. e.g. Repeated
  // fields are merged by appending them, bytes are merged concatenated.
  repeated ChunkedField chunked_fields = 2;
}

message ChunkedField {
  // Stores the nested field numbers, which tell us where and how to merge
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

We chose to use the proto format here for familiarity. In the event that even `ChunkedMessage` exceeds 2GB, we can (in the future) add recursion to `Splitter.Split()` to chunk the output proto until it is less than 2GB. The version field is there to handle this future feature.

##### Versioning

The outputs of `Splitter.Split()` (`ChunkMetadata` and `ChunkedMessage`) have a version field to handle future updates.

###### Hypothetical version updates

- The initial version of `Splitter` writes a single riegeli file, but in a new version it outputs several files.
- The `ChunkedMessage` itself could exceed 2GB. In a new version of `Splitter`, the `ChunkedMessage` can be recursively chunked and recursively merged when read.
- Riegeli is no longer used. `*.chunk.riegli` is kept so that previous versions of the code can still read it. Old code will read the `VersionDef` and see that it can no longer process the data, and report an understandable error.
- The initial version of `Splitter` returns serialized proto chunks, but in a new version, returns a chunk reference that is actually a URL to an external file.

### Alternatives Considered

We decided to explore different serialization formats, even outside of proto. There haven't been any major changes to `SavedModel` in ages - lots of libraries are built on top of the `saved_model.pb` or a `GraphDef` file, and are parsing that proto directly. Any change we make to the format now is almost equally painful, and since we've decided to do this, we explored all options out there.

The conclusion was that re-using proto is the best, as it gave us an easy solution to compatibility so that we can focus on other areas. However, even splitting the proto comes with compatibility problems: What happens when we need to split the proto again for constants? This breaks compatibility *again*. Also in the future, are we anticipating more cross-framework models (JAX2TF is using a workaround by storing StableHLO as an attribute in a custom op)? Why not invest this time in MLIR, which supports multiple dialects and will never hit a file size limit?

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

The problem is that many existing converters and libraries parse the proto directly from file, or utilize the proto as some in-memory representation. When a new ref field is added, the existing code will ignore this field unless they are updated. They will see an incomplete Graph, or a constant node with no specified value. These differences in the representation will bubble up as errors later down the line (for example, when tracing the graph to find connected ops, or trying to create the constant tensor).

#### Alternate Formats

We explored written formats for both the image dump and function graphs (which are represented as Proto data structures in-memory). The format for the image does not necessarily need to be Proto-compatible, but the serialization of the function graph should be.

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

###### RecordIO

- Fault-tolerant to a fault (will silently skip records with errors). This is not permissible in model graphs.
- No 2GB limit

###### CapnProto

- Extra work in translating schema and ensuring they are in sync
- Has a data blob limit of 512 MB
- No 2GB limit
- Very fast

###### Home-brewed format [.tf]

- We have to write our own encoder and decoder
- Efficiency, performance, compatibility are on us
- Homemade format can target versioning. Proto doesn't have versions. Differences in schema are handled automatically by proto encoding/decoding logic, it's easy to edit schema without bumping up any versions [`GraphDef` mitigates this by using a bot that updates the version every day]. Differences in the representation are handled programmatically, leading to legacy code floating around to handle older serialized models.
- Can exceed 2GB as needed

#### MLIR: TFG Dialect

One of the options that was heavily considered was converting the `GraphDef` to a non-proto representation. TFG is an MLIR dialect that has exact roundtrip conversions to/from `GraphDef`, and is fairly mature. The issue here is that the MLIR Bytecode format is not yet mature enough for `SavedModel` - close teams (Foundations and DNA) will be investigating this format for the cross-framework serving use case.

### Performance Implications

#### Splitting Algorithm

Computing the size of protos can take a long time, especially if called repeatedly for each branch of a nested structure. We used `GetCachedSize` after a call to `ByteSizeLong` to get the cached sizes. `SerializeWithCachedSizesToArray` was used if messages were directly serialized to bytes before being passed to the Riegeli writer.

#### Riegeli Compression

Riegeli has many compression options, one of which is storing protos transposed. Thus, a future optimization is storing protos in a separate riegeli file from other bytes. In the first prototype, we simply stored everything into a single file as bytes.

#### Lazy Parsing

With the way that the `ChunkedMessage` is defined, we can easily tell which fields have been chunked / not chunked, and which chunks contain the data for the fields. We can use this to implement a lazy proto reader which parses and merges fields when directly requested. This may be useful for potential cases with edge devices where the entire serialized model cannot be loaded into memory at once.

### Dependencies

This change brings the Riegeli dependency to TensorFlow, and uses it as the new underlying format for SavedModels.

### Engineering Impact

SavedModels are no longer simply serialized and deserialized, but now go through an intermediary splitting and merging stage. The underlying file format has also changed from a standard protobuf to a riegeli protobuf. One might expect the worst case that this will result in larger files and longer save/load times, but it turns out to be the opposite! Riegeli compression results in gains across the board. With a testing model that began as a `.pb` of size 427MB, we see a `.cpb` size of only 16MB with only 13s needed to chunk the model. A 674MB `.pb` model was reduced to a `.cpb` of size 27MB, and its load time fell from 2m58s to 2m28s, only 3.4s of which was merging.

Since this code was developed by the Persistence and Lifecycle team (formerly TensorFlow SavedModel), it will be maintained by that team. It was specifically not placed in the `tensorflow/python/saved_model` or `tensorflow/cc/saved_model` directories, since it is, at its core, a generic proto splitting api. If a user decides to subclass the `ComposableSplitter`, they too can serialize protos that exceed the 2GB limit. As such, the majority of the `proto_splitter` code is self-contained and can be tested in isolation. Currently, visibility is restricted to select parts of the TF codebase, but can be extended upon request.

### Tutorials and Examples

A tutorial will be added to tensorflow.org that covers usage and details of this new format.

### Compatibility

This change is to the underlying format of the `SavedModel`, which users shouldn't be concerned with. Users who rely on raw access to the proto have violated the layering of the api, but will still be unaffected, as it was previously impossible to save models >2GB. Additionally, the `experimental_image_format` flag is needed to save with the new format. When the flag is enabled, models <2GB will still not save with the new `.cpb` format, since the splitter defaults to the standard `.pb` in that case. Thus, every precaution has been taken to ensure backwards compatibility.

### User Impact

The vast majority of SavedModel users will be unaffected by this change. Only select teams with use cases for models beyond the 2GB limit will be able to take advantage of the proto splitter via the new `experimental_image_format` flag in the python `tf.saved_model.save` api call.

## Detailed Design

### Public C++ API

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

### Public Python API

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

Recommended steps to subclass `ComposableSplitter` in python (C++ is very similar):

1. (required) Override `build_chunks()`. This method sets the values of `self._chunks` and `self._chunked_message` based on the user-passed proto.
2. Update `version_def`. This is important to ensure that users are able to apply the Merger to the chunked proto, or get understandable version errors.
3. If `__init__` is overridden: call `super().__init__(proto, **kwargs)`. This is optional but highly recommended since it sets up basic attributes that may be needed by other splitters.

## Questions and Discussion Topics

The new format is [implemented](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/proto_splitter), but we're accepting comments and suggestions through this RFC.
