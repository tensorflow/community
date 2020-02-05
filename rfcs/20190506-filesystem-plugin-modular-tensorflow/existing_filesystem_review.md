# Filesystem operations in TensorFlow

This document accompanies the [modular filesystem plugin][filesystem_rfc] RFC
and describes in full details how TensorFlow accesses filesystems at various
levels of implementation.

The implementation of a filesystem support in TensorFlow is hairy, due to the
complexity of operations that needs to be provided. This section is a full
overview of the architecture.

TensorFlow’s support for filesystems can be decomposed into 3 types of API:

* Low level API mainly focused on offering interfaces to hide complexity and
  provide cross-platform compatibility;
* Convenience API providing functions to read/write files in a generic way as
  well as providing support for compressed data and for buffered I/O;
* High level API which offers functionality needed by the kernels, ops and other
  utilities.

Furthermore, the codebase has some additional methods to access the filesystem.
These are used sporadically in some kernels and ops or for special cases such as
very special filesystems or test environments. We can translate these additional
APIs to use the design presented here, unifying API all across the codebase.

![Filesystem layers](20190506-filesystem-plugin-modular-tensorflow/big_picture.png "Overview of filesystem support")

This diagram illustrates this decomposition. Classes are full gray rectangles,
interfaces are blue shaded ones and functions are rectangles with a dashed
border. Arrows represent inheritance and the line between `Env` and `FileSystem`
illustrates the coupling between these two structures as will be described in
the next subsection.

## Low level filesystem API

This subsection is present in the main document too, but it is included here
with some additional contents.

The low-level filesystem API in TensorFlow guarantees the same interface both
across multiple operating systems as well as across multiple filesystems hiding
the complexities arising from the differences between these platforms. Thus,
there are two different things that need consideration: implementing a new
filesystem and making it accessible for each operating system (i.e.
registering).

### The filesystem implementation

In the world before this proposal, a new filesystem is implemented by
subclassing 4 interfaces: one for the filesystem operation ([`FileSystem`][class
FileSystem]), two for the file operations ([`RandomAccessFile`][class
RandomAccessFile], [`WritableFile`][class WritableFile]) and one for read only
memory mapped files ([`ReadOnlyMemoryRegion`][class ReadOnlyMemoryRegion]).

The `Filesystem` interface is needed to define common functionality to create or
delete files and directories as well as manipulating directory contents and
getting information about files present on a filesystem: listing directory
contents, getting statistics about files and finding paths matching a globbing
pattern.

```cpp
class Filesystem {
  // File creation
  virtual Status NewRandomAccessFile(const string& fname, std::unique_ptr<RandomAccessFile>* result) = 0;
  virtual Status NewWritableFile(const string& fname, std::unique_ptr<WritableFile>* result) = 0;
  virtual Status NewAppendableFile(const string& fname, std::unique_ptr<WritableFile>* result) = 0;
  virtual Status NewReadOnlyMemoryRegionFromFile(const string& fname, std::unique_ptr<ReadOnlyMemoryRegionFile>* result) = 0;

  // Creating directories
  virtual Status CreateDir(const string& dirname) = 0;
  virtual Status RecursivelyCreateDir(const string& dirname);

  // Deleting
  virtual Status DeleteFile(const string& fname) = 0;
  virtual Status DeleteDir(const string& dirname) = 0;
  virtual Status DeleteRecursively(const string& dirname, int64* undeleted_files, int64* undeleted_dirs);

  // Changing directory contents
  virtual Status RenameFile(const string& src, const string& target) = 0;
  virtual Status CopyFile(const string& src, const string& target);

  // Filesystem information
  virtual Status FileExists(const string& fname) = 0;
  virtual bool FilesExist(const std::vector<string>& files, std::vector<Status>* status);
  virtual Status GetChildren(const string& dir, std::vector<string>* result) = 0;
  virtual Status Stat(const string& fname, FileStatistics* stat) = 0;
  virtual Status IsDirectory(const string& fname);
  virtual Status GetFileSize(const string& fname, uint65* file_size) = 0;

  // Globbing
  virtual Status GetMatchingPaths(const string& pattern, std::vector<string>* results) = 0;

  // Misc
  virtual void FlushCaches();
  virtual string TranslateName(const string& name) const;
};
```

In TensorFlow, files are identified by a URI of form
`[<scheme>://[<host>]]<filename>` and `scheme` is always used to identify which
implementation to use for accessing them (for example, a path specified as
`hdfs:///path/to/file` will use the Hadoop filesystem implementation).

Because sometimes we only need the filename part, the FileSystem API defines
`TranslateName()` which also ensures that paths are canonical, properly
resolving `.` and `..` entries that might be present in the path. Although
filesystems can reimplement this, by default
[`tensorflow::lib::io::CleanPath()`][CleanPath] is used.

The default `CopyFile()` implementation just calls the
[`FileSystemCopyFile()`][FileSystemCopyFile] helper method using the same
`FileSystem` argument for both source and target.

The [`FileStatistics`][class FileStatistics] class is just a value class
containing information related to a dentry:

```cpp
class FileStatistics {
  int64 length = -1;
  int64 mtime_nsec = 0;
  bool is_directory = false;
};
```

There is also a [`TF_FileStatistics`][struct TF_FileStatistics] data structure
used by the C API. Both have the same values but the `TF_FileStatistics`
structure doesn’t have constructors/destructors as it has to be pure C.

The other 3 interfaces that need to be subclassed to get a functional filesystem
are simple:

```cpp
class RandomAccessFile {
  virtual Status Name(StringPiece* result) const;
  virtual Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const = 0;
};

class WritableFile {
  virtual Status Name(StringPiece* result) const;
  virtual Status Append(StringPiece data) = 0;
  virtual Status Append(const absl::Cord& cord);
  virtual Status Tell(int64* position);
  virtual Status Close() = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;
};

class ReadOnlyMemoryRegion {
  virtual const void* data() = 0;
  virtual uint64 length() = 0;
};
```

Once  these 4 interfaces are implemented, we have a fully functioning filesystem
at the lowest API level. Each such filesystem can be tested in isolation from
the rest of TensorFlow.

If a filesystem doesn’t support one of the file types, then we don’t need to
implement the corresponding interface. Instead, the method creating the file of
that type can just error, as in this [HadoopFileSystem][HadoopFileSystem]
example.

There are multiple filesystems supported by TensorFlow, some of which are
present in the next diagram:

![Example filesystems](20190506-filesystem-plugin-modular-tensorflow/filesystem.png "Example filesystems")

### Cross operating system compatibility

The operating system compatibility is ensured by the [`Env`][class Env]
interface. This contains API calls used to access functionality that depends on
the platform the process runs on: threading, clock information, loading dynamic
libraries and filesystem methods. We will focus only on the latter in this
document. As the number of environments is usually very small, we can achieve
minimal code by using preprocessor guards and compiling only for one platform.
Alternatively, we can have a similar design as this one and create plugins for
each operating system, although this will require additional work in defining
interfaces and versioning and won’t bring much benefit over the preprocessor
macros approach.

Restricting only to the filesystem related API, the `Env` interface looks like
the following code snippet:

```cpp
class Env {
  // Filesystem registration
  virtual Status GetFileSystemForFile(const string& fname, FileSystem** result);
  virtual Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes);
  virtual Status RegisterFileSystem(const string& scheme, FileSystemRegistry::Factory factory);

  // Creating files, including memory mapped
  Status NewRandomAccessFile(const string& fname, std::unique_ptr<RandomAccessFile>* result);
  Status NewWritableFile(const string& fname, std::unique_ptr<WritableFile>* result);
  Status NewAppendableFile(const string& fname, std::unique_ptr<WritableFile>* result);
  Status NewReadOnlyMemoryRegionFromFile(const string& fname, std::unique_ptr<ReadOnlyMemoryRegionFile>* result);

  // Creating directories
  Status CreateDir(const string& dirname);
  Status RecursivelyCreateDir(const string& dirname);

  // Deleting
  Status DeleteFile(const string& fname);
  Status DeleteDir(const string& dirname);
  Status DeleteRecursively(const string& dirname, int64* undeleted_files, int64* undeleted_dirs);

  // Changing directory contents
  Status RenameFile(const string& src, const string& target);
  Status CopyFile(const string& src, const string& target);

  // Filesystem information
  Status FileExists(const string& fname);
  bool FilesExist(const std::vector<string>& files, std::vector<Status>* status);
  Status GetChildren(const string& dir, std::vector<string>* result);
  Status Stat(const string& fname, FileStatistics* stat);
  Status IsDirectory(const string& fname);
  Status GetFileSize(const string& fname, uint64* file_size);

  // Globbing
  virtual bool MatchPath(const string& path, const string& pattern) = 0;
  virtual Status GetMatchingPaths(const string& pattern, std::vector<string>* results);

  // Misc
  Status FlushFileSystemCaches();
  string GetExecutablePath();
  virtual string GetRunfilesDir() = 0;
  bool LocalTempFilename(string* filename);
  bool CreateUniqueFileName(string* prefix, const string& suffix);
  virtual void GetLocalTempDirectories(std::vector<string>* list) = 0;
  static Env* Default();

  // Other methods of the class, not relevant here
};
```

Each process accesses the environment by calling the static `Env::Default()`
method and then gains access to all the functionality that depends on the
operating system. Then, to operate on the filesystem, any process will either
use the methods provided by the `Env` API or those of the `FileSystem` object
obtained from `GetFileSystemForFile()`. In fact, most of the default
implementations of the methods above use `GetFileSystemForFile()` and then
delegate to the filesystem.

Note that in the case of `RenameFile()` and `CopyFile()` the target and the
source files might be in a different filesystem. Renaming results in an error
(as it is currently not implemented), whereas copying is done via the
`FileSystemCopyFile()` helper introduced in the previous section.

The `GetMatchingPaths()` call can be used to identify all paths in the current
filesystem that match a specific pattern. The default implementation uses
[`GetMatchingPaths()`][GetMatchingPaths]:

The following diagram shows several of the implementations of the `Env`
interface:

![Example environments](20190506-filesystem-plugin-modular-tensorflow/env.png "Example environments")

### Registering a filesystem to the environment

In order to access a filesystem via `GetFileSystemForFile()`, first it has to be
registered to the environment. This can be done via `RegisterFileSystem()` but
it is preferable to use the [`REGISTER_FILE_SYSTEM(scheme,
factory)`][REGISTER_FILE_SYSTEM] macro instead, where `factory` is an empty
constructor of a subclass of `FileSystem` and `scheme` is the URI scheme needed
to access the file. Any filesystem call using `scheme` in the URI will then get
handled by the subclass used for registration.

Filesystems are registered into a [`FileSystemRegistry`][class
FileSystemRegistry], which is semantically equivalent to a map from file scheme
to filesystem. Since this has to be unique over the lifetime of the program, the
interface is [opaque outside of Env][FileSystemRegistry init].

```cpp
class FileSystemRegistry {
  typedef std::function<FileSystem*()> Factory;
  virtual Status Register(const string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const string& scheme) = 0;
  virtual Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) = 0;
};
```

## Convenience API

All the functionality presented until this point is enough to transparently
operate with files across filesystems and operating systems. However, each such
operation requires getting the environment via Env::Default() then accessing the
needed functionality from a method of that class. This quickly becomes
repetitive, so TensorFlow has some convenience API that can be used. For
example, there are these helper methods:

```cpp
Status ReadFileToString(Env* env, const string& fname, string* data);
Status ReadBinaryProto(Env* env, const string& fname, MessageLite* proto);
Status ReadTextProto(Env* env, const string& fname, protobuf::Message* proto);

Status WriteStringToFile(Env* env, const string& fname, const StringPiece& data);
Status WriteBinaryProto(Env* env, const string& fname, const protobuf::MessageLite& proto);
Status WriteTextProto(Env* env, const string& fname, const protobuf::Message& proto);
```

### Buffered (streaming) support

Furthermore, as some files can be large, we need a way to stream the data from
them. There are several options that we can use:

An [`InputBuffer`][class InputBuffer] is a buffer on top of a RandomAccessFile
passed to the constructor. Its API is simple:

```cpp
class InputBuffer {
  Status ReadLine(string* result);
  Status ReadNBytes(int64 bytes_to_read, string* result);
  Status ReadNBytes(int64 bytes_to_read, char* result, size_t* read);
  Status ReadVarint32(uint32* result);
  Status ReadVarint64(uint64* result);
  Status SkipNBytes(int64 bytes_to_skip);
  Status Seek(int64 position);
  int64 Tell() const;
};
```

However, the `InputBuffer` interface is too tied to a `RandomAccessFile` and
thus it is only being used in one place, by the `TextLineReaderOp` kernel.

A replacement method is provided by [`InputStreamInterface`][class
InputStreamInterface]:

```cpp
class InputStreamInterface {
  virtual Status ReadNBytes(int64 bytes_to_read, string* result) = 0;
  virtual Status SkipNBytes(int64 bytes_to_skip);
  virtual int64 Tell() const = 0;
  virtual Status Reset() = 0;
};
```

A [`RandomAccessInputStream`][class RandomAccessInputStream] object is an
`InputStreamInterface` object which wraps a `RandomAccessFile`. To provide more
functionality (e.g., `ReadLine()`, `Seek()`), this can be wrapped by a
[`BufferedInputStream`][class BufferedInputStream].

### File compression APIs

Another descendant of `InputStreamInterface` is [`MemoryInputStream`][class
MemoryInputStream] which wraps a memory buffer and is used in
`DecodeCompressedOp`. This doesn’t use filesystem operations, although it can be
implemented in terms of `ReadOnlyMemoryRegion` if needed.

There are two more relevant compression APIs built from `InputStreamInterface`.
First, [`SnappyInputBuffer`][class SnappyInputBuffer] provides access to files
compressed using Snappy and represented as `RandomAccessFile` objects. There is
also a [`SnappyOutputBuffer`][class SnappyOutputBuffer] class to write
compressed input, but there is no `OutputStreamInterface`.

The second compression API is given by [`ZlibInputStream`][class
ZlibInputStream]. Similar to the `RandomAccessInputStream` case, this just wraps
an  `InputStreamInterface`. To write to compressed zlib files, we can use
[`ZlibOutputBuffer`][class ZlibOutputBuffer] to wrap around a `WritableFile`.

## High level filesystem API

Building on top of the convenience API, TensorFlow code has several custom
classes and helpers to provide high level access to the filesystem.

For example `tf.data` uses `TFRecord` support provided by [`RecordReader`][class
RecordReader], [`RecordWriter`][class RecordWriter] and
[`SequentialRecordReader`][class SequentialRecordReader] to build the
[`FileDataset`][class FileDataset] and [`FileIterator`][class FileIterator]
abstractions (and subclasses such as [`FileWriterIterator`][class
FileWriterIterator], [`FileReaderIterator`][class FileReaderIterator]). Then,
kernels and ops such as [`TextLineDatasetOp`][class TextLineDatasetOp] and
[`TFRecordDatasetOp`][class TFRecordDatasetOp] are built and then used in Python
APIs.

Other examples include:

* [`DumpGraphDefToFile`][DumpGraphDefToFile],
  [`DumpGraphToFile`][DumpGraphToFile],
  [`DumpFunctionDefToFile`][DumpFunctionDefToFile];
* [`Table`][class Table] - immutable mapping from strings to strings backed up
  by a file;
* [`BundleWriter`][class BundleWriter], [`BundleReader`][class BundleReader] -
  checkpointing;
* [`TensorSliceWriter`][class TensorSliceWriter], [`TensorSliceReader`][class
  TensorSliceReader] - checkpointing partitioned variables;
* [`LoadSavedModel`][LoadSavedModel]  - `SavedModel` loading;
* [`EventsWriter`][class EventsWriter];
* [`DebugFileIO`][class DebugFileIO];
* API generation tools inside TensorFlow.

## Additional APIs

For some special cases or due to historical reasons, TensorFlow has a set of
filesystem APIs which cannot be directly mapped on the hierarchy described so
far. We will mention them here for completeness, but it turns out we can either
ignore them for the rest of the design as they don’t require many changes.

### `FileBlockCache`

The [`FileBlockCache`][class FileBlockCache] class can be used by filesystems to
cache files at the block level. Each entry in the cache is keyed by the filename
and the offset of the block in the file. Filesystem implementations are free to
ignore this class, but we’re mentioning it in this document as it is being used
by the cloud filesystems.

In the new filesystem design, filesystems needing this functionality can just
import the corresponding targets. Since this is an implementation detail, the
caching information doesn’t need to cross the module boundary.

### `ReaderInterface`

The [`ReaderInterface`][class ReaderInterface] and its subclasses provides an
alternative method of reading records from files in a TensorFlow graph. It has
the following API:

```cpp
class ReaderInterface : public ResourceBase {
  // Reads one single record
  virtual void Read(QueueInterface* queue, string* key, string* value, OpKernelContext* context) = 0;
  // Reads up to num_records
  virtual int64 ReadUpTo(const int64 num_records, QueueInterface* queue, std::vector<string>* keys, std::vector<string>* values, OpKernelContext* context) = 0;

  // Misc
  virtual Status Reset() = 0;
  virtual int64 NumRecordsProduced() = 0;
  virtual Status SerializeState(string* state) = 0;
  virtual Status RestoreState(const string& state) = 0;
};
```

The only subclass in TensorFlow is [`ReaderBase`][class ReaderBase], which adds
API entry-points for reading while a mutex is being held:

```cpp
class ReaderBase : public ReaderInterface {
  virtual Status ReadLocked(string* key, string* value, bool* produced, bool* at_end) = 0;
  virtual Status ReadUpToLocked(int64 num_records, std::vector<string>* keys, std::vector<string>* values, int64* num_read, bool* at_end);
};
```

In this subclass, the `Read()` and `ReadUpTo()` methods are implemented in terms
of their corresponding locked methods. Furthermore, descendants of this class
only need to implement `ReadLocked()`, the only pure virtual member of the API.

Before this proposal, there are 7 descendants of `ReaderBase` and they are used
to implement kernels and ops. All of these kernels inherit from
[`ReaderOpKernel`][class ReaderOpKernel] which handles access to the source to
be read from. Whereas [`TextLineReader`][class TextLineReader],
[`TFRecordReader`][class TFRecordReader] and [`WholeFileReader`][class
WholeFileReader] access the filesystem using the environment approach
illustrated above, the other subclasses either work with in-memory data
([`IdentityReader`][class IdentityReader], [`LMDBReader`][class LMDBReader]),
with data obtained from a remote call ([`BigQueryReader`][class BigQueryReader])
or with data that is read using the `InputStreamInterface` mechanism described
previously ([`FixedLengthRecordReader`][class FixedLengthRecordReader]).

In the modular design, most of these descendants will be converted for free. The
only class that might not be trivial to transfer to the new API might be
`BigQueryReader`. But, since before this proposal this class completely ignores
the filesystem APIs, we can continue with this design ignoring it for the
moment.

### `SummaryWriterInterface`

A [`SummaryWriterInterface`][class SummaryWriterInterface] allows writing
TensorFlow resources to a summary file, being a subclass of `ResourceBase`, just
like `ReaderInterface`.

```cpp
class SummaryWriterInterface : public ResourceBase {
  virtual Status Flush() = 0;
  virtual Status WriteTensor(int64 global_step, Tensor t, const string& tag, const string& serialized_metadata) = 0;
  virtual Status WriteScalar(int64 global_step, Tensor t, const string& tag) = 0;
  virtual Status WriteHistogram(int64 global_step, Tensor t, const string& tag) = 0;
  virtual Status WriteImage(int64 global_step, Tensor t, const string& tag, int max_images, Tensor bad_color) = 0;
  virtual Status WriteAudio(int64 global_step, Tensor t, const string& tag, int max_outputs_, float sample_rate) = 0;
  virtual Status WriteGraph(int64 global_step, std::unique_ptr<GraphDef> graph) = 0;
  virtual Status WriteEvent(std::unique_ptr<Event> e) = 0;
};
```

When writing to files, we use the [`SummaryFileWriter`][class SummaryFileWriter]
subclass which uses the `Env` method. However, the other subclass,
[`SummaryDbWriter`][class SummaryDbWriter], uses an `Sqlite` object to talk to a
SQL database. Similar to the `BigQueryReader` case from the previous section, we
can postpone converting this to the new filesystem API until a later time.

## IO from external libraries or other languages

To provide functionality to read/write image/audio files, TensorFlow uses some
external libraries. Although they can do their own IO, there is glue code for
[gif][gif io], [jpeg][jpeg io], [png][png io] and [wav][wav io] files. This glue
ensures that the calls into the library use strings representing file contents
instead of letting the library do its own IO. This way, the filesystem support
that TensorFlow does is the only one accessing files on disk.

However, there is nothing stopping future code development from directly reading
from a file on the disk (for example), either in the C/C++ code or in the Python
code. Preventing this is out of scope for the current design document.

## Existing C API implementation

There is an [existing C API for filesystems][c api] which we might reuse in this
design. We cannot build on top of this API as it only offers C entry points
which then call the existing C++ API functions. But, it gives us an indication
on what API we should provide, at the minimum:

```cpp
typedef struct TF_FileStatistics {
  int64_t length;
  int64_t mtime_nsec;
  bool is_directory;
} TF_FileStatistics;

// filesystem manipulation
TF_CAPI_EXPORT extern void TF_CreateDir(const char* dirname, TF_Status* status);
TF_CAPI_EXPORT extern void TF_DeleteDir(const char* dirname, TF_Status* status);
TF_CAPI_EXPORT extern void TF_DeleteRecursively(const char* dirname, int64_t* undeleted_file_count, int64_t* undeleted_dir_count, TF_Status* status);
TF_CAPI_EXPORT extern void TF_DeleteFile(const char* filename, TF_Status* status);
TF_CAPI_EXPORT extern void TF_FileStat(const char* filename, TF_FileStatistics* stats, TF_Status* status);

// writable files API
TF_CAPI_EXPORT extern void TF_NewWritableFile(const char* filename, TF_WritableFileHandle** handle, TF_Status* status);
TF_CAPI_EXPORT extern void TF_CloseWritableFile(TF_WritableFileHandle** handle, TF_Status* status);
TF_CAPI_EXPORT extern void TF_SyncWritableFile(TF_WritableFileHandle** handle, TF_Status* status);
TF_CAPI_EXPORT extern void TF_FlushWritableFile(TF_WritableFileHandle** handle, TF_Status* status);
TF_CAPI_EXPORT extern void TF_AppendWritableFile(TF_WritableFileHandle** handle, const char* data, size_t length, TF_Status* status);
```

This API is very incomplete, for example it lacks support for random access
files and memory mapped files.  Furthermore, the `TF_WritableFileHandle` object
is just an opaque struct (in C-land) which maps over a `WritableFile` pointer
from the C++ implementation.

Since this has to be pure C, Status objects are replaced by pointers to a
`TF_Status` structure. There is code to convert from one to the other. Moreover,
these APIs assume that the `TF_Status` object is created and managed from the
calling code, instead of returning the status from every function call. We will
follow a similar design in this proposal.


[filesystem_rfc]: https://github.com/tensorflow/community/pull/101 "RFC: Modular Filesystems C API"

[class FileSystem]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L46 "class FileSystem"
[class RandomAccessFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L232 "class RandomAccessFile"
[class WritableFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L271 "class WritableFile"
[class ReadOnlyMemoryRegion]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L342 "class ReadOnlyMemoryRegion"
[class Env]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.h#L48 "class Env"
[class FileSystemRegistry]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L360 "class FileSystemRegistry"
[class FileStatistics]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_statistics.h#L23 "class FileStatistics"
[struct TF_FileStatistics]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/c/env.h#L36 "struct TF_FileStatistics"
[class InputBuffer]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/inputbuffer.h#L32 "class InputBuffer"
[class InputStreamInterface]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/inputstream_interface.h#L27 "class InputStreamInterface"
[class RandomAccessInputStream]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/random_inputstream.h#L27 "class RandomAccessInputStream"
[class BufferedInputStream]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/buffered_inputstream.h#L27 "class BufferedInputStream"
[class MemoryInputStream]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/decode_compressed_op.cc#L30 "class MemoryInputStream"
[class SnappyInputBuffer]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/snappy/snappy_inputbuffer.h#L35 "class SnappyInputBuffer"
[class SnappyOutputBuffer]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/snappy/snappy_outputbuffer.h#L46 "class SnappyOutputBuffer"
[class ZlibInputStream]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/zlib_inputstream.h#L40 "class ZlibInputStream"
[class ZlibOutputBuffer]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/zlib_outputbuffer.h#L38i "class ZlibOutputBuffer"
[class RecordReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/record_reader.h#L59 "class RecordReader"
[class RecordWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/record_writer.h#L50 "class RecordWriter"
[class SequentialRecordReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/record_reader.h#L120 "class SequentialRecordReader"
[class FileDataset]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/cache_dataset_ops.cc#L50 "class FileDataset"
[class FileIterator]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/cache_dataset_ops.cc#L112 "class FileIterator"
[class FileWriterIterator]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/cache_dataset_ops.cc#L195 "class FileWriterIterator"
[class FileReaderIterator]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/cache_dataset_ops.cc#L439 "class FileReaderIterator"
[class TextLineDatasetOp]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/reader_dataset_ops.cc#L35 "class TextLineDatasetOp"
[class TFRecordDatasetOp]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/data/reader_dataset_ops.cc#L740 "class TFRecordDatasetOp"
[class Table]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/table.h#L34 "class Table"
[class BundleWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/tensor_bundle/tensor_bundle.h#L108 "class BundleWriter"
[class BundleReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/tensor_bundle/tensor_bundle.h#L182 "class BundleReader"
[class TensorSliceWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/tensor_slice_writer.h#L43 "class TensorSliceWriter"
[class TensorSliceReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/tensor_slice_reader.h#L54 "class TensorSliceReader"
[class EventsWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/events_writer.h#L31 "class EventsWriter"
[class DebugFileIO]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/debug/debug_io_utils.h#L147 "class DebugFileIO"
[class FileBlockCache]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/cloud/file_block_cache.h#L39 "class FileBlockCache"
[class ReaderInterface]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/framework/reader_interface.h#L44 "class ReaderInterface"
[class ReaderBase]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/framework/reader_base.h#L30 "class ReaderBase"
[class ReaderOpKernel]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/framework/reader_op_kernel.h#L35 "class ReaderOpKernel"
[class TextLineReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/text_line_reader_op.cc#L28 "class TextLineReader"
[class TFRecordReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/tf_record_reader_op.cc#L28 "class TFRecordReader"
[class WholeFileReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/whole_file_read_ops.cc#L45 "class WholeFileReader"
[class IdentityReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/identity_reader_op.cc#L29 "class IdentityReader"
[class LMDBReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/lmdb_reader_op.cc#L27 "class LMDBReader"
[class BigQueryReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/contrib/cloud/kernels/bigquery_reader_ops.cc#L52 "class BigQueryReader"
[class FixedLengthRecordReader]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/fixed_length_record_reader_op.cc#L33 "class FixedLengthRecordReader"
[class SummaryWriterInterface]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/kernels/summary_interface.h#L30 "class SummaryWriterInterface"
[class SummaryFileWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/summary/summary_file_writer.cc#L30 "class SummaryFileWriter"
[class SummaryDbWriter]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/summary/summary_db_writer.cc#L875 "class SummaryDbWriter"

[CleanPath]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/io/path.h#L74 "tensorflow::lib::io::CleanPath()"
[FileSystemCopyFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.h#L430 "FileSystemCopyFile()"
[GetMatchingPaths]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system_helper.h#L45 "GetMatchingPaths()"
[DumpGraphDeftoFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/dump_graph.h#L36 "DumpGraphDefToFile()"
[DumpGraphToFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/dump_graph.h#L41 "DumpGraphToFile()"
[DumpFunctionDefToFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/util/dump_graph.h#L47 "DumpFunctionDefToFile()"
[LoadSavedModel]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/cc/saved_model/loader.h#L50 "LoadSaveDModel()"

[HadoopFileSystem]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/hadoop/hadoop_file_system.cc#L368-L377 "HadoopFileSystem::NewReadOnlyMemoryRegionFromFile()"
[FileSystemRegistry init]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.cc#L93 "initialization of FileSystemRegistry"

[REGISTER_FILE_SYSTEM]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.h#L480 "REGISTER_FILE_SYSTEM(scheme, factory)"

[gif io]: https://github.com/tensorflow/tensorflow/tree/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/gif
[jpeg io]: https://github.com/tensorflow/tensorflow/tree/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/jpeg
[png io]: https://github.com/tensorflow/tensorflow/tree/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/png
[wav io]: https://github.com/tensorflow/tensorflow/tree/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/lib/wav

[c api]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/c/env.h

[load_library]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/python/framework/load_library.py#L132
