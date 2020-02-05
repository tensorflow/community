# Modular Filesystems C API

| Status        | Accepted                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Mihai Maruseac (mihaimaruseac@google.com)            |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                     |
| **Updated**   | 2019-05-06                                           |

## Abstract

TensorFlow has grown large since its introduction and we are currently working
on ensuring a better development workflow by a team-wide effort designed to make
TensorFlow more modular according to the [current proposal][modular_rfc].

A major module of the modular TensorFlow involves all operations regarding files
and the filesystem. Saving models, checkpointing, reading files as input to
models (i.e. images, audio, etc.), and many other file manipulations are all
tasks that TensorFlow has to support.

While users can use the underlying operating system’s API (such as calling fopen
and friends on Posix environments, for example), it is definitely better to have
a cross-platform, cross-environment API that can be used everywhere. This
becomes more evident when we consider cloud environments: a user of TensorFlow
should not have to compile support for all existing cloud environments if the
program only touches local files.

TensorFlow provides such an API at the moment but it results in compiling code
for all known filesystems in the final binary. We propose a new design where all
filesystems are implemented as plugins and only the required ones are added at
runtime. To do so, we will first need to design a C API for the filesystems and
then integrate this with the rest of the modularization effort.

In a nutshell, the proposed API uses several data structures of function
pointers to mimic the current class-based API. Every plugin will fill in the
appropriate fields of these data structures during registration. When doing a
filesystem operation, core TensorFlow identifies the plugin/filesystem needed to
implement the operation, reads the function pointer from the corresponding table
and executes it. This way, we are able to offer support for all kinds of files,
including memory mapped files, remote files, compressed files or filesystem with
specific rules.

The rest of the document presents a set of requirements that this design tries
to satisfy, a (somewhat very detailed) overview of the existing implementation,
the proposed changes and estimates on the amount of work needed to integrate
this design with TensorFlow code.

## Requirements

Moving to a modular design adds several new requirements to the design, as we
want to prevent code from one plugin to interfere with code from another plugin
or the core TensorFlow. As such, here are a few requirements that we think are
needed:

1. Once a plugin is registered, the function pointer to plugin provided
   implementation should never change. Since unloading shared libraries after
   code from them has been run can lead to bad behavior (resource leaks, copies
   of pointers to functions within the library, object size mismatches, etc.) we
   will not attempt to deregister/unload plugins.

1. Interface between TensorFlow and plugins must be as simple as possible as
   well as thoroughly tested. It is better to localize errors in either
   TensorFlow core or the plugin providing the filesystem functionality than in
   the glue code between the two.

1. Plugin registration should support mandatory metadata such as version
   numbers, checks for backwards and forwards compatibility as well as optional
   fields such as author information, bug reporting URL, etc.

1. The mandatory metadata must guarantee _binary compatibility_: allow plugins
   built at one version of TensorFlow core to be loaded by a range of TensorFlow
   core versions. If integration tests for API compatibility between version A
   of the plugin and version Z of TensorFlow core pass, as detailed on [modular
   TensorFlow proposal][modular_rfc], then version A of the plugin should be
   allowed to work with version Z of TensorFlow core and mandatory metadata
   should allow this. If integration tests between version B of plugin and
   version Y of core fail then the metadata must forbid this combination.

1. Furthermore, it is ideal to have compilation failures when trying to compile
   plugins against an old API interface which will surely fail the version
   metadata tests. For example, if a method's signature changes this will break
   integration tests but will also be caught during compilation of plugin.
   However, something that might not be caught at compile time is when an
   optional method becomes required or a new method is added. We can maintain
   _source compatibility_ by always increasing version numbers when a method
   changes from being optional to being required and by always adding new
   methods as optionals. But since version changes might be mistakenly
   forgotten, we only guarantee it as best effort, providing users with clear
   runtime error messages when compatibility is accidentally broken.

1. We should minimize the number of times we need to cross module boundaries to
   implement a filesystem functionality. Each time the boundary is crossed we
   have to do conversions from C++ types to C types and back and that incurs
   some performance costs.

1. The API should provide some _type safety_, to ensure that plugin operations
   only act upon data structures that they should work on. Thus, instead of
   using opaque `void*` pointers, we will need to differentiate between the
   structures being operated on. We do this by having data structures which just
   wrap around these opaque pointers. These structures will only have one
   `void*` field.

1. We should provide a way for plugins to register that they don’t support some
   functionality, either because it doesn’t make sense for the specific plugin
   or because of compatibility reasons.

1. We should have a minimal amount of allocation at the boundary between plugins
   and core TensorFlow. Most of the memory should either be owned by TensorFlow
   or by the plugin and the only transfer between these should happen when
   reading/writing files as well as in converting from C++ types to C types
   (e.g. marshalling from `string` to `char*`)

1. The proposed API should minimize the amount of sweeping changes that need to
   happen on TensorFlow code. Ideally, we should allow some period of time when
   both plugins and the existing filesystem implementation will be present in
   the code, allowing users to gradually shift towards the new design.
   Fortunately, the proposed design allows for transparently rewriting the
   current C++ APIs, causing minimal disruption.

1. Although this design is mostly about C++ implementation, we should ensure
   Python code and other language bindings will be able to continue working with
   the modular design.

## Existing implementation

The implementation of a filesystem support in TensorFlow is hairy, due to the
complexity of operations that needs to be provided. This section is an overview
of the architecture.

In short summary, TensorFlow’s support for filesystems can be decomposed into 3
types of API:

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

For a better understanding of this proposal, we will present only the low level
details here as these are the ones that we propose to change. The rest of the
filesystem implementation is moved to [the appendix][filesystem appendix].

### Low level filesystem API

The low-level filesystem API in TensorFlow guarantees the same interface both
across multiple operating systems as well as across multiple filesystems hiding
the complexities arising from the differences between these platforms. Thus,
there are two different things that need consideration: implementing a new
filesystem and making it accessible for each operating system (i.e.
registering).

#### The filesystem implementation

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

#### Cross operating system compatibility

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

The following diagram shows several of the implementations of the `Env`
interface:

![Example environments](20190506-filesystem-plugin-modular-tensorflow/env.png "Example environments")

#### Registering a filesystem to the environment

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

For the rest of the filesystem related implementations that TensorFlow offers,
consult [the appendix][filesystem appendix].

## Proposed implementation

Since the existing implementation is layered, as described above, we can focus
this design on the low level API part. Everything else can be incorporated into
the core of TensorFlow in the modular world and it should support any filesystem
in a transparent way (as long as the plugin is loaded).

### Filesystem interface

In this section, we present the interface that all filesystem plugins must
implement before they can be loaded and used.

Across core-plugin boundary, we can represent open files and a filesystem as
opaque pointers (`void*`) but this results in a loss of type safety. It is easy
to imagine passing a pointer to a read only file to a method expecting to write
to such a file. To prevent this from happening, we propose the following 4 data
structures which just incorporate the `void*` pointer:

```cpp
typedef struct TF_RandomAccessFile {
  void* plugin_file;
} TF_RandomAccessFile;

typedef struct TF_WritableFile {
  void* plugin_file;
} TF_WritableFile;

typedef struct TF_ReadOnlyMemoryRegion {
  void* plugin_memory_region;
} TF_ReadOnlyMemoryRegion;

typedef struct TF_Filesystem {
  void* plugin_filesystem;
} TF_Filesystem;
```

The glue code between the core TensorFlow and the plugin will
allocate/deallocate these structures but they can also be allocated/deallocated
by plugins during testing, for example:

```cpp
class PluginFilesystemIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    filesystem_ = new TF_Filesystem;
    Init(filesystem_);
    status = TF_NewStatus();
  }

  void TearDown() override {
    TF_DeleteStatus(status);
    Cleanup(filesystem_);
    delete filesystem_;
  }

  TF_Filesystem* filesystem_;
  TF_Status* status;
};

TEST_F(PluginFilesystemIOTest, TestIO) {
  // ...
}
```

These structures are useless without operations to operate on them. We could
incorporate function pointer in the structures themselves but that will result
in a lot of duplication and object size changes when one functionality gets
added/removed. This, in turn results in compatibility being harder to maintain,
which is something we don’t want.

Instead, we provide 3 different data structures to act as function tables, to
hold function pointers for files and the filesystem. For this proposal, these
structures are filled with the functionality already present in the existing
filesystem API, but we can expand them as the review progresses:

```cpp
// Operations on a TF_RandomAccessFile
typedef struct TF_RandomAccessFileOps {
  // versioning information elided for now
  // ...
  // API information below
  size_t (*const Read)(const TF_RandomAccessFile*, uint64, size_t, char*, TF_Status*);
  void (*const Cleanup)(TF_RandomAccessFile*);
} TF_RandomAccessFileOps;

// Operations on a TF_WritableFile
typedef struct TF_WritableFileOps {
  // versioning information elided for now
  // ...
  // API information below
  void (*const Append)(const TF_WritableFile*, const char*, size_t, TF_Status*);
  void (*const Close)(TF_WritableFile*, TF_Status*);
  void (*const Cleanup)(TF_WritableFile*);
  // optional
  int64 (*const Tell)(const TF_WritableFile*, TF_Status*);
  void (*const Flush)(const TF_WritableFile*, TF_Status*);
  void (*const Sync)(const TF_WritableFile*, TF_Status*);
} TF_WritableFileOps;

// Operations on a TF_ReadOnlyMemoryRegion
typedef struct TF_ReadOnlyMemoryRegionOps {
  // versioning information elided for now
  // ...
  // API information below
  const void* (*const Data)(const TF_ReadOnlyMemoryRegion*);
  uint64 (*const Length)(const TF_ReadOnlyMemoryRegion*);
  void (*const Cleanup)(TF_ReadOnlyMemoryRegion*);
} TF_ReadOnlyMemoryRegionOps;


// Operations on a TF_Filesystem
typedef struct TF_FilesystemOps {
  // versioning information elided for now
  // ...
  // API information below
  void (*const NewRandomAccessFile)(const TF_Filesystem*, const char*, TF_RandomAccessFile*, TF_Status*);
  void (*const NewWritableFile)(const TF_Filesystem*, const char*, TF_WritableFile*, TF_Status*);
  void (*const NewAppendableFile)(const TF_Filesystem*, const char*, TF_WritableFile*, TF_Status*);
  void (*const NewReadOnlyMemoryRegionFromFile)(const TF_Filesystem*, const char*, TF_ReadOnlyMemoryRegion*, TF_Status*);

  void (*const CreateDir)(const TF_Filesystem*, const char*, TF_Status*);
  void (*const RecursivelyCreateDir)(const TF_Filesystem*, const char*, TF_Status*);

  void (*const DeleteFile)(const TF_Filesystem*, const char*, TF_Status*);
  void (*const DeleteDir)(const TF_Filesystem*, const char*, TF_Status*);
  void (*const DeleteRecursively)(const TF_Filesystem*, const char*, int64*, int64*, TF_Status*);

  void (*const RenameFile)(const TF_Filesystem*, const char*, const char*, TF_Status*);
  void (*const CopyFile)(const TF_Filesystem*, const char*, const char*, TF_Status*);

  void (*const FileExists)(const TF_Filesystem*, const char*, TF_Status*);
  bool (*const FilesExist)(const TF_Filesystem*, const char**, int, TF_Status**);
  int (*const GetChildren)(const TF_Filesystem*, const char*, char***, TF_Status*);

  void (*const Stat)(const TF_Filesystem*, const char*, TF_FileStatistics*, TF_Status*);
  void (*const IsDirectory)(const TF_Filesystem*, const char*, TF_Status*);
  uint64 (*const GetFileSize)(const TF_Filesystem*, const char*, TF_Status*);
  int (*const GetMatchingPaths)(const TF_Filesystem*, const char*, char***, TF_Status*);

  void (*const FlushCaches)(const TF_Filesystem*);
  const char* (*const TranslateName)(const TF_Filesystem*, const char*);

  void (*const Init)(TF_Filesystem*);
  void (*const Cleanup)(TF_Filesystem*);
} TF_FilesystemOps;
```

We also provide some convenience calls that plugins can use to fill in these
function tables, for example:

```cpp
void FlushCaches_NoOp(const TF_Filesystem* fs);
const char* TranslateName_Identity(const TF_Filesystem* fs, const char* name);
```

As another benefit, having separate function tables can be checked at plugin
registration to ensure that all required functionality is present. This makes it
easier to guarantee compatibility between versions. This will be the focus of
the next section.

### Plugin initialization and registration

When the shared object containing the plugin is loaded the `TF_InitPlugin`
symbol is called to register the plugin:

```cpp
TF_CAPI_EXPORT extern void TF_InitPlugin(TF_Status* status);
```

This function must allocate all of the above tables and then call a function
provided by the core TensorFlow to register plugin filesystems:

```cpp
TF_CAPI_EXPORT extern void RegisterFileSystemPlugin(
    const string& plugin, const string& scheme,
    const TF_Filesystem_Plugin_Metadata* metadata,
    const TF_FilesystemOps* filesystemOps,
    const TF_RandomAccessFileOps* randomAccessFileOps,
    const TF_WritableFileOps* writableFileOps,
    const TF_ReadOnlyMemoryRegionOps* readOnlyMemoryRegionOps,
    TF_Status* status);
```

The `RegisterFileSystemPlugin` call is the one place where we can check that all
required functionality is implemented, that the provided version number is
compatible with the current version of TensorFlow core, etc. Once everything is
validated, we register the new filesystem and it becomes available to the
runtime:

```cpp
void RegisterFileSystemPlugin(
    const string& plugin, const string& scheme,
    const TF_Filesystem_Plugin_Metadata* metadata,
    const TF_FilesystemOps* filesystemOps,
    const TF_RandomAccessFileOps* randomAccessFileOps,
    const TF_WritableFileOps* writableFileOps,
    const TF_ReadOnlyMemoryRegionOps* readOnlyMemoryRegionOps,
    TF_Status* status) {
  if (!ValidateMetadata(metadata, status)) return;
  if (!ValidateFilesystemOps(filesystemOps, status)) return;
  if (!ValidateRandomAccessFileOps(randomAccessFileOps, status)) return;
  if (!ValidateWritableFileOps(writableFileOps, status)) return;
  if (!ValidateReadOnlyMemoryOps(readOnlyMemoryRegionOps, status)) return;

  // ... rest of initialization and registration
}
```

### Incorporating C API into filesystem classes

To incorporate this new filesystem mechanism, we will provide new instances for
`RandomAccessFile`, `WritableFile` and `ReadOnlyMemoryRegion` classes: they need
to allocate the corresponding `TF_*` object (e.g., `TF_RandomAccessFile` for
`RandomAccessFile`). Furthermore, each of these classes will incorporate the
corresponding function table (e.g., `TF_RandomAccessFileOps` for
`RandomAccessFile`). Then, methods of these classes will just load the
corresponding function from the table and call it, marshalling arguments and
return values. For example, here is the implementation of
`ModularRandomAccessFile`:

```cpp
class ModularRandomAccessFile : public RandomAccessFile {
 public:
  ModularRandomAccessFile(const string& filename,
                          const TF_RandomAccessFile* file,
                          const TF_RandomAccessFileOps* ops)
      : filename_(filename), file_(file), ops_(ops) {}
  virtual ~ModularRandomAccessFile() {
    ops_->Cleanup(const_cast<TF_RandomAccessFile*>(file_));
    delete file_;
  }

  Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const;
  Status Name(StringPiece* result) const;

 private:
  string filename_;
  const TF_RandomAccessFile* file_;
  const TF_RandomAccessFileOps* ops_;
  TF_DISALLOW_COPY_AND_ASSIGN(ModularRandomAccessFile);
};

Status ModularRandomAccessFile::Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const {
  TF_Status* plugin_status = TF_NewStatus();
  size_t read = ops_->Read(file_, offset, n, scratch, plugin_status);
  *result = StringPiece(scratch, read);
  Status status = StatusFromTF_Status(plugin_status);
  TF_DeleteStatus(plugin_status);
  return status;
}

Status ModularRandomAccessFile::Name(StringPiece* result) const {
  *result = filename_;
  return Status::OK();
}
```

Similarly, we will have to implement a subclass of the `FileSystem` class,
`ModularFileSystem`, in the same vein.

Another change we have to do is to `FileSystemRegistry`: the `Register()` call
after this proposal will use a `unique_ptr<FileSystem>` argument instead of the
existing `Factory`-based implementation since the constructor of
`ModularFileSystem` needs to take as argument the function tables for the
operations supported by the filesystem.

Because of this change, we also need a new subclass of `Env`, as each instance
of `Env` has the `FileSystemRegistry` baked in at construction time. In fact, we
can just change the existing class to have two lookup layers for the two
registries.

Furthermore, since most of the functionality calls `Env::Default()` to obtain
the static instance of `Env` and then extract the filesystem from there, we can
get rid of the level of indirection via `Env` and create a static instance of
`ModularFileSystem`. This will be possible only after all filesystems use the
modular approach, after the changes mentioned in the next section.

Since the function tables in the filesystem interface mention this, we will have
to move `TF_FileStatistics` from `tensorflow/c/env.h` to the same location as
the filesystem plugin interface. Then, the outer level C API can just include
this header and the structure will still be visible there.

Once these changes are in place, the rest of the TensorFlow code can use the
modular filesystem API transparently, with no change.

### Optional additional changes to TensorFlow C/C++ core

Something we might elect to do is to force all future filesystem implementations
to use the plugin design suggested above. To do that, we will have to convert
all code that uses additional APIs to use this proposed API. Furthermore, we
might want to mark all filesystem classes as `final`, deprecating and removing
the current subclasses that are present in the source code, after converting
them to the modular world. This process should happen in two steps: first we
have to add deprecation warnings to the classes (printed at compile time) and
release a TensorFlow release. After that release we can convert the classes to
`final` and remove subclasses. This step will also allow us to avoid
indirections via `Env` class.

Optionally, to not have to always translate from `TF_FileStatistics` to
`FileStatistics`, we might implement a mechanism to convert from one to the
other within as few CPU cycles as possible.

### Python considerations

Since these changes only happen at the very low level of the C++ TensorFlow
core, Python and other language bindings should continue to behave the same way
without any changes to the code.

A user having a custom filesystem implementation can use it from Python by
calling [`load_library`][load_library]. When the shared library is loaded, the
plugin initialization starts, as mentioned before. Once the filesystem
implementation is available to TensorFlow core, it can also be accessed from
Python.

## Versioning

Since the code of a plugin can develop at a different velocity than the code of
core TensorFlow, we need to ensure backwards and forwards compatibility. Any
change in a method’s signature or reordering the methods in the `TF_*Ops` data
structures needs a new version as plugins compiled against one signature will
either crash or not load properly if core has a different interface.

Thus, the `TF_*Ops` data structures need to record version information which is
checked at plugin loading time. For example, this is a potential implementation
of `TF_RandomAccessFileOps`, including versioning fields:

```cpp
typedef struct TF_RandomAccessFileOps {
  // these fields must always be at the beginning
  const int version = 1;
  // alternatively: const int size = sizeof(TF_RandomAccessFileOps);
  const int num_ops = 2;

  // the real ops in the table
  size_t (*const Read)(const TF_RandomAccessFile*, uint64, size_t, char*, TF_Status*);
  void (*const Cleanup)(TF_RandomAccessFile*);
} TF_RandomAccessFileOps;
```

We will allow each structure to evolve independently. That is, the `version`
field in each one can be increased independently of the value of the field in
the other structures.

To increase the range of compatibility, we will give a guarantee that, except on
new major versions, new methods will only be added to the end of each of these
structures and that no method will be deleted from the table. Changing the
signature of a method will be handled by adding the method with the new
signature to the end of the table and marking the old method as deprecated as
well as ensuring that the new method will call the old one where needed.

Deprecation of a method will be done in two parts. First, we will add compiler
pragmas to warn the users compiling against the interface and relying on
deprecated methods. Second, at plugin’s registration, TensorFlow code can check
the function pointers sent by the plugin to see which ones are for deprecated
methods. In case there is any, a warning message should be printed.

To allow both the core and the plugin to know which methods from the tables are
supported, we also add the `num_ops` field, counting how many methods are
defined in the table. This field can be used to determine the amount of
available API calls, as it is guaranteed to only increase during minor releases.
We provide a `NUM_OPS_IN_TABLE` macro to allow the compiler to set the value of
`num_ops` automatically.

Alternatively, instead of `num_ops` field we can have a size field which is the
size of the entire structure and can be obtained via `sizeof`. The benefits of
using the `size` field is that we don’t need an additional macro but this comes
at the downside that it will be harder to determine how many methods are
available in the function table.

Major versions of TensorFlow core allow breaking changes so we can completely
change the `TF_*Ops` data structures to remove fields which are no longer used.
We will still increase the `version` field but `num_ops` can be allowed to
decrease.

Similar changes are needed for the metadata information of a plugin, for
example:

```cpp
typedef struct TF_Filesystem_Plugin_Metadata {
  const int version = 1;
  const int num_fields = 2;

  const char* plugin_version_string;
  const char* author;
} TF_Filesystem_Plugin_Metadata;
```

During registration we can check all of the fields and store some for later use
(e.g., for displaying better error message, localizing the plugin in error).


## Questions and comments raised during review

1. **Should we make the function tables public or hide them and only offer
   allocate/free/set/get methods?** The proposed API above uses public
   structures for simplicity but both can be implemented. A benefit of using
   private structures is that we get some extra memory safety guarantees as
   plugin code won't be able to directly access fields of the structure and
   won't be required to keep track of structure size during
   allocation/deallocation. The downside of having private structures is that
   there is slightly more code needed at plugin initialization both on the
   plugin side (calling all the setters instead of just filling in a structure
   using initializer lists) as well as on core side (ensuring that once a
   pointer has been set it won't be set again). Having structures public serves
   as a quick documentation of the API that needs to be provided, documentation
   which is much clearer than reading through several setters of function
   pointers. After analyzing the cost/benefits, we're leaning towards
   **private**.

1. **Should we use the size of the structures or the number of methods in the
   structures as versioning information?** If the structures are private, we can
   do either. If the structures are public storing the size helps in memory
   safety. Leaning towards **size**.

1. **Should we enforce file ops to also take a pointer to the filesystem?** (for
   example `Read` in `TF_RandomAccessFileOps` should take both a pointer to
   `TF_RandomAccessFile` and a pointer to `TF_Filesystem` or only the first
   pointer? Having a direct pointer to `TF_Filesystem` allows storing the state
   of the filesystem but we argue that this is not needed for all filesystem
   implementations and in the cases where is needed the plugin can store a
   pointer to the filesystem inside of the memory pointed to by the
   `TF_RandomAccessFile` pointer. Leaning towards **No**.

1. **Should we pass plugin structs by value during registration?** Doing so
   makes ownership clear and prevents memory allocation bugs. Passing them by
   value results in a copy but we need a copy for security reasons anyway. We're
   leaning towards **Yes**.


[modular_rfc]: https://github.com/tensorflow/community/pull/77 "RFC: Modular TensorFlow"
[filesystem appendix]: 20190506-filesystem-plugin-modular-tensorflow/existing_filesystem_review.md "A detailed presentation of TensorFlow's filesystem support"

[class FileSystem]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L46 "class FileSystem"
[class RandomAccessFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L232 "class RandomAccessFile"
[class WritableFile]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L271 "class WritableFile"
[class ReadOnlyMemoryRegion]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L342 "class ReadOnlyMemoryRegion"
[class Env]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.h#L48 "class Env"
[class FileSystemRegistry]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_system.h#L360 "class FileSystemRegistry"
[class FileStatistics]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/file_statistics.h#L23 "class FileStatistics"
[struct TF_FileStatistics]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/c/env.h#L36 "struct TF_FileStatistics"

[HadoopFileSystem]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/hadoop/hadoop_file_system.cc#L368-L377 "HadoopFileSystem::NewReadOnlyMemoryRegionFromFile()"
[FileSystemRegistry init]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.cc#L93 "initialization of FileSystemRegistry"

[REGISTER_FILE_SYSTEM]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/core/platform/env.h#L480 "REGISTER_FILE_SYSTEM(scheme, factory)"

[load_library]: https://github.com/tensorflow/tensorflow/blob/69bd23af10506b0adae9b9795a00d4dc05b8a7fd/tensorflow/python/framework/load_library.py#L132
