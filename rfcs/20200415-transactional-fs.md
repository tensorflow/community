# Transactional File Systems Support

| Status        | (Proposed / Accepted / Implemented / Obsolete)                                                |
| :------------ | :-------------------------------------------------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #) |
| **Author(s)** | kamsami@amazon.com                                                                            |
| **Sponsor**   | A N Expert (whomever@tensorflow.org)                                                          |
| **Updated**   | 2020-04-15                                                                                    |

## Objective

The aim of this RFC to extend filesystem access support to persistent storage that
provide transactional access and eventual consistency.

## Motivation

Current persistent storage implementation in Tensorflow relies on certain guarantees that
existing local file systems provide. But in the age of big data, local filesystems are not always
sufficient and/or prefetching the data to local system and then uploading after processing can be error-prone
and harder to implement for end users. Direct access to persistent storage that provides different gurantees is desirable.
Cloud storage solutions offered by Amazon, Google and others and databases can be examples of such persistent storages.
Moreover even though local file systems provide certain atomic-like transactions, they are on file level.
For the use cases like checkpointing, transactions are emulated through creating a temp directory, adding files to there and then
renaming/moving temp directory to final location. Such operations would also benefit from enhancements proposed in this RFC.

## User Benefit

With this extension proposal, users will have a more stable access to cloud storage systems as well as checkpointing redundancy can improve.

## Design Proposal

This RFC proposes to extend the [filesystem plugin rfc][filesystem_plugin] api with transaction markers. There can be different levels of transactions. First level can be global transactions, user starts a transaction scope. Any operations done within this scope will be grouped in this transaction. This is easiest to implement but have drawbacks such as only one transaction can be present at a time and different type of transactions may not always be reordered.
Alternative to this approach is having multiple transaction scopes. User can create a transaction token and pass it to filesystem operations for plugin to differentiate among independent transactions. This token can be per file or per directory level granularity. Even though per file would give most flexibility, per directory level of transaction detail is most likely sufficient.

Filesystem plugins may choose to ignore the transaction scopes or can delay the operations until the termination of transaction scope.

### Extension to existing filesystem implementation

Existing filesystem C++ api can easily be expanded by addition of a two methods, an opaque structure and possibly a helper class.

```cpp
struct TransactionToken{
  // use int for simplicity. Can be changed to void*
  // to make it completely opaque for more advanced use case.
  void* id;
};

template <typename T>
class TokenScope{
  public:
  // Transaction name can be filename or directory name
    TokenScope(T* fs_,const string& transaction_name):fs(fs_){
      auto status=fs->StartTransaction(transaction_name,&token);
    }
    ~TokenScope(){
      fs->EndTransaction(token);
    }
    TokenScope(const TokenScope&) = delete;
    const std::unique_ptr<TransactionToken>* GetToken() const {return &token;}
  private:
    std::unique_ptr<TransactionToken> token;
    T* fs;
};
```

Then signature of each method needs to be expanded with a unique pointer to this `TransactionToken` structure argument which defaults to `nullptr`
for minimizing the impact on the existing code and allow incremental migration to implementation of transactions.

```cpp
class Filesystem {
  // Transaction Tokens
  virtual Status GetTransactionTokenForFile(const string& file_name,std::unique_ptr<TransactionToken>* token) = 0
  virtual Status StartTransaction(const string& transaction_name, std::unique_ptr<TransactionToken>* token) = 0;
  virtual Status EndTransaction(std::unique_ptr<TransactionToken>* token) = 0;

  // File creation
  virtual Status NewRandomAccessFile(const string& fname, std::unique_ptr<RandomAccessFile>* result, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status NewWritableFile(const string& fname, std::unique_ptr<WritableFile>* result, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status NewAppendableFile(const string& fname, std::unique_ptr<WritableFile>* result, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status NewReadOnlyMemoryRegionFromFile(const string& fname, std::unique_ptr<ReadOnlyMemoryRegionFile>* result, std::unique_ptr<TransactionToken>* token=nullptr) = 0;

  // Creating directories
  virtual Status CreateDir(const string& dirname, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status RecursivelyCreateDir(const string& dirname), std::unique_ptr<TransactionToken>* token=nullptr;

  // Deleting
  virtual Status DeleteFile(const string& fname, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status DeleteDir(const string& dirname, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status DeleteRecursively(const string& dirname, int64* undeleted_files, int64* undeleted_dirs, std::unique_ptr<TransactionToken>* token=nullptr);

  // Changing directory contents
  virtual Status RenameFile(const string& src, const string& target, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status CopyFile(const string& src, const string& target, std::unique_ptr<TransactionToken>* token=nullptr);

  // Filesystem information
  virtual Status FileExists(const string& fname, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual bool FilesExist(const std::vector<string>& files, std::vector<Status>* status,std::unique_ptr<TransactionToken>* token=nullptr);
  virtual Status GetChildren(const string& dir, std::vector<string>* result, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status Stat(const string& fname, FileStatistics* stat, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status IsDirectory(const string& fname, std::unique_ptr<TransactionToken>* token=nullptr);
  virtual Status GetFileSize(const string& fname, uint64* file_size, std::unique_ptr<TransactionToken>* token=nullptr) = 0;

  // Globbing
  virtual Status GetMatchingPaths(const string& pattern, std::vector<string>* results, std::unique_ptr<TransactionToken>* token=nullptr) = 0;

  // Misc
  virtual void FlushCaches();
  virtual string TranslateName(const string& name) const;
};
```

File classes can be modified to keep TransactionToken, assigned by the filesystem on their construction using given scope, or default scope if not given. Filesystems may ignore it if transaction at that level doesn't make sense.

```cpp
class RandomAccessFile {
  virtual Status Name(StringPiece* result) const;
  virtual Status Read(uint64 offset, size_t n, StringPiece* result, char* scratch) const = 0;
  private:
  TransactionToken token;
};

class WritableFile {
  virtual Status Name(StringPiece* result) const;
  virtual Status Append(StringPiece data) = 0;
  virtual Status Append(const absl::Cord& cord);
  virtual Status Tell(int64* position);
  virtual Status Close() = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;
  private:
  TransactionToken token;
};

class ReadOnlyMemoryRegion {
  virtual const void* data() = 0;
  virtual uint64 length() = 0;
  private:
  TransactionToken token;
};
```

Then respective `Env` class methods needs to receive transaction tokens to relay on the file system. Arguments are defaulted to nullptr, indicating use of default transaction.
Transaction tokens should be taken from respective filesystems. Alternatively, they can be constructed with an `UNINITIALIZED` token and then respective filesystem can populate it.

```cpp
class Env {
  // Filesystem registration
  virtual Status GetFileSystemForFile(const string& fname, FileSystem** result);
  virtual Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes);
  virtual Status RegisterFileSystem(const string& scheme, FileSystemRegistry::Factory factory);

  // Transaction Token related
  virtual Status GetTransactionTokenForFile(const string& file_name,std::unique_ptr<TransactionToken>* token) = 0
  virtual Status StartTransaction(const string& transaction_name, std::unique_ptr<TransactionToken>* token) = 0;
  virtual Status EndTransaction(std::unique_ptr<TransactionToken>* token) = 0;

  // Creating files, including memory mapped
  Status NewRandomAccessFile(const string& fname, std::unique_ptr<RandomAccessFile>* result, std::unique_ptr<TransactionToken>* token=nullptr);
  Status NewWritableFile(const string& fname, std::unique_ptr<WritableFile>* result, std::unique_ptr<TransactionToken>* token=nullptr);
  Status NewAppendableFile(const string& fname, std::unique_ptr<WritableFile>* result, std::unique_ptr<TransactionToken>* token=nullptr);
  Status NewReadOnlyMemoryRegionFromFile(const string& fname, std::unique_ptr<ReadOnlyMemoryRegionFile>* result, std::unique_ptr<TransactionToken>* token=nullptr);

  // Creating directories
  Status CreateDir(const string& dirname, std::unique_ptr<TransactionToken>* token=nullptr);
  Status RecursivelyCreateDir(const string& dirname, std::unique_ptr<TransactionToken>* token=nullptr);

  // Deleting
  Status DeleteFile(const string& fname, std::unique_ptr<TransactionToken>* token=nullptr);
  Status DeleteDir(const string& dirname, std::unique_ptr<TransactionToken>* token=nullptr);
  Status DeleteRecursively(const string& dirname, int64* undeleted_files, int64* undeleted_dirs,std::unique_ptr<TransactionToken>* token=nullptr);

  // Changing directory contents
  Status RenameFile(const string& src, const string& target, std::unique_ptr<TransactionToken>* token=nullptr);
  Status CopyFile(const string& src, const string& target, std::unique_ptr<TransactionToken>* token=nullptr);

  // Filesystem information
  Status FileExists(const string& fname);
  bool FilesExist(const std::vector<string>& files, std::vector<Status>* status);
  Status GetChildren(const string& dir, std::vector<string>* result, std::unique_ptr<TransactionToken>* token=nullptr);
  Status Stat(const string& fname, FileStatistics* stat, std::unique_ptr<TransactionToken>* token=nullptr);
  Status IsDirectory(const string& fname, std::unique_ptr<TransactionToken>* token=nullptr);
  Status GetFileSize(const string& fname, uint64* file_size, std::unique_ptr<TransactionToken>* token=nullptr);

  // Globbing
  virtual bool MatchPath(const string& path, const string& pattern, std::unique_ptr<TransactionToken>* token=nullptr) = 0;
  virtual Status GetMatchingPaths(const string& pattern, std::vector<string>* results, std::unique_ptr<TransactionToken>* token=nullptr);

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

For the new proposed filesystem plugin mechanism, two possible approaches exists. For `TF_RandomAccessFile`, `TF_WritableFile`, and `TF_ReadOnlyMemoryRegion` structures,

- Opaque pointers stay as is, thus no changes is needed in structures. Then each filesystem attach tokens to their own internal structures pointed by `void*`.
- Structures are extended to keep a pointer `TransactionToken` structure.

Second method is more explicit but constrains all filesystems to use same token type, which is most likely not useful for any filesystem other than the one created it. Thus first solution may allow for more complicated data structures and flexibility to filesystems. Similar to `Env` class, FilesystemOps signatures need to be `TransactionToken` pointers

``` cpp
// Operations on a TF_Filesystem
typedef struct TF_FilesystemOps {
  // versioning information elided for now
  // ...
  // API information below
  void (*const NewRandomAccessFile)(const TF_Filesystem*, const char*, TF_RandomAccessFile*, TF_Status*, TransactionToken*);
  void (*const NewWritableFile)(const TF_Filesystem*, const char*, TF_WritableFile*, TF_Status*, TransactionToken*);
  void (*const NewAppendableFile)(const TF_Filesystem*, const char*, TF_WritableFile*, TF_Status*, TransactionToken*);
  void (*const NewReadOnlyMemoryRegionFromFile)(const TF_Filesystem*, const char*, TF_ReadOnlyMemoryRegion*, TF_Status*, TransactionToken*);

  void (*const CreateDir)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  void (*const RecursivelyCreateDir)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);

  void (*const DeleteFile)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  void (*const DeleteDir)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  void (*const DeleteRecursively)(const TF_Filesystem*, const char*, int64*, int64*, TF_Status*, TransactionToken*);

  void (*const RenameFile)(const TF_Filesystem*, const char*, const char*, TF_Status*, TransactionToken*);
  void (*const CopyFile)(const TF_Filesystem*, const char*, const char*, TF_Status*, TransactionToken*);

  void (*const FileExists)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  bool (*const FilesExist)(const TF_Filesystem*, const char**, int, TF_Status**, TransactionToken*);
  int (*const GetChildren)(const TF_Filesystem*, const char*, char***, TF_Status*, TransactionToken*);

  void (*const Stat)(const TF_Filesystem*, const char*, TF_FileStatistics*, TF_Status*, TransactionToken*);
  void (*const IsDirectory)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  uint64 (*const GetFileSize)(const TF_Filesystem*, const char*, TF_Status*, TransactionToken*);
  int (*const GetMatchingPaths)(const TF_Filesystem*, const char*, char***, TF_Status*, TransactionToken*);

  void (*const FlushCaches)(const TF_Filesystem*);
  const char* (*const TranslateName)(const TF_Filesystem*, const char*);
  // Transaction management
  void (*const StartTransaction)(const TF_Filesystem*, TransactionToken*);
  void (*const EndTransaction)(const TF_Filesystem*, TransactionToken*);
  // misc
  void (*const Init)(TF_Filesystem*);
  void (*const Cleanup)(TF_Filesystem*);
} TF_FilesystemOps;
```

TODO:more

Make sure you’ve thought through and addressed the following sections. If a section is not relevant to your specific proposal, please explain why, e.g. your RFC addresses a convention or process, not an API.

### Alternatives Considered

- TODO

### Performance Implications

- This will allow filesystem plugin implementations to optimize access to non-local file systems and likely improve performance.

### Dependencies

- This proposal do not require any additional dependencies, but may lead to implementation of more persistent storage access plugins.

### Engineering Impact

- There exists quite complex workarounds for shortcomings of filesystems that can be hidden behind filesystem with proposals in this RFC. Changing this workarounds may require some implementation changes.

<!--
- The expected engineering impact is minimal. Required changes involve grouping filesystem i/o operations in to transaction groups that will likely be no-ops for traditional file systems.
-->

### Platforms and Environments

- Proposed changes are platform independent and should not affect code generation or execution environments.

# TODO

### Best Practices

- Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples

- If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
  - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
  - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
  - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer.
  - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged.

### Compatibility

- Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
- How will this proposal interact with other parts of the TensorFlow Ecosystem?
  - How will it work with TFLite?
  - How will it work with distribution strategies?
  - How will it interact with tf.function?
  - Will this work on GPU/TPU?
  - How will it serialize to a SavedModel?

### User Impact

- What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.

[filesystem_plugin]: https://github.com/tensorflow/community/blob/master/rfcs/20190506-filesystem-plugin-modular-tensorflow.md
