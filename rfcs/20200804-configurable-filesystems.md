# Configurable File Systems

| Status        | Proposed                                                                                      |
| :------------ | :-------------------------------------------------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #) |
| **Author(s)** | Sami Kama (kamsami@amazon.com)                                                                |
| **Sponsor**   | Mihai Maruseac (mihaimaruseac@google.com)                                                     |
| **Updated**   | 2020-08-04                                                                                    |

## Objective

The aim of this RFC to extend filesystem API to enable users to pass configuration parameters to tune the behavior of implementation to their use cases.

## Motivation

There are many FileSystem implementations in Tensorflow that enable interaction with various storage solutions. Most of these implementations have internal parameters that are suitable for generic use case but not necessarily optimal for all cases. For example accessing remote filesystems through multiple threads can improve the throughput if there is a high bandwidth connection to the remote thus increasing number of connections might be beneficial. On the other hand if the connection is slow, a higher number of threads will just waste resources and may even reduce the throughput. Depending on the resources available during the execution, users should be able to alter some of the parameters of the Filesystems to improve the performance of their execution. This can be especially useful for the the cases where the execution is data i/o bound.

## User Benefit

With this proposal users will be able to fine tune some parameters that developers expose through configuration API and get an improved perfomance for file i/o.

## Design Proposal

This proposal introduces two new methods to plugin api structure `TF_FilesystemOps` as shown below.

```cpp
struct TF_FilesystemOps{
    // other members are ignored for brevity
    void (*const get_filesystem_configuration)(char** serialized_config, int *serialized_length, TF_Status* status);
    void (*const set_filesystem_configuration)(const char* serialized_config, int serialized_length, TF_Status* status);
}
```

where `serialized_config` is a pointer to the buffer containing serialized human readable form of the protobuf object described below and `serialized_length` is the length of the buffer.

For non-plugin based filesystems, FileSystem API can be extended similarly.

```cpp
class FileSystem{
    public:
    // existing methods are not shown.
    Status GetConfiguration(std::unique_ptr<FilesystemConfig>* config);
    Status SetConfiguration(std::unique_ptr<FilesystemConfig> new_configuration);
}
```

Since each filesystem will likely to have different set of tunable parameters, a `FilesystemConfig` object can be used to unify the API and allow discovery of existing tunable parameters at runtime. We propose a protobuf object with the following schema

``` proto
message FilesystemAttr{
  message ListValue {
    repeated bytes s = 2;                        // "list(string)"
    repeated int64 i = 3 [packed = true];        // "list(int)"
    repeated float f = 4 [packed = true];        // "list(float)"
    repeated bool b = 5 [packed = true];         // "list(bool)"
  }
  oneof value {
    bytes s = 2;                 // "string"
    int64 i = 3;                 // "int"
    float f = 4;                 // "float"
    bool b = 5;                  // "bool"
    ListValue list = 1;          // any "list(...)"
  }
  optional string  description = 2;
}

message FilesystemConfig{
  string owner = 1;
  string version = 2;
  map<string, FilesystemAttr> options = 3 ;
}
```

It is possible to choose `FilesystemConfig` to be another human readable key-value store format with a similar structure such as `json` or `yaml`, though this may limit the data types that can be used for configuration.
Filesystems which doesn't have user configurable parameters can leave these methods unimplemented. In that case default implementations will return `nullptr` wherever applicable. `FilesystemConfig` object can be exposed to python layer for modifications at python level.

Typical use pattern would be that user queries the Filesystem implementation for current configuration. Filesystem returns an object populated with all configurable parameters and their existing or default values which also serves as a schema. User creates a copy of the configuration, modified desired parameters in protobuf object and passes this back to Filesystem through `SetConfiguration()` call. Then Filesystem alter its operational parameters if modifications are within acceptable limits or return an error with apropriate message describing the issue.

### Alternatives Considered

Alternative to this proposal is to use a side-channel such as an environment variable to modify the internal parameters. However this is cumbersome, error prone and may not be possible to use at all under certain circumstances.

### Performance Implications

This proposal should help improve persistent storage i/o performance.

### Dependencies

This proposal do not introduce any new dependencies though, plugin based filesystems may have to link against protobuf (and hide its symbols) or respective library if an alternative form for `FilesystemConfig` is chosen.

### Engineering Impact

Engineering impact of this change is negligable. Amount of change needed is proportional to configurability that developers choose to expose to user.

### Platforms and Environments

This proposal is applicable to all Filesystems on all supported platforms.

### Best Practices

This proposal provides tuning handles to users for tuning the i/o performance. These can be documented in performance guides, in filesystem implementations or the `FilesystemAttr.description` field of the configuration object.

### Tutorials and Examples

An example use of the new API could be as follows.

```cpp
Status SetFilesystemThreads(int thread_count) {
  ModularFileSystem* fs = Env::Default()->GetFileSystemForFile(
      "remote://some_configurable_remote_filesystem");
  std::unique_ptr<FilesystemConfig> config;
  auto s = fs->GetConfiguration(&config);
  if (!s.ok()) return s;
  if (!config) return Status::OK();  // No configuration support
  std::unique_ptr<FilesystemConfig> new_config =
      std::make_unique<FilesystemConfig>() new_config->CopyFrom(*config);
  if (config->options.contains("ThreadPoolSize")) {
    new_config->options.at("ThreadPoolSize").set_i(8);
  }
  fs->SetConfiguration(std::move(new_config));
  return Status::OK();
}
```

### Compatibility

This proposal have no effect on compatibility of existing code.

### User Impact

This proposal will expose new methods to user to query and modify operational parameters of Filesystems. Users wishing to tune their Filesystem access will be able to do so.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
