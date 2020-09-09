# StreamExecutor C API Versioning Strategy
| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | Extension of #[257](https://github.com/tensorflow/community/pull/257) |
| **Authors** | Yi Situ (yisitu@google.com), Penporn Koanantakool (penporn@google.com), Anna Revinskaya (annarev@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                        |
| **Updated**   | 2020-09-08                                              |

In reply to a question on [PR #262](https://github.com/tensorflow/community/pull/262#issuecomment-653690654).

StreamExecutor C API (SE C API) follows Semantic Versioning 2.0.0
([semver](http://semver.org/)). Each release version has a format
`MAJOR.MINOR.PATCH`, as outlined in [TensorFlow version compatibility](https://www.tensorflow.org/guide/versions#semantic_versioning_20).
We also use `struct_size` to track compatibility.

## Updating Guidelines
This section outlines when to update version numbers specific to SE C API
(`SE_MAJOR`, `SE_MINOR`, and `SE_PATCH`).

### SE_MAJOR
* Potentially backwards incompatible changes.
* If a change is backwards incompatible, it requires an RFC because it will
  break all current plug-ins. This should be rare.
* An `SE_MAJOR` update should be planned in a way that bundles as many pending
  backwards incompatible changes together as possible to avoid breaking plug-ins
  multiple times.
* There will be an announcement giving a grace period before the update happens.

### SE_MINOR
* Backwards compatible changes.
  * Adding a new variable, struct, method, enumeration, etc.
  * Trivial deprecation of a variable, etc. by setting it to a no-op values,
    e.g., 0 or `NULL`.

### SE_PATCH
* Backwards compatible bug fixes.

## Conventions
* Once a member is added to a struct, it cannot be removed, reordered, renamed,
  or repurposed (i.e., assigned a different functionality).
* "Renaming" a member is equivalent to adding a new member with a new name and
  eventually deprecating the member with the old name.
* Fields that cannot be 0 or `NULL` can be deprecated in a backwards compatible
  manner by zero-initialization. 
  * Plug-ins must perform input validation on these fields for 0 and `NULL`
    before consuming them.
      * Plug-ins know the fields are deprecated when they find 0 or `NULL` in
        these fields.
  * Such fields must be explicitly marked by comments, to ensure all plug-ins
    have consistent behavior (e.g., non of the plug-ins is using 0 or `NULL` as
    a special case). See `// 0 is no-op` and `// NULL is no-op` in the 
    [By value inspection](#By-value-inspection) section for example.


## Detecting Incompatibility

### By Comparing SE_MAJOR at Registration Time
At load time, both plug-in and core TensorFlow should check for version
compatibility. If the versions are not compatible, plug-in should output an
error and core TensorFlow should unload the plug-in. See code example below.

Core TensorFlow passes its SE C API version number when calling plug-in's
initialization routine (`SE_InitPlugin`):
```c++
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams*, TF_Status*);
SE_PlatformRegistrationParams params{SE_PLATFORM_REGISTRATION_PARAMS_SIZE};
params.major_version = SE_MAJOR;
params.minor_version = SE_MINOR;
params.patch_version = SE_PATCH;
TF_Status status;

// Core TensorFlow sends its view of version numbers to plugin.
void* initialize_sym = dlsym(plugin, "SE_InitPlugin");
if (!initialize_sym) {
  // Output error and skip this plug-in.
}
SEInitPluginFn initialize_plugin_fn = reinterpret_cast<SEInitPluginFn>(initialize_sym);
initialize_plugin_fn(&params, &status);
if(!tensorflow::StatusFromTF_Status(status).ok()) {
  // Output error and skip this plug-in.
}
```

Plug-in checks the `SE_MAJOR` version numbers and outputs error if they don't
match:
```c++
void SE_InitPlugin(SE_PlatformRegistrationParams* params,
                         TF_Status* status) {
  if (params->struct_size == 0) {
    // *status = ...
    LOG(ERROR) << "Invalid argument.";
    return;
  }
  if (SE_MAJOR != params->major) {
    // *status = ...
    LOG(ERROR) << "Unsupported major version. Given: " << params->major
               << " Expected: " << SE_MAJOR;
    return;
  }
  ...
}
```

### By Value Inspection
Deprecation of an attribute can sometimes be done in a backwards compatible
manner by leaving the attribute zero initialized.

* The plugin performs input validation on each field for `NULL` or 0 value
  before consuming it, preventing it from entering a bad state.
* If deprecation by zero-initialization is not possible (e.g., because default
  value of zero may be a valid input), then the change is API incompatible;
  TensorFlow has to bump the major version when the attribute is deprecated.

For example,

```c++
struct example {
  int cannot_be_zero;  // 0 is no-op.
  void* cannot_be_null;  // NULL is no-op.
  int can_be_zero;
  void* can_be_null;
  int optional_zero_default;  // Optional. 0 by default.
  void* optional_null_default;  // Optional. NULL by default.
};
```
* `cannot_be_zero` and `cannot_be_null` here can be deprecated by
  zero-initializing.
* `can_be_zero` and `can_be_null` need a MAJOR version bump for deprecation,
  since 0 and `NULL` are valid values for them.
* `optional_zero_default` and `optional_null_default` are optional fields that
  use 0 / `NULL` to indicate that the field is not provided. This needs an
  `SE_MAJOR` version bump for deprecation as well, since 0 and `NULL` are valid
  here.

For other unintentional changes which are caused by bugs (e.g., data was
forgotten to be initialized by mistake), file a Github issue.

### By Checking Struct Size
Backwards compatible changes within the same `SE_MINOR` version can only add new
members to a struct and cannot modify any existing member. Because of this, we
can check the byte offset of the variable we want to consume against the struct
size to see if the struct has the variable or not.

# Usage Example

Following are concrete examples of how TensorFlow remains compatible with
plug-ins when functionality is added to or removed from StreamExecutorInterface.

## Extending Functionality
The following snippet shows `void* new_field1` and `int new_field2` being added
to a `Toy` struct.

```diff
#define SE_MAJOR 1
- #define SE_MINOR 0
+ #define SE_MINOR 1  // Increment minor version.
#define SE_PATCH 0

typedef struct Toy {
  size_t struct_size;
  void* ext;          // Free-form data set by plugin.
  int32_t old_field;  // Device index.
+ void* new_field1;   // NULL is no-op.
+ int new_field2;     // 0 is no-op.
} Toy;

- // Evaluates to 20
- #define TOY_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, old_field)
+ // Evaluates to 36
+ #define TOY_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, new_field2)
```

To concisely cover compatibility of cases where structs are created by core
TensorFlow and by plug-ins, we will call the side that creates the struct
`producer`, and the side that takes the struct `consumer`.

### Producer Has Older Header Files

```cpp
// Producer implementation has v1.0.0 headers.
Toy* create_toy() {
  Toy* toy = new Toy{TOY_STRUCT_SIZE};
  // Based on header v1.0.0, toy->struct_size is 20.
  ...
  old_field = set_old_field();
  return toy;
}
// Consumer implementation has v1.1.0 headers.
void take_toy(const Toy* toy) {
  // Consumer checks for `struct_size` greater than 24 (offset of `new_field1`).
  // In this case, `toy->struct_size` = 20 so this `if` is not entered.
  if (toy->struct_size > offsetof(Toy, new_field1) && new_field1 != NULL) {
    // Safe to access `new_field1`.
  }
  // Consumer checks for `struct_size` greater than 32 (offset of `new_field2`).
  // In this case, `toy->struct_size` = 20 so this `if` is not entered.
  if (toy->struct_size > offsetof(Toy, new_field2) && new_field2 != 0) {
    // Safe to access `new_field2`.
  }
}
```

### Producer Has Newer Header Files

```cpp
// Producer implementation has v1.1.0 headers.
Toy* create_toy() {
  Toy* toy = new Toy{TOY_STRUCT_SIZE};
  // Based on header v1.1.0, toy->struct_size is 36.
  ...
  old_field = set_old_field();
  new_field1 = set_new_field1();
  new_field2 = set_new_field2();
  return toy;
}
// Consumer implementation has v1.0.0 headers.
void take_toy(const Toy* toy) {
  // `new_field1` and `new_field2` are safely ignored 
  // because consumer doesn't know about them.
}
```

If `producer` depends on `consumer` knowing about `new_field1` and `new_field2`,
adding `new_field1` and `new_field2` would be a backwards incompatible change
and `SE_MAJOR` should be bumped instead.
 
## Deprecating Functionality

When functionality is being deprecated, there will be comments next to the
member indicating so. The member is left in place to preserve the alignment and
offset of the existing structure members. General guidelines:
* Add comments saying which field will be deprecated.
* The minor update will still support `deprecating_feature` to allow time for
  transition. This would be a good time to raise concerns on Github.
* After the transition time has passed, `deprecating_feature` can be removed in
  a major update.
 
Since members are not allowed to be removed or reordered, refactors (e.g.,
renaming device_handle to dev_handle) or changing of member types (e.g., from
`int` to `float`) are considered as
[deprecation with extension](#Deprecation-with-extension).


```diff
#define SE_MAJOR 1
- #define SE_MINOR 1
+ #define SE_MINOR 2  // Increment minor version.
#define SE_PATCH 0

typedef struct Toy {
  size_t struct_size;
  void* ext;          // Free-form data set by plugin.
  int32_t old_field;  // Device index.
- void* new_field1;   // NULL is no-op.
+ void* new_field1;   // Deprecated.  // NULL is no-op.
  int new_field2;     // 0 is no-op.
} Toy;

// Evaluates to 36
#define TOY_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, new_field2)
```

To concisely cover compatibility of cases where structs are created by core
TensorFlow and by plug-ins, we will call the side that creates the struct
`producer`, and the side that takes the struct `consumer`.

### Producer Has Older Header Files

```diff
// Producer implementation has v1.1.0 headers.
Toy* create_toy() {
  Toy* toy = new Toy{TOY_STRUCT_SIZE};
  ...
  old_field = set_old_field();
  new_field1 = set_new_field1();
  new_field2 = set_new_field2();
  return toy;
}
// Consumer implementation has v1.2.0 headers.
void take_toy(const Toy* toy) {
- // Consumer removes the code using `new_field1`.
- if (toy->struct_size > offsetof(Toy, new_field1) && new_field1 != NULL) {
-   // Safe to access `new_field1`.
- }
  if (toy->struct_size > offsetof(Toy, new_field2) && new_field2 != 0) {
    // Safe to access `new_field2`.
  }
}
```

The producer, being older, initializes the recently deprecated `new_field1`.
Since `take_toy` does not access it anymore, `new_field1` will be safely ignored
(even though it was initialized).

### Producer Has Newer Header Files

```diff
// Producer implementation has v1.2.0 headers.
Toy* create_toy() {
  Toy* toy = new Toy{TOY_STRUCT_SIZE};
+ // `new_field1` is zero-initialized with the line above.
  ...
  old_field = set_old_field();
- new_field1 = set_new_field1();  // Stops setting the deprecated `new_field1`.
  new_field2 = set_new_field2();
  return toy;
}
// Consumer implementation has v1.1.0 headers.
void take_toy(const Toy* toy) {
+ // `new_field1` is `NULL` so it is safely ignored.
+ // Can also add code raise an error here when `NULL` is detected.
  if (toy->struct_size > offsetof(Toy, new_field1) && new_field1 != NULL) {
    // Safe to access `new_field1`.
  }
  if (toy->struct_size > offsetof(Toy, new_field2) && new_field2 != 0) {
    // Safe to access `new_field2`.
  }
}
```

This way, plug-ins can safely remove implementation of deprecated functionality.

## Deprecation with Extension

This is the more common form of deprecation where the struct is extended with a
new attribute that replaces an existing one. The analysis is the same as
[Adding functionality](#Adding-functionality) and
[Deprecating functionality](#Deprecating-functionality) combined.
General guidelines:
* Add comments saying which field will be deprecated and which one will replace
  it.
* Increment the minor version.
* The minor update will support both `name` and `better_name` to allow time for
  transition. This would be a good time to raise concerns on Github.
* After the transition time has passed, `name` can be removed in a major update.

Below are some examples.

```diff
#define SE_MAJOR 5
- #define SE_MINOR 0
+ #define SE_MINOR 1  // Increment minor version
#define SE_PATCH 0

// Case 1 - Renaming an attribute
typedef struct Device {
  size_t struct_size;
  void* ext;
  int32_t ordinal;

- const char* name;
+ const char* name;  // Deprecating soon. Use `better_name`.
  void* device_handle;
  const char* better_name;  // Replaces `name`.
} Device;


// Case 2 - Deprecation of an entire struct can be done without a replacement...
+ // `Device` struct will be deprecated soon.
typedef struct Device {
...
} Device;

// ...or with a replacement
+ // Replaces `Device`.
+ typedef struct BetterDevice {
+ ...
+ } Device;

// Case 3 - Renaming a function.
typedef struct ExportFunctions {
...
+ // create_device will be deprecated soon.
  void (*create_device)(Device* device);

+  // Replaces `create_device`.
+  void (*create_better_device)(BetterDevice* device);
} ExportFunctions;
```

# Limitations
* Maximum supported alignment is 8 bytes.
