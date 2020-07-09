**Authors**: yisitu@, penporn@, annarev@

**Date**: 7/9/20

In reply to a question on [PR #262](https://github.com/tensorflow/community/pull/262#issuecomment-653690654).

# TensorFlow Versioning Strategy

TensorFlow StreamExecutorInterface (SEI) uses struct_size for version checking. Struct members are not allowed to be removed or reordered.  Following are concrete examples of how TensorFlow remains compatible with plug-ins when functionality is added to or removed from StreamExecutorInterface. We will be using a simplified SE_Device as an example.
 
## When TensorFlow extends functionality
### Backwards compatibility
TensorFlow is compiled against a newer SEI header (v2), which has SE_Device extended with device_handle.
 
**Future TensorFlow compiled against StreamExecutorInterface v2**
```cpp
// StreamExecutorInterface header version 2
typedef struct SE_Device {
  size_t struct_size;
  void* next;           // Always set to zero, reserved by TF for future use.

  const char* name;
  size_t name_len;
  void* device_handle;
} SE_Device;

// Evaluates to 40
#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, device_handle)
```
 
The plugin was compiled against an older version of SEI header without device_handle. 
 
**Older Plugin compiled against StreamExecutorInterface v1**
```cpp
// StreamExecutorInterface header version 1
typedef struct SE_Device {
  size_t struct_size;
  void* next;

  const char* name;
  size_t name_len;
} SE_Device;


// Evaluates to 32
#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, name_len)

// Plugin Implementation

SE_Device* Plugin_CreateDevice() {
  SE_Device* se = new SE_Device{ SE_DEVICE_STRUCT_SIZE };
  // Based on header v1, se->struct_size will be 32
  ...
  return se;
}
```
 
TensorFlow checks that struct_size must be greater than the offset of device_handle before accessing it. 

```cpp
// TF Implementation

void TF_Foo(const SE_Device* device) {
  // TF checks for struct_size greater than 32.
  if (device->struct_size > offsetof(SE_Device, device_handle)) {
     // TF knows that device_handle can be safely read from.
     DoSomething(device->device_handle);
  }
}
```

### Forwards compatibility
 
In the event that a plugin is up to date or newer, se->struct_size would have been initialized to 48. This would then pass the TF_Foo() check above and device_handle can be safely accessed.
 
**Future Plugin compiled against StreamExecutorInterface v3**

```cpp
// StreamExecutorInterface header version 3
typedef struct SE_Device {
  size_t struct_size;
  void* next;

  const char* name;
  size_t name_len;
  void* device_handle;
  void* data;          // Added in v3
} SE_Device;

// Evaluates to 48
#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, data)


// Plugin Implementation

SE_Device* Plugin_CreateDevice() {
  SE_Device* se = new SE_Device{ SE_DEVICE_STRUCT_SIZE };
  // se->struct_size will be 48
  ...
  return se;
}
```
 
Using the same TF_Foo() above, TF_Foo()  was implemented before SE_Device::data was added after SE_Device::device_handle. Since TensorFlow only knows about the members that come before SE_Device::data, the newly added device->data will not be accessed.
 
## When TensorFlow deprecates functionality
When functionality is being deprecated, there will be comments next to the member indicating so. The member is left in place to preserve the alignment and offset of the existing structure members.
 
Since members are not allowed to be removed or reordered, refactors (e.g. renaming device_handle to dev_handle) or changing of member types (e.g. from int to float) are considered as deprecation.
 
### Backwards compatibility
SE_Device::data has been deprecated in version 4, and a comment in the header indicated as such.
 
**Future TensorFlow compiled against StreamExecutorInterface v4**

```cpp
// StreamExecutorInterface header version 4
typedef struct SE_Device {
  size_t struct_size;
  void* next;

  const char* name;
  size_t name_len;
  void* device_handle;
  void* data;            // Deprecated
} SE_Device;

// Evaluates to 48
#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, data)

// TF Implementation

void TF_Foo(const SE_Device* device) {
  // TF checks for struct_size greater than 32.
  if (device->struct_size > offsetof(SE_Device, device_handle)) {
     // TF knows that device_handle can be safely accessed.
     DoSomething(device->device_handle);
  }

  // TensorFlow removes implementation to stop using deprecated functionality.
  /*
  if (device->struct_size > offsetof(SE_Device, data)) {
     // TF knows that device->data can be safely accessed.
     DoSomething(device->data);
  }
  */

}
``

The plugin, being older, was initializing the recently deprecated SE_Device::data. Since TF_Foo() does not access it anymore, SE_Device::data will be safely ignored (even though it was initialized).
 
### Forwards compatibility
Plugins may choose to support older TensorFlow releases that have deprecated functionality.
 
In a simple case, TensorFlow is already performing input validation and capable of providing best effort forward compatibility with newer plugins. 
 
**Older TensorFlow compiled against StreamExecutorInterface v4 with data validation**

```cpp
void TF_Foo(const SE_Device* device) {
 ...
  // TF checks for struct_size greater than offset of data, and also validates device->data.
  if (device->struct_size > offsetof(SE_Device, data) && device->data != nullptr) {
     // TF knows that data can be safely accessed.
     DoSomething(device->data);
  }
}
```

 
This way, plugins can safely remove implementation of deprecated functionality.
 
**Future Plugin compiled against StreamExecutorInterface v5**
```cpp
// StreamExecutorInterface header version 5
typedef struct SE_Device {
  size_t struct_size;
  void* next;

  const char* name;
  size_t name_len;
  void* device_handle;
  void* data;            // Deprecated in v4
  void* data2;
} SE_Device;

// Evaluates to 56
#define SE_DEVICE_STRUCT_SIZE TF_OFFSET_OF_END(SE_Device, data2)


// Plugin Implementation

SE_Device* Plugin_CreateDevice() {
  SE_Device* se = new SE_Device{ SE_DEVICE_STRUCT_SIZE };
  // se->struct_size will be 56
  se->device_handle = GetHandle();

  // se->data was deprecated so ignore it. It was already zero initialized 
  //   at “SE_Device{ SE_DEVICE_STRUCT_SIZE }” above.

  se->data2 = GetData2();
  return se;
}
```
 
In a more complex scenario, an older TensorFlow release might consume deprecated functionality for granted.
 
**Older TensorFlow compiled against StreamExecutorInterface v4 without data validation**

```cpp
void TF_Foo(const SE_Device* device) {
 ...
  // TF checks for struct_size greater than offset of data.
  // No input validation.
  if (device->struct_size > offsetof(SE_Device, data)) {
     // Will crash on null pointer dereference.
     Dereference(device->data);
  }
}
```
 
In this case, it is recommended for plugins to continue to keep the deprecated implementation around. Once the plugin stops supporting the latest version of TensorFlow that uses the deprecated functionality, the implementation can be safely removed. This comes at the cost of maintenance of legacy deprecated code on the plugin side.
 
## Limitations
* Maximum supported alignment is 8 bytes.
