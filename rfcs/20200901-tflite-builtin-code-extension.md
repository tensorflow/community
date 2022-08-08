# Builtin operator code extension in TensorFlow Lite

Status        | Accepted
:------------ | :----------------------------------
**Author(s)** | Jae sung Chung (jaesung@google.com)
**Sponsor**   | Jared Duke (jdduke@google.com)
**Updated**   | 2020-09-01

## Objective

This RFC proposes a FlatBuffer schema change in TensorFlow Lite, which extends
the builtin operator code in the `OperatorCode` table section to resolve the
shortage problem in the builtin operator code. Even though Flatbuffer is an
internal details of the TensorFlow Lite, some advanced users, who write
visualization tools or third-party delegates, depend on reading Flatbuffer
schema directly via C++ API. This RFC focuses on the solution for the shortage
problem and its compatibility issues.

### Goals:

*   Discuss how to resolve the builtin operator code shortage problem in
    TensorFlow Lite Flatbuffer schema without breaking backward-compatibility of
    the existing TensorFlow Lite Flatbuffer schema version 3 files in future
    versions of the TensorFlow Lite library.

## Background

A byte type is used for builtin operator code in Flatbuffer, so only 127 builtin
operators can be added. TensorFlow Lite builtin operator library already reached
126 operators.

```
enum BuiltinOperator : byte {
  ...
}
```

TensorFlow Lite keeps extending to support more domains in ML and users require
more builtin operators to increase ML model coverage on device, for example Hash
table support and so on.

### Compatibility issues

TensorFlow Lite has used its Flatbuffer schema version 3 since 2018 and a lot of
TensorFlow Lite models can be found over the TensorFlow project sites.
Supporting the schema version 3 is critical for future versions of TensorFlow
Lite.

The compatibility issue of newly generated models, that will contain new builtin
operators after this proposal's change, from old TensorFlow Lite libraries won't
be a problem because the TensorFlow Lite library's version always should be the
same or the newer than the TensorFlow Lite Converter API's version for builtin
operator availability.

## Proposal

### Schema changes

For resolving compatibility issues easily, the old builtin operator code will
remain and the new builtin operator code will be added into the last position as
follows.

```
enum BuiltinOperator : int32 {
  ... // Existing builtin operator codes
  PLACEHOLDER_FOR_GREATER_OP_CODES = 127,
  // New builtin operators will be added in here.
  BROADCAST_TO = 128
}

// An OperatorCode can be an enum value (BuiltinOperator) if the operator is a
// builtin, or a string if the operator is custom.
table OperatorCode {
  // This field is for backward compatibility. This field will be used when
  // the value of the extended builtin_code field has less than
  // BulitinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES.
  deprecated_builtin_code:byte;
  custom_code:string;

  // The version of the operator. The version need to be bumped whenever new
  // parameters are introduced into an op.
  version:int = 1;

  // This field is introduced for resolving op builtin operator code shortage problem
  // (the original BuiltinOperator enum field was represented as a byte).
  // This field will be used when the value of the extended builtin_code field
  // is greater than BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES.
  builtin_code:BuiltinOperator;
}

```

### How to Read `builtin_code`

The builtin operator code value will co-exist in the two fields. Old schema
version 3 models will have the default value in the new field. For those old
schema models, the `deprecated_builtin_code` field should be read when the
default value is set in the new `builtin_code` field.

```
  BuiltinOperator builtin_code = (op_code->builtin_code ? op_code->builtin_code
        : static_cast<BuiltinOperator>(op_code->deprecated_builtin_code));
```

#### Handling compatibility issues

|                    | Old .tflite model                        | New .tflite model                                                                                                                                  |
| ------------------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Old TFLite runtime | OK<br>(Current behavior)                 | OK<br>(As long as the model is not using<br>any *new* builtin ops. If it is, it will<br>fail as it would when using any<br> other unsupported op.) |
| New TFLite runtime | OK<br>(Uses the legacy builtin op field) | OK<br>(Uses the new builtin op field)                                                                                                              |

The details will be described in the following sections.

##### Supporting schema version 3 models

The TensorFlow Lite library built after the proposal will read the existing
`deprecated_builtin_code` field from old model files.

The new `builtin_code` field is not available in the version 3 models. Flatbuffer
library will feed zero value, which is the default value in the version 3a schema. The
actual builtin operator code value will exist in the deprecated, renamed
`deprecated_builtin_code` field.

##### Compatibility with old TensorFlow Lite libraries

The TensorFlow Lite library built before the proposal will read the existing
`deprecated_builtin_code` field. However, it will not handle new operations
declared above 127. This is intended behavior, as it doesn't have a definition
for the higher op values anyway.

### Schema version

Schema version will remain as before since we are not going to break backward
compatibility in the schema level. However, in the viewpoint of the Flatbuffer
tool, renaming a field is treated as compatibility breakage even though the
runtime code could handle schema formats before and after both. The schema
change can be regarded as a minor version. The schema version "3a" can be an
option.

### New utils for schema manipulation

After the proposal is landed, the codes, that read builtin operator code from
TensorFlow Lite Flatbuffer, requires accessing both `builtin_code` and
`deprecated_builtin_code` fields.

To avoid redundant logics in a lot of places, the RFC proposes the following
helper functions in the new C++ library, `tensorflow/lite/schema:schema_utils`.

```
BuiltinOperator GetBuiltinCode(const OperatorCode *op_code);
BuiltinOperator GetBuiltinCode(const OperatorCodeT *op_code);
```

The above library also provides the following `OperatorCode` table object
creation methods for backward compatibility.

These are manually copied from the flatbuffer generated code from schema version 3.
They serve as overloads for the version 3a's CreateOperatorCode functions in
schema_generated.h and enable code that still assumes flatbuffer schema version 3 to be
unchanged with the inclusion of the schema_utils header.

```
flatbuffersf:Offset<OperatorCode> CreateOperatorCode(
    flatbuffers::FlatBufferBuilder &_fbb,
    tflite::BuiltinOperator builtin_code = tflite::BuiltinOperator_ADD,
    flatbuffers::Offset<flatbuffers::String> custom_code = 0,
    int32_t version = 1);

flatbuffers::Offset<OperatorCode> CreateOperatorCodeDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    tflite::BuiltinOperator builtin_code = tflite::BuiltinOperator_ADD,
    const char *custom_code = nullptr, int32_t version = 1);
```
