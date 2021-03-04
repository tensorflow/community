# GRPC Fail Fast By Default

| Status        | Proposed                                                |
| :------------ | :------------------------------------------------------ |
| **RFC #**     | [355](https://github.com/tensorflow/community/pull/355) |
| **Author(s)** | Haoyu Zhang (haoyuzhang@google.com)                     |
| **Sponsor**   | Bramandia Ramadhana (bramandia@google.com)              |
| **Updated**   | 2021-03-04                                              |

## Objective

We propose to set the default value of the `GRPC_FAIL_FAST` environment variable
to `use_caller`. This change prevents TensorFlow distributed jobs from hanging
indefinitely due to task failures, and allows users and TF libraries (e.g.,
distribution strategies) to handle the connection errors for better failure and
preemption recovery.

## Background

`GRPC_FAIL_FAST` is a TensorFlow distributed runtime environment variable that
controls the behavior of RPC requests when observing a network disconnection
with remote servers. It can be configured to the following values:

*   `true`, which immediately reports an `UnavailableError` when there is a
    connection issue for all RPCs, regardless of the per-RPC configurations;
*   `false`, which blocks and waits until successfully connected to the remote
    server (see
    [gRPC `wait_for_ready`](https://github.com/grpc/grpc/blob/master/doc/wait-for-ready.md)),
    regardless of the per-RPC configurations;
*   `use_caller`, which allows customization per RPC basis; in the current
    implementation, `true` is used for RPCs used in distributed execution (such
    as `RecvTensor`, `RunComponentFunction`), and `false` is used for RPCs in
    initializing remote execution environments (e.g., `GetStatus`).

The default value of `GRPC_FAIL_FAST` is currently set to `false`. One of the
consequences is that users and/or high-level distribute libraries (such as
`ParameterServerStrategy`) need to
[manually configure this environment variable](https://github.com/tensorflow/tensorflow/blob/1178262a2a55fa634a2390291fc633c515e28884/tensorflow/python/distribute/parameter_server_strategy_v2.py#L106)
to receive reasonable exceptions when workers fail / get preempted; otherwise
the cluster will hang and cannot recover from failures.

## Proposed Change

We propose to set the default value of `GRPC_FAIL_FAST` to `use_caller`. By
doing so, the runtime reports errors quickly to detect remote server failures
during execution, while still allowing the client to start early and wait for
remote servers to establish initial connections. This should be the desired
behavior for most use cases.

In the context of TensorFlow 2, the default behavior of the following RPCs used
for distributed execution will be changed from hanging on failures (current
behavior) to immediately reporting failures (after the change):

*   `EagerService.CreateContext`
*   `EagerService.UpdateContext`
*   `EagerService.WaitQueueDone`
*   `EagerService.KeepAlive`
*   `EagerService.Enqueue`
*   `EagerService.RunComponentFunction`
*   `WorkerService.RecvTensor`
*   `WorkerService.RecvBuf`

The default behavior of the following RPC will not change: it will still hang if
the remote task cannot be reached.

*   `WorkerService.GetStatus`

The `GetStatus` RPC is typically the first RPC sent from the client to
initialize a distributed execution environment, in both the single- and the
multi-client mode. The underlying implementation uses GRPC's
[`wait_for_ready`](https://github.com/grpc/grpc/blob/master/doc/wait-for-ready.md)
flag, which allows the client to start before the remote server in the
deployment.

## User Impact

When this change is made to the codebase, subsequent TensorFlow 2 releases will
have this new default behavior. TensorFlow 1.x users who use the stable releases
(e.g., TensorFlow 1.15 or earlier) should not be affected by this change. Users
who build TensorFlow directly from source at the head will also be affected.

Most users should see the new default as expected behaviors in distributed
execution. Users can take advantage of the built-in fault tolerance support in
`ParameterServerStrategy` without having to make changes to the environment
variable configurations. In other setups, exceptions will be raised to the model
training loop code, where users can catch and handle these errors with custom
logic instead of hanging indefinitely.

Certain users might receive "false alarms" if there are transient connection
errors to the remote servers. We expect this to happen very rarely since GRPC
(built on top of HTTP and TCP protocols) should already handle packet drops and
network flakinesses in most cases, and only report errors when there are real
network or server failures. However, if this does happen to some users, please
set `GRPC_FAIL_FAST=false` to override the default value and revert to the
previous behavior. Please also file an issue to inform the TensorFlow Runtime
team.

