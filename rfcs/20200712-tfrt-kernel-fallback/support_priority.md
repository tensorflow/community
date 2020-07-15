## Order of OpKernelContext/OpKernelConstruction method support priority in Kernel Fallback

Higher priority is assigned to methods that enable us to support the most number of kernels. At the same time, a high-priority method that requires a significant amount of work might be implemented later. `resource_manager` specifically is a frequently-used method, but would require a lot of work. Therefore, we might first implement other methods. Also, number of supported kernels is calculated assuming all methods above it in the table are implemented.

Note that this list excludes some the methods we already support in our prototype, specifically:

* OpKernelConstruction: `GetAttr`, `CtxFailure`, `CtxFailureWithWarning`.
* OpKernelContext: `input`, `num_inputs`, `set_output`, `num_outputs`, `allocate_output`, `expected_output_dtype`, `eigen_device`, `CtxFailure`, `CtxFailureWithWarning`.

Note: This table is created by analyzing which methods are called down to 2 function calls of indirection. As such it is an approximation.

| Method | Num of kernels supported if implemented |
| :----- | :-------------------------------------- |
| tensorflow::OpKernelContext::resource_manager | 46 |
| tensorflow::OpKernelContext::forward_input_or_allocate_output | 35 |
| tensorflow::OpKernelContext::allocate_temp | 40 |
| tensorflow::OpKernelContext::device | 26 |
| tensorflow::OpKernelContext::input_list | 25 |
| tensorflow::OpKernelContext::SetStatus | 25 |
| tensorflow::OpKernelContext::op_device_context | 23 |
| tensorflow::OpKernelConstruction::HasAttr | 18 |
| tensorflow::OpKernelContext::env | 15 |
| tensorflow::OpKernelContext::output_list | 12 |
| tensorflow::OpKernelConstruction::device_type | 10 |
| tensorflow::OpKernelContext::eigen_cpu_device | 9 |
| tensorflow::OpKernelContext::eigen_gpu_device | 8 |
| tensorflow::OpKernelContext::allocate_persistent | 7 |
| tensorflow::OpKernelContext::function_library | 6 |
| tensorflow::OpKernelContext::set_output_ref | 5 |
| tensorflow::OpKernelConstruction::env | 5 |
| tensorflow::OpKernelContext::cancellation_manager | 9 |
| tensorflow::OpKernelContext::op_kernel | 4 |
| tensorflow::OpKernelContext::forward_input_or_allocate_temp | 4 |
| tensorflow::OpKernelContext::forward_input | 4 |
| tensorflow::OpKernelContext::input_memory_type | 6 |
| tensorflow::OpKernelContext::input_alloc_attr | 6 |
| tensorflow::OpKernelConstruction::input_type | 4 |
| tensorflow::OpKernelConstruction::device | 4 |
| tensorflow::OpKernelContext::output_required | 3 |
| tensorflow::OpKernelConstruction::resource_manager | 3 |
| tensorflow::OpKernelConstruction::graph_def_version | 3 |
| tensorflow::OpKernelConstruction::allocate_persistent | 3 |
| tensorflow::OpKernelConstruction::output_type | 9 |
| tensorflow::OpKernelContext::MatchSignature | 5 |
| tensorflow::OpKernelContext::session_state | 2 |
| tensorflow::OpKernelContext::mutable_input | 2 |
| tensorflow::OpKernelContext::input_ref_mutex | 2 |
| tensorflow::OpKernelContext::input_dtype | 31 |
| tensorflow::OpKernelContext::forward_ref_input_to_ref_output | 34 |
| tensorflow::OpKernelContext::step_container | 8 |
| tensorflow::OpKernelContext::expected_output_dtype | 3 |
| tensorflow::OpKernelContext::slice_reader_cache | 2 |
| tensorflow::OpKernelContext::output_alloc_attr | 2 |
| tensorflow::OpKernelContext::mutable_output | 2 |
| tensorflow::OpKernelContext::forward_input_to_output_with_shape | 2 |
| tensorflow::OpKernelConstruction::def | 2 |
| tensorflow::OpKernelConstruction::SetStatus | 2 |
| tensorflow::OpKernelContext::tensor_store | 1 |
| tensorflow::OpKernelContext::step_id | 1 |
| tensorflow::OpKernelContext::frame_iter | 8 |
| tensorflow::OpKernelContext::collective_executor | 4 |
| tensorflow::OpKernelContext::session_metadata | 1 |
| tensorflow::OpKernelContext::run_all_kernels_inline | 1 |
| tensorflow::OpKernelContext::mutable_input_list | 1 |
| tensorflow::OpKernelConstruction::function_library | 1 |


