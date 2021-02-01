exports_files(["LICENSE"])

load(
    "@org_tensorflow_plugin//third_party:common.bzl",
    "template_rule",
)
load("@local_config_dpcpp//dpcpp:build_defs.bzl", "if_dpcpp_build_is_configured")

config_setting(
    name = "clang_linux_x86_64",
    values = {
        "cpu": "k8",
        "define": "using_clang=true",
    },
)

# Create the file mkldnn_version.h with OneDNN version numbers.
# Currently, the version numbers are hard coded here. If OneDNN is upgraded then
# the version numbers have to be updated manually. The version numbers can be
# obtained from the PROJECT_VERSION settings in CMakeLists.txt. The variable is
# set to "version_major.version_minor.version_patch". The git hash version can
# be set to NA.
# TODO(agramesh1) Automatically get the version numbers from CMakeLists.txt.

template_rule(
    name = "dnnl_version_h",
    src = "include/oneapi/dnnl/dnnl_version.h.in",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = {
        "@DNNL_VERSION_MAJOR@": "1",
        "@DNNL_VERSION_MINOR@": "95",
        "@DNNL_VERSION_PATCH@": "0",
        "@DNNL_VERSION_HASH@": "7bd73fb5297425e3d38a80a49559f55fdbad2d8c",
    },
)

template_rule(
    name = "dnnl_config_h",
    src = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = {
        "#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}": if_dpcpp_build_is_configured("#define DNNL_CPU_RUNTIME DNNL_RUNTIME_DPCPP", "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_SYCL"),
        "#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}": "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ",
        "#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}": if_dpcpp_build_is_configured("#define DNNL_GPU_RUNTIME DNNL_RUNTIME_DPCPP", "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_SYCL"),
        "#cmakedefine DNNL_SYCL_DPCPP": if_dpcpp_build_is_configured("#define DNNL_SYCL_DPCPP", "/* #undef DNNL_SYCL_DPCPP */"),
        "#cmakedefine DNNL_SYCL_COMPUTECPP": if_dpcpp_build_is_configured("/*#undef DNNL_SYCL_COMPUTECPP*/", "#define DNNL_SYCL_COMPUTECPP"),
        "#cmakedefine DNNL_WITH_LEVEL_ZERO": if_dpcpp_build_is_configured("/*#undef DNNL_WITH_LEVEL_ZERO*/", "/*#undef DNNL_WITH_LEVEL_ZERO*/"),
        "#cmakedefine DNNL_SYCL_CUDA": if_dpcpp_build_is_configured("/*#undef DNNL_SYCL_CUDA*/", "/*#undef DNNL_SYCL_CUDA*/"),
    },
)

template_rule(
    name = "kernel_list_cpp",
    src = "src/gpu/ocl/ocl_kernel_list.cpp.in",
    out = "src/gpu/ocl/ocl_kernel_list.cpp",
    substitutions = {
        "@KER_LIST_EXTERN@": "extern const char *gen12lp_gemm_nocopy_scale_x8x8s32_kernel[]; \n \
extern const char *gen12lp_gemm_nocopy_x8x8s32_kernel[]; \
extern const char *gen9_gemm_beta_kernel[]; \
extern const char *gen9_gemm_compute_kernel[]; \
extern const char *gen9_gemm_copy_kernel[]; \
extern const char *gen9_gemm_nocopy_f16_kernel[]; \
extern const char *gen9_gemm_nocopy_f32_kernel[]; \
extern const char *gen9_gemm_nocopy_scale_x8x8s32_kernel[]; \
extern const char *gen9_gemm_nocopy_superkernel_f32_kernel[]; \
extern const char *gen9_gemm_nocopy_x8x8s32_kernel[]; \
extern const char *ref_gemm_kernel[]; \
extern const char *gemm_inner_product_kernel[]; \
extern const char *gemm_post_ops_inner_product_kernel[]; \
extern const char *gen12lp_1x1_conv_fwd_data_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_bwd_data_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_dw_fwd_data_mb_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_dw_fwd_data_ow_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_fwd_data_first_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_fwd_data_mb_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_fwd_data_ow_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_nhwc_fwd_dw_mb_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_nhwc_fwd_dw_ow_block_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_nhwc_fwd_first_x8s8s32x_kernel[]; \
extern const char *gen12lp_conv_nhwc_fwd_x8s8s32x_kernel[]; \
extern const char *gen12lp_nhwc_1x1_conv_fwd_x8s8s32x_kernel[]; \
extern const char *gen12lp_x8s8s32x_compensation_kernel[]; \
extern const char *gen9_binary_kernel[]; \
extern const char *gen9_bnorm_kernel[]; \
extern const char *gen9_conv_bwd_data_kernel[]; \
extern const char *gen9_conv_bwd_weights_kernel[]; \
extern const char *gen9_conv_dw_bwd_data_kernel[]; \
extern const char *gen9_conv_dw_fwd_data_kernel[]; \
extern const char *gen9_conv_fwd_data_kernel[]; \
extern const char *gen9_conv_nhwc_bwd_data_kernel[]; \
extern const char *gen9_conv_nhwc_bwd_weights_f32_kernel[]; \
extern const char *gen9_conv_nhwc_fwd_data_kernel[]; \
extern const char *gen9_eltwise_kernel[]; \
extern const char *gen9_pooling_kernel[]; \
extern const char *gen9_softmax_kernel[]; \
extern const char *gen9_wino_conv_fwd_data_2x3_kernel[]; \
extern const char *gen9_wino_conv_fwd_data_6x3_kernel[]; \
extern const char *ref_binary_kernel[]; \
extern const char *ref_bnorm_kernel[]; \
extern const char *ref_convolution_kernel[]; \
extern const char *ref_deconv_backward_bias_kernel[]; \
extern const char *ref_eltwise_kernel[]; \
extern const char *ref_inner_product_kernel[]; \
extern const char *ref_layer_normalization_kernel[]; \
extern const char *ref_lrn_kernel[]; \
extern const char *ref_matmul_kernel[]; \
extern const char *ref_pooling_kernel[]; \
extern const char *ref_reduction_kernel[]; \
extern const char *ref_resampling_kernel[]; \
extern const char *ref_shuffle_kernel[]; \
extern const char *ref_softmax_kernel[]; \
extern const char *ref_zero_pad_kernel[]; \
extern const char *ref_rnn_kernel[]; \
extern const char *rnn_reorder_kernel[]; \
extern const char *simple_concat_kernel[]; \
extern const char *simple_reorder_kernel[]; \
extern const char *simple_sum_kernel[]; ",
        "@KER_LIST_ENTRIES@": "{ \"gen12lp_gemm_scale_x8x8s32\", gen12lp_gemm_nocopy_scale_x8x8s32_kernel }, \n \
{ \"gen12lp_gemm_compute_x8x8s32\", gen12lp_gemm_nocopy_x8x8s32_kernel },\
{ \"gen12lp_gemm_compute_x8x8s32\", gen12lp_gemm_nocopy_x8x8s32_kernel },\
{ \"gen12lp_gemm_compute_x8x8s32\", gen12lp_gemm_nocopy_x8x8s32_kernel },\
{ \"gen12lp_gemm_compute_x8x8s32\", gen12lp_gemm_nocopy_x8x8s32_kernel },\
{ \"gen9_gemm_beta\", gen9_gemm_beta_kernel },\
{ \"gen9_gemm_compute\", gen9_gemm_compute_kernel },\
{ \"gen9_gemm_copy\", gen9_gemm_copy_kernel },\
{ \"gen9_gemm_nocopy_f16\", gen9_gemm_nocopy_f16_kernel },\
{ \"gen9_gemm_nocopy_f16\", gen9_gemm_nocopy_f16_kernel },\
{ \"gen9_gemm_nocopy_f16\", gen9_gemm_nocopy_f16_kernel },\
{ \"gen9_gemm_nocopy_f16\", gen9_gemm_nocopy_f16_kernel },\
{ \"gen9_gemm_nocopy_f32\", gen9_gemm_nocopy_f32_kernel },\
{ \"gen9_gemm_nocopy_f32\", gen9_gemm_nocopy_f32_kernel },\
{ \"gen9_gemm_nocopy_f32\", gen9_gemm_nocopy_f32_kernel },\
{ \"gen9_gemm_nocopy_f32\", gen9_gemm_nocopy_f32_kernel },\
{ \"gen9_gemm_scale_x8x8s32\", gen9_gemm_nocopy_scale_x8x8s32_kernel },\
{ \"gen9_gemm_nocopy_superkernel_f32\", gen9_gemm_nocopy_superkernel_f32_kernel },\
{ \"gen9_gemm_nocopy_superkernel_f32\", gen9_gemm_nocopy_superkernel_f32_kernel },\
{ \"gen9_gemm_compute_x8x8s32\", gen9_gemm_nocopy_x8x8s32_kernel },\
{ \"ref_gemm\", ref_gemm_kernel },\
{ \"gemm_inner_product_forward_bias\", gemm_inner_product_kernel },\
{ \"gemm_inner_product_backward_weights_bias\", gemm_inner_product_kernel },\
{ \"gemm_post_ops_inner_product\", gemm_post_ops_inner_product_kernel },\
{ \"gen12lp_1x1_conv_fwd_x8s8s32x\", gen12lp_1x1_conv_fwd_data_x8s8s32x_kernel },\
{ \"conv_bwd_data_x8s8s32x\", gen12lp_conv_bwd_data_x8s8s32x_kernel },\
{ \"conv_dw_fwd_mb_block_x8s8s32x\", gen12lp_conv_dw_fwd_data_mb_block_x8s8s32x_kernel },\
{ \"conv_dw_fwd_ow_block_x8s8s32x\", gen12lp_conv_dw_fwd_data_ow_block_x8s8s32x_kernel },\
{ \"conv_fwd_first_x8s8s32x\", gen12lp_conv_fwd_data_first_x8s8s32x_kernel },\
{ \"conv_fwd_mb_block_x8s8s32x\", gen12lp_conv_fwd_data_mb_block_x8s8s32x_kernel },\
{ \"conv_fwd_ow_block_x8s8s32x\", gen12lp_conv_fwd_data_ow_block_x8s8s32x_kernel },\
{ \"conv_nhwc_fwd_dw_mb_block_x8s8s32x\", gen12lp_conv_nhwc_fwd_dw_mb_block_x8s8s32x_kernel },\
{ \"conv_nhwc_fwd_dw_ow_block_x8s8s32x\", gen12lp_conv_nhwc_fwd_dw_ow_block_x8s8s32x_kernel },\
{ \"conv_nhwc_fwd_first_x8s8s32x\", gen12lp_conv_nhwc_fwd_first_x8s8s32x_kernel },\
{ \"conv_nhwc_fwd_x8s8s32x\", gen12lp_conv_nhwc_fwd_x8s8s32x_kernel },\
{ \"gen12lp_nhwc_1x1_conv_fwd_x8s8s32x\", gen12lp_nhwc_1x1_conv_fwd_x8s8s32x_kernel },\
{ \"gen12lp_x8s8s32x_compensation\", gen12lp_x8s8s32x_compensation_kernel },\
{ \"gen12lp_x8s8s32x_compensation\", gen12lp_x8s8s32x_compensation_kernel },\
{ \"gen9_binary\", gen9_binary_kernel },\
{ \"gen9_binary\", gen9_binary_kernel },\
{ \"gen9_calc_mean\", gen9_bnorm_kernel },\
{ \"gen9_reduce_mean\", gen9_bnorm_kernel },\
{ \"gen9_calc_variance\", gen9_bnorm_kernel },\
{ \"gen9_reduce_variance\", gen9_bnorm_kernel },\
{ \"gen9_bnorm_fwd\", gen9_bnorm_kernel },\
{ \"gen9_calculate_stats\", gen9_bnorm_kernel },\
{ \"gen9_reduce_stats\", gen9_bnorm_kernel },\
{ \"gen9_bnorm_bwd\", gen9_bnorm_kernel },\
{ \"gen9_conv_bwd_data\", gen9_conv_bwd_data_kernel },\
{ \"gen9_conv_bwd_weights\", gen9_conv_bwd_weights_kernel },\
{ \"gen9_conv_dw_bwd_data\", gen9_conv_dw_bwd_data_kernel },\
{ \"gen9_conv_dw_fwd\", gen9_conv_dw_fwd_data_kernel },\
{ \"gen9_conv_fwd\", gen9_conv_fwd_data_kernel },\
{ \"gen9_conv_nhwc_bwd_data\", gen9_conv_nhwc_bwd_data_kernel },\
{ \"gen9_conv_nhwc_bwd_weights\", gen9_conv_nhwc_bwd_weights_f32_kernel },\
{ \"gen9_conv_nhwc_fwd\", gen9_conv_nhwc_fwd_data_kernel },\
{ \"gen9_eltwise_fwd\", gen9_eltwise_kernel },\
{ \"gen9_eltwise_bwd\", gen9_eltwise_kernel },\
{ \"gen9_pooling_fwd\", gen9_pooling_kernel },\
{ \"gen9_pooling_bwd\", gen9_pooling_kernel },\
{ \"gen9_softmax_fwd\", gen9_softmax_kernel },\
{ \"gen9_wino_wei_transform_2x3\", gen9_wino_conv_fwd_data_2x3_kernel },\
{ \"gen9_wino_src_transform_2x3\", gen9_wino_conv_fwd_data_2x3_kernel },\
{ \"gen9_wino_dst_transform_2x3\", gen9_wino_conv_fwd_data_2x3_kernel },\
{ \"gen9_wino_conv_fwd_2x3\", gen9_wino_conv_fwd_data_2x3_kernel },\
{ \"gen9_wino_wei_transform_6x3\", gen9_wino_conv_fwd_data_6x3_kernel },\
{ \"gen9_wino_conv_fwd_6x3\", gen9_wino_conv_fwd_data_6x3_kernel },\
{ \"ref_binary\", ref_binary_kernel },\
{ \"ref_binary\", ref_binary_kernel },\
{ \"calculate_mean\", ref_bnorm_kernel },\
{ \"calculate_variance\", ref_bnorm_kernel },\
{ \"calculate_mean\", ref_bnorm_kernel },\
{ \"calculate_variance\", ref_bnorm_kernel },\
{ \"reduce_mean\", ref_bnorm_kernel },\
{ \"reduce_variance\", ref_bnorm_kernel },\
{ \"ref_bnorm_fwd\", ref_bnorm_kernel },\
{ \"calculate_stats\", ref_bnorm_kernel },\
{ \"reduce_stats\", ref_bnorm_kernel },\
{ \"calculate_stats\", ref_bnorm_kernel },\
{ \"reduce_stats\", ref_bnorm_kernel },\
{ \"ref_bnorm_bwd\", ref_bnorm_kernel },\
{ \"ref_convolution_fwd\", ref_convolution_kernel },\
{ \"ref_convolution_bwd_data\", ref_convolution_kernel },\
{ \"ref_convolution_bwd_weights\", ref_convolution_kernel },\
{ \"ref_deconv_backward_bias\", ref_deconv_backward_bias_kernel },\
{ \"ref_eltwise_fwd\", ref_eltwise_kernel },\
{ \"ref_eltwise_bwd\", ref_eltwise_kernel },\
{ \"ref_inner_product_fwd\", ref_inner_product_kernel },\
{ \"ref_inner_product_bwd_data\", ref_inner_product_kernel },\
{ \"ref_inner_product_bwd_weights\", ref_inner_product_kernel },\
{ \"ref_lnorm_fwd\", ref_layer_normalization_kernel },\
{ \"ref_lnorm_bwd_scaleshift\", ref_layer_normalization_kernel },\
{ \"ref_lnorm_bwd\", ref_layer_normalization_kernel },\
{ \"ref_lrn_fwd\", ref_lrn_kernel },\
{ \"ref_lrn_bwd\", ref_lrn_kernel },\
{ \"ref_matmul\", ref_matmul_kernel },\
{ \"ref_pooling_fwd\", ref_pooling_kernel },\
{ \"ref_pooling_bwd\", ref_pooling_kernel },\
{ \"ref_reduce\", ref_reduction_kernel },\
{ \"ref_resampling_fwd\", ref_resampling_kernel },\
{ \"ref_resampling_bwd\", ref_resampling_kernel },\
{ \"ref_shuffle\", ref_shuffle_kernel },\
{ \"ref_softmax_fwd_generic\", ref_softmax_kernel },\
{ \"ref_softmax_bwd_generic\", ref_softmax_kernel },\
{ \"ref_zero_pad\", ref_zero_pad_kernel },\
{ \"ref_rnn_copy_init_layer\", ref_rnn_kernel },\
{ \"ref_rnn_copy_init_iter\", ref_rnn_kernel },\
{ \"ref_rnn_copy_res_layer\", ref_rnn_kernel },\
{ \"ref_rnn_copy_res_iter\", ref_rnn_kernel },\
{ \"ref_rnn_ws_set\", ref_rnn_kernel },\
{ \"ref_rnn_ws_print\", ref_rnn_kernel },\
{ \"ref_rnn_bias_prepare\", ref_rnn_kernel },\
{ \"ref_rnn_elemwise_fwd\", ref_rnn_kernel },\
{ \"ref_rnn_elemwise_fwd\", ref_rnn_kernel },\
{ \"ref_rnn_elemwise_bwd\", ref_rnn_kernel },\
{ \"ref_rnn_gates_reduction\", ref_rnn_kernel },\
{ \"wei_reorder\", rnn_reorder_kernel },\
{ \"simple_concat\", simple_concat_kernel },\
{ \"simple_reorder\", simple_reorder_kernel },\
{ \"simple_sum\", simple_sum_kernel } \n",
    },
)

genrule(
    name = "onednn_genrule",
    outs = [
        "gemm_inner_product_kernel.cpp",
        "gemm_post_ops_inner_product_kernel.cpp",
        "gen12lp_1x1_conv_fwd_data_x8s8s32x_kernel.cpp",
        "gen12lp_conv_bwd_data_x8s8s32x_kernel.cpp",
        "gen12lp_conv_dw_fwd_data_mb_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_dw_fwd_data_ow_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_fwd_data_first_x8s8s32x_kernel.cpp",
        "gen12lp_conv_fwd_data_mb_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_fwd_data_ow_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_nhwc_fwd_dw_mb_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_nhwc_fwd_dw_ow_block_x8s8s32x_kernel.cpp",
        "gen12lp_conv_nhwc_fwd_first_x8s8s32x_kernel.cpp",
        "gen12lp_conv_nhwc_fwd_x8s8s32x_kernel.cpp",
        "gen12lp_gemm_nocopy_scale_x8x8s32_kernel.cpp",
        "gen12lp_gemm_nocopy_x8x8s32_kernel.cpp",
        "gen12lp_nhwc_1x1_conv_fwd_x8s8s32x_kernel.cpp",
        "gen12lp_x8s8s32x_compensation_kernel.cpp",
        "gen9_binary_kernel.cpp",
        "gen9_bnorm_kernel.cpp",
        "gen9_conv_bwd_data_kernel.cpp",
        "gen9_conv_bwd_weights_kernel.cpp",
        "gen9_conv_dw_bwd_data_kernel.cpp",
        "gen9_conv_dw_fwd_data_kernel.cpp",
        "gen9_conv_fwd_data_kernel.cpp",
        "gen9_conv_nhwc_bwd_data_kernel.cpp",
        "gen9_conv_nhwc_bwd_weights_f32_kernel.cpp",
        "gen9_conv_nhwc_fwd_data_kernel.cpp",
        "gen9_eltwise_kernel.cpp",
        "gen9_gemm_beta_kernel.cpp",
        "gen9_gemm_compute_kernel.cpp",
        "gen9_gemm_copy_kernel.cpp",
        "gen9_gemm_nocopy_f16_kernel.cpp",
        "gen9_gemm_nocopy_f32_kernel.cpp",
        "gen9_gemm_nocopy_scale_x8x8s32_kernel.cpp",
        "gen9_gemm_nocopy_superkernel_f32_kernel.cpp",
        "gen9_gemm_nocopy_x8x8s32_kernel.cpp",
        "gen9_pooling_kernel.cpp",
        "gen9_softmax_kernel.cpp",
        "gen9_wino_conv_fwd_data_2x3_kernel.cpp",
        "gen9_wino_conv_fwd_data_6x3_kernel.cpp",
        "ref_binary_kernel.cpp",
        "ref_bnorm_kernel.cpp",
        "ref_convolution_kernel.cpp",
        "ref_deconv_backward_bias_kernel.cpp",
        "ref_eltwise_kernel.cpp",
        "ref_gemm_kernel.cpp",
        "ref_inner_product_kernel.cpp",
        "ref_layer_normalization_kernel.cpp",
        "ref_lrn_kernel.cpp",
        "ref_matmul_kernel.cpp",
        "ref_pooling_kernel.cpp",
        "ref_reduction_kernel.cpp",
        "ref_resampling_kernel.cpp",
        "ref_rnn_kernel.cpp",
        "ref_shuffle_kernel.cpp",
        "ref_softmax_kernel.cpp",
        "ref_zero_pad_kernel.cpp",
        "rnn_reorder_kernel.cpp",
        "simple_concat_kernel.cpp",
        "simple_reorder_kernel.cpp",
        "simple_sum_kernel.cpp",
    ],
    cmd = "PROJECT_SOURCE_DIR=`pwd` && " +
          "DNNL_SOURCE_DIR=$$PROJECT_SOURCE_DIR/external/onednn_gpu && " +
          "KER_GEN_DIR=kernel_gen_dir && " +
          "cd $$DNNL_SOURCE_DIR/src/gpu/ocl && " +
          "mkdir -p $$KER_GEN_DIR &&" +
          "KER_GEN_DIR=$$DNNL_SOURCE_DIR/src/gpu/ocl/$$KER_GEN_DIR &&" +
          "KER_INC_DIR=$$DNNL_SOURCE_DIR/src  && " +
          "for file in `find -iname \"*.cl\"`; " +
          "do KER_FILE=$$file; DCPP_FILE=`echo $${file##*/}`; DCPP_FILE=$${DCPP_FILE%%.cl}_kernel.cpp; " +
          "cmake -DKER_FILE=$$KER_FILE -DGEN_FILE=$$DCPP_FILE -DKER_INC_DIR=$$KER_INC_DIR -P $$DNNL_SOURCE_DIR/cmake/gen_gpu_kernel.cmake; " +
          "mv $$DCPP_FILE $$KER_GEN_DIR; " +
          "done && " +
          "cd $$KER_GEN_DIR && " +
          "output_path=$$PROJECT_SOURCE_DIR/bazel-out/k8-opt/bin/external/onednn_gpu/ && " +
          "[[ -d $$output_path ]] && cp -f *_kernel.cpp $$output_path; " +
          "output_path=$$PROJECT_SOURCE_DIR/bazel-out/host/external/onednn_gpu/ && " +
          "[[ -d $$output_path ]] && cp -f *_kernel.cpp $$output_path; " +
          "output_path=$$PROJECT_SOURCE_DIR/execroot/org_tensorflow/bazel-out/k8-dbg/bin/external/onednn_gpu/ && " +
          "[[ -d $$output_path ]] && cp -f *_kernel.cpp $$output_path; " +
          "rm -rf $$KER_GEN_DIR",
)

cc_library(
    name = "onednn_gpu",
    srcs = glob(
        [
            "src/*/*.cpp",
            "src/*/*.hpp",
            "src/*/*/*.cpp",
            "src/*/*/*.h",
            "src/*/*/*/*.cpp",
            "src/*/*/*/*.h",
            "src/*/*/*/*/*.cpp",
            "src/*/*/*/*/*.c",
            "src/*/*/*/*/*.h",
        ],
        exclude = [
            "src/cpu/aarch64/*",
            "src/gpu/nvidia/*",
        ],
    ) + [
        ":dnnl_version_h",
        ":dnnl_config_h",
        ":kernel_list_cpp",
        ":onednn_genrule",
    ],
    hdrs = glob([
        "include/*",
        "include/oneapi/dnnl/*",
    ]),
    copts = [
        "-fexceptions",
        "-DDNNL_ENABLE_PRIMITIVE_CACHE",
        #TODO(quintin): for symbol collision, may be removed in produce version
        "-fvisibility=hidden",
    ],
    includes = [
        "include",
        "include/oneapi",
        "include/oneapi/dnnl",
        "src",
        "src/common",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/xbyak",
        "src/ocl",
        "src/sycl",
    ],
    #nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = ["@local_config_dpcpp//dpcpp:dpcpp_headers"],
)
