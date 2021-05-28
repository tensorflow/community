# Platform-specific build configurations.

load("@com_google_protobuf//:protobuf.bzl", "proto_gen")
load("//tensorflow_plugin:workspace.bzl", "clean_dep")
load("@rules_cc//cc:defs.bzl", "cc_library")

def cc_proto(name, src, deps = []):
    native.genrule(
        name = "%s_cc" % name,
        outs = ["%s.pb.cc" % name, "%s.pb.h" % name],
        cmd = "echo $(GENDIR); which $(location @com_google_protobuf//:protoc); $(location @com_google_protobuf//:protoc) --cpp_out=$(GENDIR) $<",
        srcs = [src],
        tools = ["@com_google_protobuf//:protoc"],
    )
    native.cc_library(
        name = "%s_proto" % name,
        srcs = ["%s.pb.cc" % name],
        hdrs = ["%s.pb.h" % name],
        deps = [                                                                                                                    
            "@com_google_protobuf//:protobuf_headers",
            "@com_google_protobuf//:protobuf",
        ] + deps,
        copts = ["-I$(GENDIR)"],
    )

def if_static(extra_deps = [], otherwise = []):
    return otherwise

def tf_protobuf_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )
