# TensorFlow Dockerfile Assembler

| Status        | Proposed       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Austin Anderson (angerson@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                     |
| **Updated**   | 2018-07-31                                           |


# Summary

This document describes a new way to manage TensorFlow's dockerfiles. Instead
of handling complexity via an on-demand build script, Dockerfile maintainers
manage re-usable chunks called partials which are assembled into documented,
standard, committed-to-repo Dockerfiles that don't need extra scripts to build.
It is also decoupled from the system that builds and uploads the Docker images,
which can be safely handled by separate CI scripts.

**Important:** This document is slim. The real meat of the design has already
been implemented in [this PR to
tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/pull/21291).

# Background

TensorFlow's Docker offerings have lots of problems that affect both users and
developers. [Our images](https://hub.docker.com/r/tensorflow/tensorflow/) are
not particularly well defined or documented, and [our
Dockerfiles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
are complicated and frightening.

## TF Docker Images need Complexity

Our Docker images support two primary use cases: development _with_ TensorFlow,
and development _on_ TensorFlow. We want a matrix of options available for both
of these types of users, the most critical being GPU development support
(currently nvidia-only) and pre-installed Jupyter support. With only those
options considered, we target eight very similar Docker images; sixteen with
Python versioning.

Our current images come from a script called `parameterized_docker_build.sh`,
which live-edits a templated Dockerfile with `sed` to insert new Dockerfile
commands. The script has a poor reputation because it's hard to understand, can
be finicky, and is not easily understood compared to vanilla Dockerfiles. Some
Dockerfiles are duplicated, some are unused, and some users have made their own
instead. None of the Dockerfiles use the ARG directive.

Furthermore, `parameterized_docker_build.sh` is tightly coupled with the
deploy-to-image-hub process we use, which is confusing because users who build
the images locally don't need that information at all.

This document proposes a new way for the TF team to maintain this complex set
of similar Dockerfiles.

# Design

We use a generator to assemble multiple partial Dockerfiles into concrete
Dockerfiles that get committed into source control. These Dockerfiles are fully
documented and support argument customization. Unlike the parameterized image
builder script, this system excludes the image deployment steps, which should
be handled by a totally different system anyway.

This section lightly describes the design, which is fully implemented in [this
pull request to the main TensorFlow
repo](https://github.com/tensorflow/tensorflow/pull/21291).

Partial files are syntactically valid but incomplete files with Dockerfile
syntax.

Assembly is controlled by a specification file, defined in yaml. The spec
defines the partials, the ARGs they use, the list of Dockerfiles to generate
based on ordered lists of partials, and documentation for those values.

The assembler is a python script that accepts a spec and generates a bunch of
Dockerfiles to be committed. The spec includes documentation and descriptions,
and the output Dockerfiles are fully documented and can be built manually.

# Impact

This approach has many convenient benefits:

*   The result is concrete, buildable, documented Dockerfiles. Users who wish
to build their own images locally do not need to also understand the build
system.
*   This implementation is agnostic to what images we would like to make
available online (i.e. our Docker story). It's very easy to add new dockerfile
outputs.
*   The build-test-and-deploy-images process is decoupled from the Dockerfile
generation process.
*   Control of the set of dockerfiles is centralized to the spec file, instead
of being spread across each Dockerfile.
*   The spec can be extended to add more conveniences. My implementation, for
example, already includes de-duplication of many similar Dockerfile
specifications.
*   All dockerfiles are consistently documented.
*   Common pieces of code, like a slick shell environment or a Jupyter
interface, can be updated in batch by updating a single partial file.
*   The spec can also be used in the image building process, e.g. to read all
available args.

# Caveats and Rejected Alternatives

I considered two alternatives while working on this.

## Modern Multi-Stage Dockerfile

"Multi-stage Building" is a powerful new Dockerfile feature that supports
multiple FROM statements in one Dockerfile. It is meant to be used for creating
artifacts with one image before using those artifacts in another image, but you
can also use variable expansion in any FROM line:


```dockerfile
# If --build-arg FROM_FOO is set, build from foo. else build from bar.
ARG FROM_FOO
ARG _HELPER=${FROM_FOO:+foo}
ARG BASE_IMAGE=${_HELPER:-bar}
FROM ${BASE_IMAGE}
…
```

...which means that you can dynamically set multiple FROM images. My first
draft used ARGs and FROMs in a single Dockerfile to manipulate build stages.
[The resulting
Dockerfile](https://gist.github.com/angersson/3d2b5ae6a01de4064b1c3fe7a56e3821)
is incredibly powerful, obscenely difficult to understand, and absolutely not
extensible: it is heavily coupled to our current environment, which may change
immensely e.g. if AMD releases Docker images similar to Nvidia's.

## Manually Maintained Dockerfiles with Script References

Another pattern that supports complicated Dockerfiles is to manually maintain
many Dockerfiles that each call out to a common set of build scripts:

```dockerfile
FROM ubuntu
COPY install_scripts/ /bin
RUN /bin/install_nvidia_dev.sh
RUN /bin/install_python_dev.sh
RUN /bin/install_bazel.sh
...
```

This is better than our current approach, but has many small drawbacks that add
up:

*   Argument passing becomes slightly more complex, because ARGs must be passed
and read as either ENV variables or as build arguments.
*   Each dockerfile has to be properly documented manually, if at all.
*   Developers have to leave the Dockerfile to read the shell scripts, which
gets annoying.
*   Maintenance is spread across the dockerfiles and the scripts, and can grow
into even more work (like some Dockerfiles having extra non-script directives,
etc.).
*   Extra overhead in the scripts can be kind of wasteful 

# Work Estimates

I have already completed a PR that will introduce these Dockerfiles without
affecting our current builds. These would probably take a week or two to
migrate.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
