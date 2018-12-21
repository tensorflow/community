# TensorFlow Dockerfile Assembler

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Austin Anderson (angerson@google.com) |
| **Sponsor**   | Gunhan Gulsoy (gunan@google.com)                     |
| **Updated**   | 2018-08-23                                           |


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

**Also Important:** This design is not currently attempting to revise the images for speed or size: the design sets out a process that makes optimizing the images much easier to do on a larger scale.

# Background

TensorFlow's Docker offerings have lots of problems that affect both users and
developers. [Our images](https://hub.docker.com/r/tensorflow/tensorflow/) are
not particularly well defined or documented, and [our
Dockerfiles](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
are complicated and frightening.

## Existing Images are Hard to Optimize

TensorFlow's current set of Dockerfiles are difficult to optimize. Developers
dislike pulling enormous Docker containers, and many of our tags could be
considered clunky (tag sizes yanked from Dockerhub, see also @flx42's comment
on this doc's PR for on-disk sizes):

| Image Tag          |   Size |
|:-------------------|-------:|
|latest-devel-gpu-py3| 1 GB   |
|latest-devel-py3    | 773 MB |
|latest-gpu-py3      |1 GB    |
|latest-py3          | 438 MB |
|latest-devel-gpu    | 1 GB   |
|latest-devel        | 727 MB |
|latest-gpu          | 1 GB   |
|latest              | 431 MB |

Including an extra dependency like Jupyter and convenience packagesnot can add
a few hundred megabytes of extra storage. Since some developers want to have
Jupyter in the images and it's too much trouble for us to maintain many similar
Dockerfiles, we've ended up with a limited set of non-optimized images. I'm not
sure if this truly a critical problem, but it's a little annoying (one of my
personal computers only has 32 GB of SSD space on the root drive, and I
regularly need to wipe my docker cache of large images).

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

**Important**: This design in its current implementation does **not** attempt
to address the limitations of our current set of images. Instead, it replicates
the current set of tags with a few easy improvements, the most notable being a
separate set of Dockerfiles that add Jupyter -- identical in every way to the
non-Jupyter images without needing any extra maintenance. This design makes it
much easier to craft TensorFlow's Docker offering in a way that satisfies
everyone with minimal extra work from the Dockerfile maintainers.

# Impact

This approach has many convenient benefits:

*   The result is concrete, buildable, documented Dockerfiles. Users who wish
to build their own images locally do not need to also understand the build
system. Furthermore, basing our images on clean Dockerfiles that live in the repository feels clean and right -- as a user, I (personally) like to be able to see how an image works. It removes the mystery and magic from the process.
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

## Hacky Multi-Stage Dockerfile

"Multi-stage Building" is a powerful new Dockerfile feature that supports
multiple FROM statements in one Dockerfile. Multi-stage builds let you build
and run an artifact (like a compiled version of a binary) in any number of
separate stages designated by FROM directives; the resulting image is only as
large as the final stage without the build-only dependencies from previous
stages.

However, Docker's ARG parameter expansion can be used in these extra FROM
directives to conditionally set base images for each build stage:


```dockerfile
# If --build-arg FROM_FOO is set, build FROM foo. else build FROM bar.
ARG FROM_FOO
ARG _HELPER=${FROM_FOO:+foo}
ARG BASE_IMAGE=${_HELPER:-bar}
FROM ${BASE_IMAGE}
…
```

This means that it's possible to use multi-stage builds and ARGs to create
stages that are conditionally based on previous stages in the Dockerfile.
[This sample
Dockerfile](https://gist.github.com/angersson/3d2b5ae6a01de4064b1c3fe7a56e3821),
which I've included only as a demonstration of a bad idea (and may currently
work), is very powerful but not extensible and not easy to understand. It is
heavily coupled to our current environment, which may change immensely e.g. if
AMD releases Docker images similar to Nvidia's or if someone would like to add
MKL support.

## Multiple Normal Dockerfiles Aggregated into Multiple Stages

In a [comment on this doc's PR](https://github.com/tensorflow/community/pull/8#issuecomment-410080344), @flx42 suggested a much-improved version of the
previous section. Another way of using ARG interpolation in FROM lines would be
to write multiple isolated Dockerfiles that can be layered together during the `docker build` process:


```
ARG from
FROM ${from}

ARG PIP
RUN ${PIP} install jupyter
```

And then:

```
$ docker build -t nvidia-devel -f Dockerfile.nvidia-devel .
$ docker build -t nvidia-devel-jupyter-py3 --build-arg from=nvidia-devel --build-arg pip=pip3 -f Dockerfile.jupyter .
```

This shares the advantage of the current design by working from many reusable parts, but carries some notable tradeoffs:

### Advantages over Current Design

I can see a variety of minor improvements:

- No need for assembler script or spec file
- Possibly faster build times due to concretely isolated image stages
- Image stages (akin to partials) may be more reusable due to slot-like usage of `--build-args`
- Because there are no concrete Dockerfiles, there's only one place that defines the Dockerhub tags and what components describe them (in the current design, the spec files describes the Dockerfiles, and then a little more logic elsewhere in our CI would configure those Dockerfiles with the tags)

### Downsides compared to Current Design

...but some downsides that I think are fairly heavy:

- Spec + Assembler have some very nice advantages (validation, re-use, etc.)
- No concrete Dockerfiles for OSS devs to use / refer to
- Advanced usage requires some unintuitive file/directory layout + build ordering
- Image-building complexity offloaded to OSS developers and to the CI scripts, which would need scripts / logic to define sets of images to build
- Updating requires familiarity with multi-stage behavior

### Conclusion

This is an interesting approach that I like a lot, but I don't think it offers
enough benefits over the current design (which has another advantage in that it
is already mostly finished) to implement.

It's worth noting that using multiple FROM stages is a powerful tool that could
possibly be leveraged in the partials for the current design.

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

* Argument passing becomes slightly more complex, because ARGs must be passed
and read as either ENV variables or as build arguments.
* Each dockerfile has to be properly documented manually, if at all.
* Developers have to leave the Dockerfile to read the shell scripts, which
gets annoying.
* Maintenance is spread across the dockerfiles and the scripts, and can grow
into even more work (like some Dockerfiles having extra non-script directives,
etc.).
* Extra overhead in the scripts can be kind of wasteful 

# Work Estimates

I have already completed a PR that will introduce these Dockerfiles without
affecting our current builds. These would probably take a week or two to
migrate.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
