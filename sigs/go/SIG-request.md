# Request for SIG

## What is this group for?

The purpose of this special interest group is to foster and facilitate
community collaboration around the TensorFlow Go bindings.  Community
collaboration is expected to result in improvements to the usability and
maintenance of the bindings.

Go is a statically-typed, compiled programming language with wide use,
especially in the development of API/RPC services, CLIs and web services and in
data processing [[1](https://blog.golang.org/survey2020-result)].  Go bindings
to the TensorFlow C API were first introduced in TensorFlow v0.11.0 and have
since resided within the main TensorFlow repository, at
[github.com/tensorflow/tensorflow/go](https://github.com/tensorflow/tensorflow/tree/r2.8/tensorflow/go).
These bindings are
[loosely integrated with the build system](https://github.com/tensorflow/tensorflow/pull/50934)
and are presently listed as being both
[unsupported by the community](https://www.tensorflow.org/versions) and
[unsupported by the TensorFlow team](https://github.com/tensorflow/build/tree/master/golang_install_guide).

Significant problems presently exist in the usability of the Go bindings.
Changes to how Go manages dependencies, specifically in the implementation of
[Go Modules](https://blog.golang.org/using-go-modules), have resulted in a set
of installation challenges (see, for instance, GitHub issues
[41808](https://github.com/tensorflow/tensorflow/issues/41808) and
[43847](https://github.com/tensorflow/tensorflow/issues/43847)). To alleviate
these issues, there has been discussion regarding relocation of the Go bindings to
a dedicated repository
([1](https://github.com/tensorflow/tensorflow/pull/44655#issuecomment-725040183),
[2](https://github.com/tensorflow/tensorflow/pull/50934)).
A component of this SIG proposal is a request for TensorFlow team support and
collaboration in migrating the TensorFlow Go bindings to a dedicated GitHub
repository.

This request proposes that an opportunity exists to establish a effective SIG
around the Go bindings. In the past, the TensorFlow developer community has
made valuable contributions to the Go bindings, but community contributions
have been sporadic. Implicit in this request is an acknowledgment that the
current approach is unlikely to fully promote community collaboration due, for
instance, to collocation of the bindings with TensorFlow core, methods for
issue assignment/handling, the current state of the bindings in terms of
usability, and the recent shift of the bindings into unsupported status.  The
expected outcome of establishing SIG Go is to increase community collaboration
around the Go bindings, leading to the resolution of these issues and a pathway
for long-term maintenance and development activities.

## Who will be part of it?

* SIG Go leads:
  - 
  - William Muir [@wamuir](https://github.com/wamuir) - wamuir at gmail
* Anyone interested in discussing or contributing to the TensorFlow Go bindings is welcome.

## What initial problems will the group tackle?

The SIG proposes a charter focused on conservative aims: engaging the
community, solving current and future usability issues, performing on-going
maintenance, and making incremental improvements to the existing codebase
(e.g., functionality, tests, examples, documentation).

More specifically, the SIG's initial focus will be on the following set of
problems/objectives:

* Fostering community involvement for the project and membership in SIG Go.
* Migrating TensorFlow/Go source code from the main TensorFlow repository to
  a dedicated repository.
* Restoring usability of the bindings, specifically achieving interoperability
  with Go tooling for module installation.
* Improving unit testing and the coverage of tests.
* Improving package documentation, including example code.
* Authoring and publishing tutorials and other instructional content.
* Assessing design and functionality of the package, identifying opportunities
  and developing a roadmap.

## What modes of communication do you intend to use?

* A dedicated mailing list backed by Google Groups.
* As needed, video conferencing on Google Hangouts.
* Other communication platforms, consistent with community preferences.

## Launch plan

1. Identification of any additional group leads and/or initial members.
2. Publication of the proposal and charter for community review and comments.
3. Notification of SIG establishment via TensorFlow general mailing lists
   (discuss@, developers ML).
4. Addition of SIG Go to the community pages on
   [tensorflow.org](https://tensorflow.org).
5. Creation of a SIG mailing list and development of list discussions about
   initial work items.
6. Establishment of [tensorflow.org/go](https://tensorflow.org/go), via
   [TensorFlow Documentation](https://github.com/tensorflow/docs) pages, for
   use as a [module import path](https://golang.org/ref/mod#vcs-find).
7. Collaboration with the TensorFlow team on establishing a dedicated 
   GitHub repository for the Go bindings and on migrating the bindings
   from the main TensorFlow repository.
8. Creation of a blog post for the TensorFlow Medium.com blog community.

# Charter

Here's the link to the [group charter](CHARTER.md).
