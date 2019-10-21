# Admin guide for RFCs

## Overview

This document describes the process for community managers administering
TensorFlow RFCs.

|Author                        |Edd Wilder-James [@ewilderj](https://github.com/ewilderj)    |
:------------------------------|:-----------------------------|
|Last updated                  |2019-10-21                    |

## RFC Submission Process

### 1. PR is submitted to `tensorflow/community`

When a PR is submitted containing an RFC proposal, check for basic
formatting concerns.

* The filename should be `rfcs/YYYYMMDD-my-rfc.md` - where YYYYMMDD is the
  date, and hyphens connect any naming parts. No underscores. No uppercase
  letters.

* The header block of the RFC should be filled in properly, including the
  status field set to "Proposed"
  
### 2. Conform the RFC title

* In GitHub ensure the PR title is `RFC: The RFC's Title`. Check past PRs to
  see how they're all consistent.

### 3. Edit the PR description

The description (the first comment on the PR) of every RFC should look the
  same. They should contain, in order:
  
  * When the public review period closes. This is at least two weeks from the
date of publication.

  * The header table from the RFC showing author, sponsor, date.

  * A summary of what the RFC is about

Here's an example:

<blockquote>

*Comment period is open until 2019-08-28*

# Kernel and Op Implementation and Registration API

| Status        | Proposed                                             |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | James Ring (sjr@google.com), Anna Revinskaya (annarev@google.com) |
| **Sponsor**   | Günhan Gülsoy (gunan@google.com)                                  |
| **Updated**   | 2019-08-14                                                        |

## Objective

Tensorflow (TF) currently provides a C++ API for implementing kernels and ops. The Voltron project aims to create a modular/plugin-based TF implementation with API and ABI surfaces. Plugins will be able to create and register custom kernel and op implementations.

In order to provide a stable ABI, the Voltron team has chosen to provide C APIs to plugin authors. This document introduces the C API for op and kernel registration. For authors who wish to continue using C++ to interface with TensorFlow, an ABI-stable C++ header-only API is provided.
</blockquote>

### 4. Apply labels

* Apply the `RFC: Proposed` label, and any other appropriate label for the
  particular area of TensorFlow concerned, e.g. `TFX`.
  
### 5. Add the PR to the `RFC Management` project

### 6. In the `RFC Management` project, move from "Needs attention" to "Under review".
   
### 7. Publicize the RFC to `developers@tensorflow.org` and any other community-relevant mailing lists
   
Here's a template announcement. Check out the [many examples](https://groups.google.com/a/tensorflow.org/g/developers/search?q=RFC).

<blockquote>
To: developers@tensorflow.org<br>
Subject: [RFC] ACME TensorFlow API<br>
<br>
Hi folks,<br><br>

I'm pleased to announce the publication of a new TensorFlow RFC,
[ACME TensorFlow API](https://github.com/tensorflow/community/pull/162).

The comment period for this RFC is open through YYYY-MM-DD. Comments are
invited to the [pull request
linked](https://github.com/tensorflow/community/pull/162). You can view the
design doc there and also leave your comments inline on the 
[document source](https://github.com/tensorflow/community/pull/162/files).

**Summary**

The TensorFlow ACME API allows usage of all vintage cartoon characters
in an agent-based simulation. Wile E Coyote and the Road Runner are 
default personas, but we also propose the addition of Yosemite Sam
and Bugs Bunny.


Thanks in advance for your feedback!
</blockquote>


## RFC Acceptance Process

When an RFC's comment period is over, a review meeting is usually held.
(There may be occasions when one is not needed, consult with the RFC author).
It is the responsibility of the author or sponsor to post the notes from
that review into a comment on the PR, but you may need to remind them to do
this.

You can move the RFC into the "Awaiting notes" part of the `RFC Management`
project to help keep track.

**If the RFC is accepted**, ask the proposer to submit a final update, changing
the status to Accepted, and adding the RFC number into the header, per
the template (an RFC's number is the same as the PR number GitHub assigned it.)

Just occasionally you might have to do this yourself: you can edit the
Markdown in the PR yourself, as a code owner for the repository.

You can then:

* Remove the `RFC: Proposed` label and add the `RFC: Accepted` one
* Approve and merge the PR.

This should automatically move it to `Accepted PRs` in the `RFC Management`
project.

**Other possible end-states**

* If revisions are required, note that in the PR comment, keep the PR open but
  move it to `In Revision` in the `RFC Management` project.
  
* If the RFC is abandoned, note that in the comments, close the PR, and move
  it to the `Not progressed` column in the `RFC Management` project.
  
  
