# `tf-nightly` and `tf-nightly-gpu` renovations

| Status        | Implemented                                             |
| :------------ | :------------------------------------------------------ |
| **Author**    | amitpatankar@google.com 								  |
| **Sponsor**   | gunan@google.com                                        |
| **Updated**   | 2018-10-26                                              |

## Objective

Plan a new process and protocol for how we build and distribute nightlies. `tf-nightly` is now widely used to test and evaluate performance as official releases are spread more far apart. If we are following `tf-estimator`'s process for individual module testing we need to make `tf-nightly` more reliable and robust.

## Motivation

Earlier we built from HEAD from the master branch every night for all operating systems. We take the builds for Ubuntu and use them to create our docker containers so that way the git hash matches. Breakages were quite common, and most of our nightly builds were quite behind on all platforms. For example, there was a three month stretch where Windows was not updated.

## Design Proposal

We will take the latest postsubmit build that has passed for each platform and get the commit hash. Based on that hash we will create nightly binaries. If the last postsubmit green is more than 24 hours old we will not publish binaries for that platform for that day.

Absolutely no tests will be run on the binaries. If it builds it ships. Refer to the diagram below for different cases.

![](https://storage.googleapis.com/amitpatankar/tf-nightly-postsubmit.png)


## Detailed Design

### Support
* We will still continue to offer packages for:

|Platform/OS:   |CPU   |GPU   |Package Type             |
|---------------|------|------|-------------------------|
|Mac            |Yes   |No    |Pip `Python 2.7-3.6`     |
|Ubuntu         |Yes   |Yes   |Pip `Python 2.7-3.6`     |
|Windows        |Yes   |Yes   |Pip `Python 2.7-3.6`     |
|Docker-dev     |Yes   |Yes   |Container `Python 2&3`   |
|Docker-nondev  |Yes   |Yes   |Container `Python 2&3`   |
* Please file bugs on [GitHub](https://github.com/tensorflow/tensorflow/issues) if a nightly build for a certain platform has not been pushed for a week. We will do our best to push builds every night, but please wait 7 days before notifying us.
* We will also be much less active for Windows builds especially GPU. We often find that those are difficult to fix, most of `tf-nightly` users are Ubuntu and Docker. That grace period before you can notify us for Windows GPU will be two weeks.


### Versioning
![](https://storage.googleapis.com/amitpatankar/tf-rename-release-diagram.png)

## Questions and Discussion Topics

* Although the package names for `tf-nightly` and `tensorflow` differ, installing one after the other will overwrite some files in site-packages.
* Hashes may be mismatched. The binary for a certain day on Windows can be a different hash from that corresponding binary on Ubuntu.
* Cannot name them something better due to [PEP440](https://www.python.org/dev/peps/pep-0440/) compliance.