# SIG IO Releases

At the moment SIG IO Releases consist of two parts:
- Release of source code with versioning in GitHub
- Release of python package in PyPI
- Release of R package to CRAN

## GitHub Source Code Release

To perform a release in GitHub, the following steps are needed:
- Create a PR to update the RELEASE.md in
  [github.com/tensorflow/io](https://github.com/tensorflow/io)
  * Add updates for new features, enhancements, bug fixes
  * Add contributors using `git shortlog <last-version>..HEAD -s`
- Merge the PR for RELEASE.md update
- Create a new version through GitHub

## PyPI Python Package Release

To perform a release in PyPI, first complete the above GitHub release, then
build pip packages locally with docker in the following commands
```
$ docker run -it -v ${PWD}:/working_dir -w /working_dir \
    tensorflow/tensorflow:custom-op bash -x /working_dir/release.sh <2.7|3.4|3.5|3.6>
```
Note the above commands has to run four times with 2.7, 3.4, 3.5, 3.6
to generate all pip packages for different python versions.

Then upload `artifacts/*.whl` files with:
```
twine upload artifacts/*
```

## CRAN R Package Release

Before submitting the R package to CRAN, manually perform and check the following items:
* Make sure the documentation in `README.md` and `vignettes` is up-to-date
* Update `Version` field in `DESCRIPTION` file
* Update `NEWS.md` to include items for this new release
* Run `devtools::check()` and fix all the notable issues, especially warnings and errors
* Update `cran-comments.md` to include any unsolvable issues from `devtools::check()` and
other comments/responses to CRAN maintainers
* Run checks on R-hub via `devtools::check_rhub()` and on win-builder via `devtools::check_win_devel()`. This is
optional since Python is not be installed on CRAN test machines and we skip the tests on
CRAN.

To submit the package to CRAN for review, do the following:
* Run `devtools::release()` to submit for review. Here's how it looks like if submission is successful:
```
Submitting file: /var/folders/zp/k98_wphd0h9c5b3zyk5xhnhm0000gn/T//RtmpHh9Wdo/tfio_0.1.0.tar.gz
File size: 483.4 Kb
Uploading package & comments
Confirming submission
Package submission successful.
Check your email for confirmation link.
```
* Check email for confirmation link and confirm the submission
* CRAN maintainers will review the submission and email you for the result of this submission.
If there are any additional issues and comments that need to be addressed, address them and re-submit

## SIG IO Release Team

Everybody with an interest in helping SIG IO releases, is welcome
to join the Release Team. To participate, create a PR to update
the doc or send an email to SIG IO mailing list
[io@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/io).
Please provide both GitHub and PyPI handle to join the release team.

Current Release Team:
- Yong Tang - GitHub: [@yongtang](https://github.com/yongtang) - PyPI: [yongtang](https://pypi.org/user/yongtang)
- Anthony Dmitriev - GitHub: [@dmitrievanthony](https://github.com/dmitrievanthony) - PyPI: [dmitrievanthony](https://pypi.org/user/dmitrievanthony)
- Yuan (Terry) Tang - GitHub: [@terrytangyuan](https://github.com/terrytangyuan) - PyPI: [terrytangyuan](https://pypi.org/user/terrytangyuan)
