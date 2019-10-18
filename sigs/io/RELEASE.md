# SIG IO Releases

At the moment SIG IO Releases consist of three parts:
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
- Release PyPI Python Package (see below)
- Release CRAN R Package (see below)
- Create a new version through GitHub

## PyPI Python Package Release

At the moment Python package (whl files) is created automatically,
upon each successful Travis CI on master branch. At the end of
each Travis CI build on master branch, all whl files
(2.7, 3.4, 3.5, 3.6, 3.7 on Linux and 2.7 on macOS) are pushed to
Dropbox and are available in:

https://www.dropbox.com/sh/dg0npidir5v1xki/AACor-91kbJh1ScqAdYpxdEca?dl=0

To perform a release in PyPI, first make sure the binary whl files
are the correct one from corresponding Travis CI build number.
This could be verified by checking the Travis CI history where at
the end of the log, the sha256 of all whl files are calculated and displayed.
The sha256 of each file displayed on Travis CI log should match the sha256
of the files downloaded from Dropbox.

Once sha256 are verified against every whl files on Dropbox, perform
a sanity check, then upload all of the whl files
(2.7, 3.4, 3.5, 3.6, 3.7 on Linux and 2.7 on macOS) to PyPI.org:

```
twine upload *.whl
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
- Bryan Cutler - GitHub: [@BryanCutler](https://github.com/BryanCutler) - PyPI: [cutlerb](https://pypi.org/user/cutlerb)
- Aleksey Vlasenko - GitHub: [@vlasenkoalexey](https://github.com/vlasenkoalexey) - PyPI: [vlasenkoalexey](https://pypi.org/user/vlasenkoalexey)
