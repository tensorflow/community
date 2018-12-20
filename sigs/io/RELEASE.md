# SIG IO Releases

At the moment SIG IO Releases consist of two parts:
- Release of source code with versioning in GitHub
- Release of python package in PyPI

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
