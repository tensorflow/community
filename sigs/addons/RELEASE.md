# SIG Addons Releases

SIG Addons release process consists of the folowing steps:
1. Create new rX.X branch on tensorflow/addons
2. Create and merge a new PR into the release branch
	* Set the correct version and suffix in [version.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/version.py)
	* Freeze the tensorflow version in
      - [setup.py](https://github.com/tensorflow/addons/blob/master/setup.py)
      - [requirements.txt](https://github.com/tensorflow/addons/blob/master/build_deps/requirements.txt)
	* Remove `--nightly` flag from [release scripts](https://github.com/tensorflow/addons/tree/master/tools/ci_build/builds)
	* Compile the docs: [instructions](https://github.com/tensorflow/addons/tree/master/tools/docs)
3. Trigger [Travis build](https://travis-ci.org/tensorflow/addons)
    * This will test and build linux+macos wheels and publish to PyPi
4. Publish and tag a [release on Github](https://github.com/tensorflow/addons/releases)
    * Add updates for new features, enhancements, bug fixes
    * Add contributors using `git shortlog <last-version>..HEAD -s`


## SIG Addons Release Team

Current Release Team:
- Sean Morgan - GitHub: [@seanpmorgan](https://github.com/seanpmorgan) - PyPI: [seanmorgan](https://pypi.org/user/seanmorgan/)
- Yan Facai(颜发才) - GitHub: [@facaiy](https://github.com/facaiy) - PyPI: [facaiy](https://pypi.org/user/facaiy/)
