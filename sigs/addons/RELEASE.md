# SIG Addons Releases

SIG Addons release process consists of the folowing steps:
1. Create new rX.X branch on tensorflow/addons
2. Create and merge a new PR into the release branch
	* Set the correct version and suffix in [version.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/version.py)
	* Ensure the proper minimum and maximum tested versions of TF are set in [version.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/version.py)
	* Ensure the proper minimum and maximum ABI compatibility versions are set in [resource_loader.py](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/resource_loader.py)
	* Ensure proper supported python libraries are set in [setup.py](https://github.com/tensorflow/addons/blob/master/setup.py)
3. Create and merge a new PR which [sets the build docker image](https://github.com/tensorflow/addons/blob/master/tools/docker/build_wheel.Dockerfile#L3) to match the tensorflow dependency
	* We keep the "latest" image on nightly, but for building old TFA versions we should pin to the proper container for that release.
4. Publish and tag a [release on Github](https://github.com/tensorflow/addons/releases)
    * Add updates for new features, enhancements, bug fixes
    * Add contributors using `git shortlog <last-version>..HEAD -s`
    * **NOTE: This will trigger a GitHub action to release the wheels on PyPi**


## SIG Addons Release Team

Current Release Team:
- Sean Morgan - GitHub: [@seanpmorgan](https://github.com/seanpmorgan) - PyPI: [seanmorgan](https://pypi.org/user/seanmorgan/)
- Yan Facai(颜发才) - GitHub: [@facaiy](https://github.com/facaiy) - PyPI: [facaiy](https://pypi.org/user/facaiy/)
 
