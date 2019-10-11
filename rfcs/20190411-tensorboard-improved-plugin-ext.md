# TensorBoard: Improved Plugin Extensibility

| Status        | Accepted       |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Stephan Lee (Google), William Chargin (Google) |
| **Sponsor**   | Mani Varadarajan (Google)                 |
| **Updated**   | 2019-05-16                                           |

## Overview

This RFC proposes a friction-free approach to enabling TensorBoard contributors to create and share TensorBoard plugins as installable extensions.

TensorBoard is composed of various plugins that show up as separate dashboards to users (e.g., scalars, graphs, hparams, histograms, etc.). Today. anyone can contribute a new plugin by forking TensorBoard, adding their plugin, and getting the PR merged. However, this can require a lengthy review to make sure it is architected correctly and is aligned with TensorBoard’s goals.

This RFC covers two changes that will make this process easier:
1. Plugins can be distributed as Python packages. Users can install them as extensions to TensorBoard.
2. Plugins are no longer limited to use Polymer but can instead use any frontend library or framework (e.g., React)

The intention is to enable any prospective contributor to easily create or reuse a visualization with tools familiar to them, and enable users to try it out quickly. Plugins that are useful to many users should be able to be packaged as a "first party" plugin, but this removes the friction needed to start getting feedback or trying out something new.

## Motivation

Making the end-to-end plugin development process easier will have the following benefits:
- Contributors can develop new plugins more quickly/easily as they can use the tools that are familiar to them, while maintaining a consistent user experience within TensorBoard
- Contributors can test out their plugins without being gated on adding the plugin directly into TensorBoard. This is useful for tools that aren’t necessarily widely used or for quick testing with early users.

## Status quo
TensorBoard builds JavaScript and TypeScript using a process called Vulcanization -- the process can be described as a language transpilation and a bundler that optionally create a single HTML file with all JS, HTML, and CSS dependencies inlined. All plugin code today lives in the TensorBoard repository and naturally gets Vulcanized, and eventually gets bundled into a TensorBoard pip package and distributed with Python code. When launched, TensorBoard starts a web server that serves the Vulcanized HTML file.

The TensorBoard frontend largely uses Polymer components (while more advanced components use D3, canvas, and WebGL; we do not have technical limitations in supporting a library like React today). TensorBoard repository comes with many shareable components (in `tf_dashboard_common`) that are essential building blocks most plugins and they promote consistency in UI and UX.

## Design Proposal
### Plugin distribution
A plugin will be distributed as a single Pip distribution package that includes both a Python backend and a web frontend.

The Python backend must export an object conforming to a standard interface that describes the plugin’s metadata and provides an option to load the plugin’s implementation. The exact shape is not yet specified, but it will be conceptually similar to a function returning a tuple containing a name, a description, and a [`tensorboard.plugins.base_plugin.TBLoader`][tb-loader] object.

In order to be discovered by TensorBoard, the distribution should include an [`entry_points` entry][entry-points] declaring itself as a TensorBoard plugin. The entry point group is `tensorboard_plugins`. The entry point name is arbitrary, and ignored. The entry point object reference should refer to the standard interface described in the previous paragraph. As an example, the setup.py file for a hypothetical “widgets” plugin might look like:

```python
import setuptools
setuptools.setup(
    name="tensorboard_plugin_widgets",
    description="Widget chart support for TensorBoard",
    packages=setuptools.find_packages(),
    # more metadata (version, license, authors…)
    entry_points={
        "tensorboard_plugins": [
            "widgets = tensorboard_plugin_widgets.plugin:get_plugin",
        ],
    },
)
```

assuming that the `tensorboard_plugin_widgets.plugin` module exposes a function `get_plugin` conforming to the interface described above.

To facilitate plugin development without requiring plugins to be packaged into a wheel and installed into a virtualenv, TensorBoard will learn an `--extra_plugins` flag, whose value will be a list of comma-separated object references. For instance, for testing the widgets plugin, one might invoke

```
tensorboard --logdir /tmp/whatever \
    --extra_plugins tensorboard_plugin_widgets.plugin:get_plugin
```

with the `tensorboard_plugin_widgets.plugin` module on the Python path.

[entry-points]: https://packaging.python.org/specifications/entry-points/
[tb-loader]: https://github.com/tensorflow/tensorboard/blob/dc71eeea403eb027612422ab6cd3061d4f9a10f3/tensorboard/plugins/base_plugin.py#L149-L162

### Frontend binary loading
One of our soft goals is to enable existing visualization suite to integrate with TensorBoard with ease without having to change, for instance, module system. Of course, a plugin has to procure data by writing a backend and some glue code to integrate with our backend, but we are hoping that most of the visualization part can remain untouched.

#### Requirement: Notebook Integration
TensorBoard recently added support for Jupyter and Colab and the feature is well received by the community. The new plugin frontend design should be compatible with the notebook integration.

Of two prominent notebooks, Colab imposes greater limitations to our designs due to its security model: running in google.com domain, Colab sandbox and isolate an output cell from the main process using various techniques. Colab, specifically, employs ServiceWorker to proxy network request originating from output cell to a server running on the kernel but this imposes restriction the iframe usage; network request to fetch the document of the iframe is not intercepted by the ServiceWorker.

#### Iframe based binary loading
When a plugin is activated by a user, TensorBoard will create an iframe and point the iframe to the endpoint that plugin defined as an entry point.

Iframe sandboxes JavaScript context and it gives plugin developers a lot of freedom in terms of frameworks -- with context disjoint from others, each plugin binary can incorporate as many dependencies as it would like to use without worrying about the correctness. This property is especially useful for Polymer-based applications because Polymer uses a global state, `CustomElementRegistry` (in v1, `document.registerElement`), and is not possible to load components with the same component name more than once. Loading plugins in its own separate frame will let plugin authors develop with ease and potentially have a standalone widgets for use outside of TensorBoard.

It must be clearly stated that the use of global state is not unique to Polymer. Although JavaScript bundles use Immediately Invoked Function Expression (IIFE) and use closure to encapsulate dependencies, a bundler like Browserify or Webpack _can_ uses global states when shimming CommonJS `window.require` and use non-unique identifier for a global module.

Despite the benefits and its simplicity, Colab makes this solution more complex -- because frames with different origins cannot be intercepted by Colab's ServiceWorker, we need some mechanism to route request to the non-publicly visible server running on the kernel. The TensorBoard team is working with Colab to introduce new Colab API that allows a port to be "exposed" to a stable URL (actual behavior is a lot more complex and is subject to changes) and the team is confident that it will make the iframe solution compatible with Colab.

### Frontend binary
Vulcanization today does not support CommonJS or packages from NPM. The team recognizes it to be an impediment for a plugin authorship and is unnatural for developing a frontend. TensorBoard team will provide a canonical build configuration that uses bazelbuild/rules_nodejs and Rollup but will not inhibit one from using Webpack and other bundlers.

The frontend binary will be a collection JavaScript modules that implements certain interface -- one for configuring the dashboard (registerDashboard) and another for render. With exceptions of Polymer1 and Polymer2, modern frameworks and libraries either use JavaScript or a JavaScript based template system to create a DOM or a WebComponent and, thus, the requirement should be easy to fulfill.

### Core module and UI/UX Consistency
One of the most important key concepts when using TensorBoard is the data selection. TensorBoard uses runs and tags to filter potentially a large set of event logs to visualize. Most plugins implemented this today and we expect third party plugins to do the same.

For ease and to promote maximal interaction and UI consistency, TensorBoard will provide a library that bundles Polymer components and encourage plugin authors to use the building blocks to build out a plugin. One of the components will be tf-runs-selector which will not only provide necessary controls but also tie with the TensorBoard core and make the selection consistent across all plugins. Note that the component can be used via Polymer or by using `document.createElement('tf-runs-selector')` if using other view libraries.

### Plugins Misc
#### Accessibility
N/A: Using an iframe does not change behavior screen reader
#### Instrumentation Considerations
A plugin may decide to put instrumentations and there is no way for TensorBoard to enforce/disallow that.
#### Docs and Tutorial
How can a plugin show a helpful manual without asking user to visit the repository to review the README? Can TensorBoard assist by showing HTMLfied README in the browser?

## Alternatives considered

#### Non-iframe based plugin loading
Generally speaking, there are largely two types of solutions for loading plugin frontend: one using iframe and one without using an iframe. Within the non-iframe based solution, there are several ways to achieve the goal, and they are depicted below.

One critical drawback of all non-iframe approaches is that TensorBoard cannot enforce plugin authors from inadvertently mutating the global object. For instance, if a library that a plugin transitively depends polyfills and monkey-patches a global object, it can influence how other plugins behave.

Within the non-iframe based solution, there are about two options to achieve load plugins. Two solutions share strengths and weaknesses and they are:

**Pro**
- easy to support
- supports Colab as long as plugins do not use iframes

**Con**
- global mutating libraries like Polymer app will not work
- can lead to an awkward UX: e.g., different version of a visualization library have different UI/UX

##### Option A: Require plugins to bundle all dependencies into one binary
Modern web binaries/bundles are often created with Webpack, Browserify, or Rollup and they often leverage techniques like vendoring or code-splitting to optimize above-fold-time. Though details vary by bundlers, it is hard to guarantee correctness depending on a shim from CommonJS: some shims use a global name as identifier for finding a module across bundles (e.g., 'react') instead of hashed unique name. This creates non-zero chance of collision and causes a plugin to load a different module than expected. As such, TensorBoard mandate plugins to bundle all transitive dependencies binary.

##### Option B: Default libraries
TensorBoard can add React to its front-end dependencies and attach the symbol to the global object, i.e., `window.React`, akin to d3, Plottable, and dagre in TensorBoard today. Plugin authors will assume presence when developing for TensorBoard. Otherwise simple, this solution comes with several drawbacks.

If React (or any library) makes a backwards incompatible change, TensorBoard upgrading it  will break plugins. This puts maintenance burden on plugin authors and, when not properly maintained, broken plugins will provide janky experience for users.

Another disadvantage of this solution is that TensorBoard will become a gatekeeper of dependencies. In a fast changing frontend ecosystem, a library can easily replace another. If, for example, Redux becomes obsolete in favor a new framework, a developer will have to raise an issue to TensorBoard and go through the process to get it added to the global.


## Questions and Discussion Topics
