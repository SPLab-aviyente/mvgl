# mvgl

A tool to learn the topology of mutliview graphs from smooth graph signals.

## Installation
To install the package, first create a virtual environment with a Python version at least `3.10`. With `conda`, this can be done by: 
```bash
$ conda create -n mvgl_env -c conda-forge python=3.10
```
Then, use `pip` to install the package from Github. 
```bash
$ pip install git+https://github.com/SPLab-aviyente/mvgl.git
```

### Development

The development of this packege is done by `poetry`, which needs to be installed 
if one wants to work with the development version of the packege. To install 
poetry please see this [link](https://python-poetry.org/docs/) (installation 
with `pipx` is preferred). Once `poetry` is installed, clone the repo in an
appropriate directory and `cd` into it:
```bash
$ git clone https://github.com/SPLab-aviyente/mvgl.git
$ cd mvgl
```
Next, create a virtual environment. If you have `conda` installed, this can 
be done as follows:
```bash
$ conda create -n mvgl -c conda-forge python=3.10
```
Activate the newly created environment and install the package:
```bash
$ conda activate mvgl
(mvgl) $ poetry install --all-extras
```
`poetry` installs the editable version of the package to your environment. Thus,
any change you make to the package is reflected to your environment. 

## Usage

See `scripts/demo.py` for a demonstration or check docstring of `learn_multiview_graph()` as follows:
```bash
>>> import mvgl
>>> help(mvgl.learn_multiview_graph)
```

## Data

## License

`mvgl` was created by Abdullah Karaaslanli. It is licensed under the terms of
the GNU General Public License v3.0 license.

## Credits

`mvgl` was created with
[`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the
`py-pkgs-cookiecutter`
[template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
