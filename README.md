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