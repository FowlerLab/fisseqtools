# FISSEQ Tools

## Installation

To install the fisseq tools package, run:

```
pip install git+https://github.com/Lilferrit/fisseqtools.git
```

To install a specific version via a Git branch, hash, or tag run:

```
pip install git+https://github.com/Lilferrit/fisseqtools.git@<branch, tag, or hash>
```

To install from the local file system, run:

```
git clone git@github.com:Lilferrit/fisseqtools.git
pip install fisseqtools
```

The `-e` flag can alternatively be added to the pip command above to install the package into your python environment as an editable package.

## Usage

This package makes use of the `python-fire` library in order to generate command line utilities.
Any top module can be ran from the command line using the fire package, using:

```
python -m fisseqtools.<module> <function> <args>
```

See the the [python-fire](https://github.com/google/python-fire) and code documentation of top level modules for more details.
