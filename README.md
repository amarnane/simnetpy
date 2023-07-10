# simnet

Python Package for the creation and analysis of similarity networks.

<!-- Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── processed
    │   └── raw
    ├── docs
    ├── notebooks
    ├── scripts 
    └── src/phd_thesis
        ├── clustering
        ├── data
        ├── igraph
        ├── plotting
        ├── prediction
        ├── similarity
        └── utils 
-->


## Installation

The source code for this project can be installed using pip. 
```
pip install .
```
To remove the package simply use 
```
pip uninstall simnet
```

### Developer Mode
To install in developer mode (have changes in source code update without reinstallation) add `-e` flag
```
pip install -e .
```
Note: removing the package is slightly more complicated and a different command is needed to uninstall 
```
python setup.py develop -u
```

### Graph Tool
There is one dependency that cannot be installed through pip - `Graph-Tool`. This is a result of it's underlying `c++` dependencies.
The simplest method for python users is to make use of a conda environment, install this package using the commands above and install `graph-tool` using `conda-forge`
```
conda install -c conda-forge graph-tool
```
Note: this will not work on Windows. Alternative (conda independent) solutions can be found on the [Graph Tool Website](https://graph-tool.skewed.de/)
