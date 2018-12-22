# Stochastic descent for quantum state preparation using bang-bang protocols

This package is a reproduction of the code used to generate the results shown in [arXiv:1803.10856](https://arxiv.org/abs/1803.10856) and to appear in PRL. Given an initial state and a desired target state along with a specified control Hamiltonian, stochastic descent optimizes the fidelity in order to find the optimal protocol. Some examples are provided below and can be run directly in the example directory.

# Install as a package

In order to install, we recommend using an Anaconda Python 3.6 then using:
```
pip install .
```
from within the ``stochastic_descent/`` directory

# Example

## Preparing a two-level system
The parameters used can be specified in the file ``para.dat``. The parameters are specified as follow:
```
task	SD
L	6
J	1.0
hz	1.0
```

