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
L	1
J	1.0
hz	1.0
hx_i	-2.
hx_f	2.
hx_max	4.
hx_min	-4.
dh	8.0
dt	-1
n_step  100
n_flip 2
n_sample    20
T	2.7
outfile	auto
root ./
verbose 1
fid_series 0
fast_protocol 1
n_partition 10
compress_output wo_protocol
```
Each line corresponds to a different parameter. Here is a short documentation of what each parameter means:

## Documentation

* ``task``:
    What computation to perform. Options are ``SD`` (stochastic descent), ``ES`` (exact spectrum) and ``FO`` (find optimum).
    SD will perform standard stochastic descent. ES computes the full set of fidelities for a given number of bangs specified.
    FO will find the optimal protocol by computing all protocols (similar to ES) but without storing the full set of protocols.

* ``L``:
    Number of lattice sites for the periodic chain of qubits

* ``J``: 
    S_zS_z coupling strenght

* ``hz``:
    Longitudinal global static field (S_z)

* ``hx_i``:
    The initial state is taken to be the ground of the control Hamiltonian with transver field ``hx_i`` and couplings ``J,hz``

* ``hx_f``:
    The target state is taken to be the ground of the control Hamiltonian with transver field ``hx_i`` and couplings ``J,hz``

* ``hx_max``, ``hx_min``:
    Max and minimum transverse field for the bang-bang protocols

* ``dh``: 
    ``hx_max-hx_min`` ... not important for this

* ``dt``:
    Size of the temporal slices used to defined the bang-bang protocols. If the number of slice and the total ramp time is specified, then ``dt`` is computed automatically : ``dt = T/n_step``.

* ``n_step``:
    Number of time steps of the bang-bang protocol

* ``n_flip``:
    Type of stochastic descent. SD algorithm will check all possible moves of at most n_flip (see PRL paper, SI for more details) to find a local minimum. 

* ``n_sample``:
    Number of samples to collect. Each sample corresponds to a random initialization of the protocol followed by a stochastic descent until a local mimimum is found.

* ``T``:
    Total ramp time.

* ``outfile``:
    File name in which to output the results. If ``auto``, will automatically generate the file name using the parameters used.

* ``root``:
    Directory where to dump the results.

* ``verbose``:
    Verbose level (0 or 1)

* ``fid_series``:
    (0 or 1) Wether or not to store the succesive fidelities encountered during a SD run (see PRX paper for instance).

* ``fast_protocol``:
    Whether to speed-up protocol evaluation (for the fidelity) by precomputing subsets of product of unitaries.
    This will take more memory.

* ``n_partition``:
    Number of partitions to use for ``fast_protocol``. For instance, suppose you have ``n_step=100``, then using 
    ``n_partition=10``, it will pre-compute the set of all possible 2^10 product of unitaries for 10 consecutive time steps.
    When computing the fidelity of a specified protocol (of 100 time steps), it will slice this protocol in 10 equal parts of 10 steps and fetch the precomputed unitaries for each slice. Then it will perform the product of the 10 fetched unitaries. This leads to a speed-up of about ``n_step/n_partition``X (minus some overheads).

* ``compress_output``:
    Wether of not to output the protocols found for each sample (this can take a lot of memory if ``n_step`` is large and if the number of samples is large. Options are ``wo_protocol`` (without protocol) and ``w_protocol`` (with protocol).
