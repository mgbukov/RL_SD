# Tabular Q-Learning to prepare the inverted position state of the Kapitza oscillator

This package implements the tabular Q-Learning code used in [Physical Review X](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031086). There are some peculiarities of appliying RL to prepare quantum states (see paper for a detailed discussion): (i) wavefunction collapse requires rewards to be sparse and given only in the end of the episode; (ii) the non-observability of quantum states imposes that the RL states consist of sequences of the RL actions; (iii) repetitions are used to collect quantum statistics (quantum measurements being non-deterministic)


## Structure of the Code:

The main control file, which contains all model and algorithm parameters is `main_kapitza.py`. It calls the `apply_RL` function, located in `/search_algos/RL.py`. The latter instantiates the `Q_learning` class, which contains the Q-Learning algorithm.

There are two models which provide simulators for the environment, located in `/models/`: `kapitza_model_quantum.py` and `kapitza_model_classical.py`. 

## Compiling the Code:

The package is written in Python (with the exception of a custom data structure for the Q-function). Additionally, some parts of the code are written in Cython/C++ for speed. C++ code is wrapped using Cython. All memory is managed by Python.


Required Python packages: [quspin](http://weinbe58.github.io/QuSpin/), cython

### Random Number Generator Wrapper

To ensure reproducibility of the results, a C++ random number generator is wrapped up using Cython and made available in Python. To compile the libraries, go to 

`
/cpp_python_RNG/
`
and execute:
`
python setup.py build_ext -i
`

### Custom Data Structure for the Q-function

The package provides a custom data structure for the Q-function, which is similar to a python dictionary, but in addition supports elementary algebraic operations. Importantly, one does not need to pre-allocate the size of this object: as a result, it only stores the observed state-action pairs, which allows the algorithm to explore exponentially large state spaces efficiently. 

The Q-function data structure can be thought of as a 3-dim array `Q[t_step, state, action_index]`, where `t_step` is the time step integer, `state` is a byte-string composed of the available actions, and `action_index` is the index, corresponding to a given action (e.g. in the default version the actions are `[-4.0,0.0,4.0]` so `action_index=1` corresponds to action `0`, etc.). 

This data structure is written in Cython, and is provided by the Cython class `algebraic_dict`. To compile it, navigate to 
`
/Q_learning/Q_learning_cython/source_Qstruct/Q_struct_cpp.pyx
`
and execute:
`
python setup.py build_ext -i
`

### Python Version

The Pytnon version of the code can be found under
`
/Q_learning/Q_learning_python/
`
The file `Q_learning_python/Q_learning_wrapper.py` contains the class `Q_learning`, which runs the Q-Learning algorithm. The algorithm itself is written in the class `Q_learning_python` located in `Q_Learning.py`. The environment class can be found in `environment.py`.

The package architecture, in particular the wrapper class, in the Python version is merely to mirror the structure of the Cython version, where it is required.

### Cython Version

We also provide a faster, Cython version of the Q-Learning algorithm, under
`
/Q_learning/Q_learning_cython/
`
Once again, we use a wrapper class located in `Q_learning_cython/Q_learning_wrapper.py` (note, this is a different file in the cython directory), which calls the Cython class `Q_learning_cython` defined in `Q_learning.pyx`. 

In the Cython version, all memory is managed by Python: in other words, the objects which store the data are created in the python wrapper, but only their memory address is passed to the cython class which fills in the values as the algorithm runs. This is required for stability, and is also the reason for using the wrapper class architecture.  

The environment class has also been cython-ized, and can be found under `/source_env/environment.pyx`. 

To compule the Cython code, navigate to
`
/Q_learning/Q_learning_cython/
`
and execute:
`
python setup.py build_ext -i
`


## Running the Code

Choosing a version: toggling between the python and cython version of the package is done in lines 6, 7 of `/search_algos/RL.py`.

To run the package, navigate to `/Q_learning/` and execute:
`
python main_kapitza.py 0 10.0 4 15 1 1 1 0
`
where the command-line parameters sequence "0 10.0 4 15 1 1 1 0" corresponds to differrent model/algorithm parameters (see `main_kapitza.py` for details)

To analyze the results, look at the attributes of the wrapper classes which store the encountered action sequences, states, and rewards during the training procedure. 

