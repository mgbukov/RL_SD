from qsp_opt import QSP

"""
Example:

Here we use L=1 (2-level system) and use have the control Hamiltonian H = S_z+h(t)S_x
The parameters of the model are specified in the para.dat file. These parameters
can be thought of as default parameters

Alternatively (which is convenient for distributed computing), the parameters can
be specified on the command line. For instance, to specify a ramp time of T=2.5
you can use the following command:

    python qsp_script.py T=2.5

This will overide the T value specified in the para.dat file and otherwise
will use the other parameters specified there.

"""

## Define the model

model = QSP()

## Run the computation. The results will be stored in the directory specified in the para.dat file

model.run()

