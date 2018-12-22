'''
Created on Oct, 20, 2017

@authors: Alexandre Day, Marin Bukov 

Implements different optimzation methods for quantum state preparation.
Stochastic descent with multi-spin flip optimization

'''

from .utils import UTILS
import numpy as np
from .Hamiltonian import HAMILTONIAN
from quspin.operators import exp_op
import time, sys, os, pickle
from itertools import product
from .model import MODEL
from .SD import SD
    
np.set_printoptions(precision=10)

class QSP:
    """ Quantum state preparation object. By default, the parameters
    used are specified in a para.dat file found in the same directory where
    the script is run in. To overide default parameters, one can specify parameters
    through the command line using the parameter=value syntax

    For instance:

    python my_qsp_script.py parameter_1=value_1 parameter_2=value_2 ... 

    Where for instance one can use: T=2.5 L=4
    """

    def __init__(self, argv = None, parameter_file = "para.dat", symm = True, quick_check=False):
        # Utility object for reading, writing parameters, etc. 
        self.utils = UTILS()
        self.parameters = {}
        
        if argv is not None:
            self.utils.read_command_line_arg(self.parameters, argv)

        # Reading parameters from para.dat file
        self.utils.read_parameter_file(self.parameters, file=parameter_file)
        
        # Command line specified parameters overide parameter file values

        if self.parameters['verbose'] == 1:
        # Printing parameters for user to see
            self.utils.print_parameters(self.parameters)

        if quick_check is False:
            # Defining Hamiltonian
            self.H = HAMILTONIAN(symm=symm, **self.parameters)
            # Defines the model, and precomputes evolution matrices given set of states
            self.model = MODEL(self.H, self.parameters)
    
    def run(self):
         # Run simulated annealing
        parameters = self.parameters
        model = self.model
        utils = self.utils

        if parameters['task'] == 'SD':
            print("Stochastic descent with n_flip = %i "%parameters['n_flip'])
            run_SD(parameters, model, utils)
        elif parameters['task'] == 'ES':
            print("Exact spectrum")
            run_ES(parameters, model, utils)
        elif parameters['task'] == 'FO':
            run_FO(parameters, model, utils)
        else:
            print("Wrong task option used")
    
    def evaluate_protocol(self, protocol):
        return self.model.compute_fidelity(protocol = protocol)

    def random_protocol(self):
        return np.random.randint(0, self.model.n_h_field, size=self.parameters['n_step'])

    def flip(self, p, i):
        ptmp = np.copy(p) # pretty slow !
        ptmp[i]^=1
        
        return ptmp
    
    def make_file_name(self, parameters = None):
        from copy import deepcopy as dp
        param_tmp = dp(self.parameters)
        if parameters is not None:
            for k, v in parameters.items():
                param_tmp[k]=v
            
        fname = self.utils.make_file_name(param_tmp)
        return fname

    def optimize(self, protocol_init = None):
        """ Just optimize using specified method and initial protocol, performs one run 
        Returns:
        --------
            best_fid, best_protocol, n_fid_eval, n_visit, fid_series
        """
        init_random = True
        if protocol_init is not None:
            self.model.update_protocol(protocol_init)
            init_random=False
        optimizer = SD(self.parameters, self.model, nflip=self.parameters['n_flip'], init_random=init_random)
        #best_fid, best_protocol, n_fid_eval, n_visit, fid_series = optimizer.run()
        return optimizer.run()

###################################################################################
###################################################################################
# ---------------> 
###################################################################################
###################################################################################

def run_SD(parameters, model:MODEL, utils, save = True):

    outfile = utils.make_file_name(parameters,root=parameters['root'])
    n_exist_sample, all_result = utils.read_current_results(outfile)
    n_sample = parameters['n_sample']

    if n_exist_sample >= n_sample :
        print("Samples already computed in file --> terminating...")
        all_fids = [f[1] for f in all_result]
        print("Max fidelity over samples :\t %.14f"%np.max(all_fids))
        print("Mean fidelity over samples :\t %.14f"%np.mean(all_fids))
        print("Std. fidelity over samples :\t %.14f"%np.std(all_fids))
        print("Goodbye !")
        return all_result
    elif n_exist_sample == 0:
        print("New data file, 0 samples availables... ")
    else:
        print("Appending data file, %i samples availables... \n"%n_exist_sample)

    if parameters['verbose'] == 0:
        blockPrint()

    print("\n\n-----------> Starting stochastic descent <-----------")
    
    n_iteration_left = n_sample - n_exist_sample  # data should be saved 10 times --> no more (otherwise things are way too slow !)
    n_mod = max([1,n_iteration_left // 10])
    optimizer = SD(parameters, model, nflip=parameters['n_flip'], init_random=True)

    for it in range(n_iteration_left):
        start_time=time.time()
        best_fid, best_protocol, n_fid_eval, n_visit, fid_series, move_history = optimizer.run()

        energy = model.compute_energy(protocol = best_protocol)
        result = [n_fid_eval, best_fid, energy, n_visit, best_protocol, fid_series, move_history] # -------> THIS IS WHAT WILL BE STORED IN THE PICKLE FILE 
        if parameters['compress_output'] == 'wo_protocol':
            result = [n_fid_eval, best_fid, energy, n_visit, [-1], fid_series, move_history]

        print("\n----------> RESULT FOR STOCHASTIC DESCENT NO %i <-------------"%(it+1))
        print("Number of fidelity eval \t%i"%n_fid_eval)
        print("Number of states visited \t%i"%n_visit)
        print("Best fidelity \t\t\t%.16f"%best_fid)
        print("Best hx_protocol\t\t",list(best_protocol))
        
        all_result.append(result)

        if save and it % n_mod == 0:
            with open(outfile,'wb') as f:
                pickle.dump([parameters, all_result],f)
                f.close()
            print("Saved iteration --> %i to %s"%(it + n_exist_sample, outfile))
        print("Iteration run time --> %.4f s" % (time.time()-start_time))
    
    print("\n Thank you and goodbye !")
    enablePrint()

    #np.mean(

    if save : # final saving !
        with open(outfile,'wb') as f:
            pickle.dump([parameters, all_result],f)
            print("Saved results in %s"%outfile)
            f.close()

    all_fids = [f[1] for f in all_result]
    print("Max fidelity over samples :\t %.14f"%np.max(all_fids))
    print("Mean fidelity over samples :\t %.8f"%np.mean(all_fids))
    print("Std. fidelity over samples :\t %.8f"%np.std(all_fids))
    if parameters['compress_output'] != 'wo_protocol':
        all_protocol = [f[4] for f in all_result]
        print("# of distinct protocols :\t %i"%len(np.unique(all_protocol, axis=0)))

    return all_result    

def run_ES(parameters, model:MODEL, utils):
    """
        Evaluate fidelity for a protocol of n_step in length, and stores the corresponding
        fidelities in a file. Also computes the energy (expectation value of the final Hamiltonian) of those protocols.
    """
    
    n_step = parameters['n_step']

    n_protocol = 2**n_step
    #exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision, with Sent
    exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision

    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
    st=time.time()
    # ---> measuring estimated time <---
    model.update_protocol(b2_array(0))
    psi = model.compute_evolved_state()
    model.compute_fidelity(psi_evolve=psi)
    #model.compute_Sent(psi_evolve=psi)
    model.compute_energy(psi_evolve=psi)
    print("Est. run time : \t %.3f s"%(0.5* n_protocol*(time.time()-st)))
    # ---> Starting real calculation <---

    st=time.time()
    for p in range(n_protocol):
        model.update_protocol(b2_array(p))
        psi = model.compute_evolved_state()
        exact_data[p] = (model.compute_fidelity(psi_evolve=psi), model.compute_energy(psi_evolve=psi)) #, model.compute_Sent(psi_evolve=psi))
    
    outfile = utils.make_file_name(parameters, root=parameters['root'])
    with open(outfile,'wb') as f:
        pickle.dump(exact_data, f, protocol=4)

    print("Saved results in %s"%outfile)
    print("Total run time : \t %.3f s"%(time.time()-st))
    print("\n Thank you and goodbye !")
    f.close()

def run_FO(parameters, model:MODEL, utils):
    """ Finds the Optimal protocol and it's fidelity """
    
    n_step = parameters['n_step']
    n_protocol = 2**n_step
    #exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision, with Sent
    #exact_data = np.zeros((n_protocol,2), dtype=np.float64) # 15 digits precision

    b2_array = lambda n10 : np.array(list(np.binary_repr(n10, width=n_step)), dtype=np.int)
    st=time.time()
    # ---> measuring estimated time <---
    model.update_protocol(b2_array(0))
    psi = model.compute_evolved_state()
    model.compute_fidelity(psi_evolve=psi)
    if parameters['para_evaluation'] == 1:
        para_n_slice = parameters['para_n_slice']
    else:
        para_n_slice = 1
    
    print("Est. run time : \t %.3f s"%((1./para_n_slice)*0.5* n_protocol*(time.time()-st)))
    # ---> Starting real calculation <---
    
    best_fid = -1
    st=time.time()

    if parameters['para_evaluation'] == 0:
        for p in range(n_protocol):

            model.update_protocol(b2_array(p))
            psi = model.compute_evolved_state()
            p_fid = model.compute_fidelity(psi_evolve=psi)
            if p_fid > best_fid:
                #print(p,'\t',"%.15f"%p_fid)
                best_fid = p_fid
                best_protocol = b2_array(p)
    elif parameters['para_evaluation'] == 1: # do you want to do parallel evaluation of the protocols
        para_n_slice = parameters['para_n_slice'] # number of slices
        para_this_slice = parameters['para_this_slice'] # this slice number
        assert para_this_slice < para_n_slice, "para_this_slice should be in range(0,para_n_slice)"
        n_prot_per_slice = n_protocol // para_n_slice

        i = para_this_slice
        dp = n_prot_per_slice

        for p in range(i*dp,(i+1)*dp):
            if p > n_protocol-1:
                break
            model.update_protocol(b2_array(p))
            psi = model.compute_evolved_state()
            p_fid = model.compute_fidelity(psi_evolve=psi)
            if p_fid > best_fid:
                #print(p,'\t',"%.15f"%p_fid)
                best_fid = p_fid
                best_protocol = b2_array(p)
    else:
        assert False, "Wrong para.dat input"

    outfile = utils.make_file_name(parameters, root=parameters['root'])
    with open(outfile,'wb') as f:
        pickle.dump([best_fid, best_protocol], f, protocol=4)
        
        #for p in range(n_protocol//n_parallel_partition) 

    print("Saved results in %s"%outfile)
    print("Total run time : \t %.3f s"%(time.time()-st))
    print("\n Thank you and goodbye !")
    f.close()

def symmetrize_protocol(hx_protocol):
    Nstep=len(hx_protocol)
    half_N=int(Nstep/2)
    for i in range(half_N):
        hx_protocol[-(i+1)]=-hx_protocol[i]

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
    
# Run main program !
if __name__ == "__main__":
    main()
