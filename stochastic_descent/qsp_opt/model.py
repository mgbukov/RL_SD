import numpy as np
import time
import copy
from scipy.linalg import expm
from numpy.linalg import multi_dot
from .cython import protocol_to_base10_int

def overlap(psi1,psi2):
    """ Square of overlap between two states """
    t=np.abs(np.dot(np.conj(psi1.T),psi2))
    return t*t

class MODEL:
    def __init__(self, H, param):
        
        self.H = H  # shallow copy 
        self.param = param

        self.bitarray_to_b10 = lambda protocol : int("".join(list(np.array(protocol,dtype=str))),2)

        # calculate initial and final states wave functions (ground-state)
        self.psi_i = H.ground_state(hx=param['hx_i'])
        self.psi_target = H.ground_state(hx=param['hx_f'])

        self.H_target = H.evaluate_H_at_hx(hx=self.param['hx_f']).todense()
        
        print("{0:<30s}{1:<5.5f}".format("Initial overlap is",overlap(self.psi_i, self.psi_target)[0][0]))
    
        self.param = param
        self.n_h_field = len(self.H.h_set)
        self.precompute_mat = {} # precomputed evolution matrices are indexed by integers 
    
        print("\n-----------------> Setting up computation <---------------------")
        print("Precomputing evolution matrices ...",end='\n')
        start=time.time()
        self.precompute_expmatrix()

        self.split_protocol = self.param['fast_protocol']
        self.n_partition = self.param['n_partition']
        if self.param['fast_protocol'] is True:
            assert self.param['n_step'] % self.n_partition == 0, "n_step needs to be a multiple of %i !"% self.n_partition
            self.precompute_split_protocol(n_partition=self.n_partition)

        print(" Done in %.4f seconds"%(time.time()-start))

    def precompute_expmatrix(self):
        # Precomputes the evolution matrix and stores them in a dictionary
        
        # set of possible -> fields 
        h_set = self.H.h_set

        for idx, h in zip(range(len(h_set)), h_set):
            self.precompute_mat[idx] = expm(-1j*self.param['dt'] * np.asarray(self.H.evaluate_H_at_hx(hx=h).todense()))
        
        self.param['V_target'] = self.H.eigen_basis(hx=self.param['hx_f'])
    
    def prot_to_b10(self, array, i1, i2):
        """ Fastest way I found to compute this ! """ 
        value = 0
        p = 1
        for b in reversed(array[i1:i2]):
            value = value + b*p
            p*=2
        return value
        
    def precompute_split_protocol(self, n_partition=2):
        # n should be the number of partitions wanted !!
        lin_dim, _ = self.precompute_mat[0].shape
        n_step = self.param['n_step']
        n_step_per_partition = n_step//n_partition
        print("Memory used for storing matrices : %.3f MB"%((lin_dim*lin_dim*16*2.*2.**n_step_per_partition)/(10**6.)))

        self.split_size = n_step_per_partition     
        # only works for binary fields
        h_set = self.H.h_set
        def b2_to_array(n10, n=10):
            str_rep = np.binary_repr(n10, width=n)
            return np.array(list(str_rep),dtype=np.int)

        self.precompute_protocol = np.zeros((2**n_step_per_partition, self.precompute_mat[0].shape[0], self.precompute_mat[0].shape[1]), dtype=np.complex)

        for i in range(2**n_step_per_partition):
            partial_protocol = b2_to_array(i, n=n_step_per_partition)
        
            ### COMPUTING PRODUCT ###
            init_matrix = np.copy(self.precompute_mat[partial_protocol[0]])
            for idx in partial_protocol[1:]:
                init_matrix = self.precompute_mat[idx].dot(init_matrix)
            #########################
            self.precompute_protocol[i] = init_matrix # all info stored here 
    
    def compute_evolved_state(self, protocol=None): 
        """ Compute the evolved state by applying unitaries to protocol """
        if protocol is None:
            protocol = self.H.hx_discrete
        psi_evolve=self.psi_i.copy().astype(np.complex128)
        #print(psi_evolve)
        #exit()
        #print(psi_evolve.dtype)
        #print(self.precompute_protocol[0].dtype)
        #exit()
        if self.split_protocol is True:
            n_partition = len(protocol) // self.split_size

            pos_i = 0
            for i in range(n_partition):
                #idx = self.prot_to_b10(protocol, pos_i, pos_i+self.split_size)
                idx = protocol_to_base10_int(protocol, pos_i, pos_i+self.split_size)
                pos_i+=self.split_size
                psi_evolve = self.precompute_protocol[idx].dot(psi_evolve)
        else:
            #psi_evolve = multi_dot(self.precompute_mat[protocol]).dot(psi_evolve)
            for idx in protocol:
                psi_evolve = self.precompute_mat[idx].dot(psi_evolve)

        return psi_evolve
    
    def compute_continuous_fidelity(self, continuous_protocol):
        """ Computes fidelity for continuous field values (much slower, matrices have not been pre-computed) """
        psi_evolve=self.psi_i.copy()
        for h in continuous_protocol:
            psi_evolve = expm(-1j*self.param['dt']*self.H.evaluate_H_at_hx(hx=h).todense()).dot(psi_evolve)
        return overlap(psi_evolve, self.psi_target)[0,0]
    
    def compute_fidelity(self, protocol = None, psi_evolve = None):
        """ Computes fidelity for a given protocol """
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)
        return overlap(psi_evolve, self.psi_target)[0,0]

    def update_protocol(self, protocol, format = 'standard'):
        """ Update full protocol
            Format is integers or real values (which are then converted to integers) """

        if format is 'standard':
            self.H.hx_discrete = copy.deepcopy(np.array(protocol))
        elif format is 'real':
            n_bang = len(protocol)
            for t, h in zip(range(n_bang),protocol):
                self.H.update_hx(time=t,hx=h) 
        self.psi_evolve = None

    def update_hx(self, time:int, hx_idx:int):
        """ Update protocol at a specific time """
        self.H.hx_discrete[time]=hx_idx

    def flip_hx(self, hx_idx):
        """ Flips protocol at postions hx_idx, where hx_idx is specified as a list or integer 
            This works only when n_h_field == 2 (binary)
        """
        self.H.hx_discrete[hx_idx]^=1

    def protocol(self):
        """ Returns current protocol """
        return self.H.hx_discrete
    
    def protocol_hx(self, time):
        """ Returns protocol value at time """
        return self.H.hx_discrete[time]
    
    def random_flip(self,time):
        """ Proposes a random update a time : """
        current_h = self.H.hx_discrete[time]
        return np.random.choice(list(set(range(self.n_h_field))-set([current_h])))

    def swap(self, time_1, time_2):
        """ Swaps field configurations h(t1) <-> h(t2) """
        tmp = self.H.hx_discrete[time_1]
        self.H.hx_discrete[time_1] = self.H.hx_discrete[time_2]
        self.H.hx_discrete[time_2] = tmp

    def compute_energy(self, protocol = None, psi_evolve = None):
        """ Compute energy of evolved state w.r.t to target Hamiltonian """
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)
        return np.real(np.dot(np.conj(psi_evolve).T, np.dot(self.H_target, psi_evolve)))[0,0]
    
    def compute_Sent(self, protocol=None, psi_evolve = None):
        """ Compute Sent of evolved state acc. to Hamiltonian """
        if psi_evolve is None:
            psi_evolve = self.compute_evolved_state(protocol=protocol)

        return self.H.basis.ent_entropy(psi_evolve.squeeze())['Sent_A']

    def compute_magnetization(self, protocol = None):
        """ Computes magnetization """
        if protocol is None:
            protocol = self.H.hx_discrete
        Nup = np.sum(protocol)
        Ndown = protocol.shape[0] - Nup
        return Nup - Ndown

