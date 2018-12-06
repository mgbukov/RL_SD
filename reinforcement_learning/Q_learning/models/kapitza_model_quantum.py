import sys,os
from quspin.operators import hamiltonian, quantum_operator, exp_op
from quspin.basis import boson_basis_1d
from quspin.tools.Floquet import Floquet

from scipy.stats import ortho_group #,unitary_group


#path = os.path.join(os.path.expanduser('~'),"Dropbox/RL_exp/quantum_RL/Q_learning/Q_learning_cython/source_Qstruct/")
#sys.path.insert(0,path)
#from Q_struct_cpp import algebraic_dict
from Q_learning_cython.source_Qstruct import algebraic_dict

import numpy as np
import matplotlib.pyplot as plt

#import numpy.random as random
#'''
rng_path = os.path.join(os.path.expanduser('../'),"cpp_python_RNG/")
sys.path.insert(0,rng_path)
from cpp_RNG import cpp_RNG
random=cpp_RNG()
#'''

class model(object):
	def __init__(self,seed,params_model):
		
		self.seed=seed
		self.type=params_model['model_type']

		self.L=params_model['L']
		self.t_evolve=params_model['t_evolve']
		self.dt=self.t_evolve[1]-self.t_evolve[0] # protocol time step
		self.N_time_steps=len(self.t_evolve)-1
		self.N_time_steps_per_period=self.t_evolve.len_T

		if not (self.N_time_steps_per_period/4.0).is_integer() and self.N_time_steps_per_period!=1:
			raise ValueError("Number of points per period must be a multiple of the 4 bangs!")


		self.allowed_states=params_model['allowed_states']
		
		self.measure_mode=params_model['measure_mode']
		self.target_state=params_model['target_state']

		# infinite frequency bool
		try:
			self.inf_frequency=params_model['inf_frequency']
		except KeyError:
			self.inf_frequency=1

		if self.inf_frequency and self.N_time_steps_per_period>1:
			raise ValueError("More than one time points per period: infinite-frequency evolution must be stroboscopic!")

		# noise in initial state
		self.initial_state_noise=params_model['initial_state_noise']
		self.noise_level=params_model['noise_level']

		# stochasticity in taking the requested action
		self.stochastic_env=params_model['stochastic_env']
		self.stochasticity_level=params_model['stochasticity_level']
		
		self.params_model=params_model

		self._set_seed(self.seed)
		self._create_basis(params_model['basis_kwargs'])
		self._create_hamiltonians()
		self._create_unitaries()
		self._create_quantum_states()
		self._create_registry()
		
		self.counter=0

	def reset(self):
		# init auxiliary state variables
		
		if self.initial_state_noise:
			xj,yj=np.random.normal(0,1,size=(2,self.L) )
			psi_random=(xj+1j*yj)/np.linalg.norm(xj+1j*yj)
			psi=self.psi_i + self.noise_level*psi_random #ortho_group.rvs(self.L)[0]
			self.psi=psi/np.linalg.norm(psi)
			#print(np.abs(self.psi_i.dot(self.psi))**2)
		else:
			self.psi=self.psi_i

	def measure(self):

		self.current_reward = np.abs(self.psi.conj().dot(self.psi_target))**2
		if self.measure_mode=='deterministic':
			return self.current_reward
		else:
			rand_num=random.uniform(0,1)
			return 1 if rand_num<=self.current_reward else -1

	def _evolve_quantum_state(self,actions):
		# evolve system state
		for j,hx in enumerate(actions):
			# compute bang index within period
			if self.N_time_steps_per_period !=1:
				a=(j%self.N_time_steps_per_period)//(self.N_time_steps_per_period//4)
				yield self.U[(a,hx)].dot(self.psi)
			else:
				yield self.U[(0,hx)].dot(self.psi)

	def _set_seed(self,seed):
		random.seed(seed) # for cpp RNG
		np.random.seed(seed) # for drawing random states using scipy.stats
	
	def _create_basis(self,basis_kwargs):
		# create spin basis for system
		self.basis=boson_basis_1d(self.L,**basis_kwargs)
		

	def _create_hamiltonians(self):
		# creates total Hamiltonian of system

		self.momenta=np.array([i-self.L//2 for i in range(self.L)])

		# period average of drive_rot2
		#drive_rot2_ave=1.0/48.0*(6.0*period - 12.0*period**2 + 7.0*period**3) # step drive
		drive_rot2_ave=0.5 # cosine drive


		# create various trigonometric operators
		identity=[[1.0/self.L,i] for i in range(self.L)]
		gradient=[[self.momenta[i],i] for i in range(self.L)]
		trap=[[self.momenta[i]**2,i] for i in range(self.L)]
		hopping1=[[0.5,i,i+1] for i in range(self.L-1)]
		hopping2=[[0.5,i,i+2] for i in range(self.L-2)]

		hopping1r=[[-0.5j,i,i+1] for i in range(self.L-1)]
		hopping1l=[[+0.5j,i,i+1] for i in range(self.L-1)]

		hop_field1r=[[-0.5j*(2*(i-self.L//2)+1),i,i+1] for i in range(self.L-1)]
		hop_field1l=[[+0.5j*(2*(i-self.L//2)+1),i,i+1] for i in range(self.L-1)]

		# Kapitza pendulum operators
		idy=[['I',identity]]
		p=[['n',gradient]]
		p2=[['n',trap]]
		costheta=[['-+',hopping1],['+-',hopping1]]
		cos2theta=[['-+',hopping2],['+-',hopping2]]
		sintheta=[['-+',hopping1r],['+-',hopping1l]]
		sin_p_anti_comm=[['-+',hop_field1r],['+-',hop_field1l]]

		no_checks=dict(check_pcon=False,check_symm=False,check_herm=False)
		
		Id=hamiltonian(idy,[],basis=self.basis,**no_checks)
		P=hamiltonian(p,[],basis=self.basis,**no_checks)
		P2=hamiltonian(p2,[],basis=self.basis,**no_checks)
		CosTheta=hamiltonian(costheta,[],basis=self.basis,**no_checks)
		Cos2Theta=hamiltonian(cos2theta,[],basis=self.basis,**no_checks)
		SinTheta=hamiltonian(sintheta,[],basis=self.basis,**no_checks)
		SinTheta_P=hamiltonian(sin_p_anti_comm,[],basis=self.basis,**no_checks)


		# infinite frequency Floquet
		self.HF0 = 1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta \
				+ self.params_model['A']**2/(2*self.params_model['mw'])*drive_rot2_ave*0.5*(Id - Cos2Theta)

		# initial non-driven Hamiltonian
		self.H_i=1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta

		# control Hamiltonian
		self.H1=SinTheta # CosTheta

		# Floquet Hamiltonian
		
		Hstep0=1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta + 1.0/(2*self.params_model['mw'])*(-self.params_model['A'])*SinTheta_P 

		Hstep1=1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta + 1.0/(2*self.params_model['mw'])*(+self.params_model['A'])*SinTheta_P + 1.0/(2*self.params_model['mw'])*(self.params_model['A']**2)*0.5*(Id - Cos2Theta)

		Hstep2=1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta + 1.0/(2*self.params_model['mw'])*(+self.params_model['A'])*SinTheta_P

		Hstep3=1.0/(2.0*self.params_model['mw'])*P2 - self.params_model['mw']*CosTheta + 1.0/(2*self.params_model['mw'])*(-self.params_model['A'])*SinTheta_P + 1.0/(2*self.params_model['mw'])*(self.params_model['A']**2)*0.5*(Id - Cos2Theta)


		self.H_list=[Hstep0,Hstep1,Hstep2,Hstep3]
		self.dT_list=[0.25*self.t_evolve.T,0.25*self.t_evolve.T,0.25*self.t_evolve.T,0.25*self.t_evolve.T]
		
		
	def time_dep_hamiltonian(self):

		# time-dependent Hamiltonian
		def drive_rot(t):
			return -self.params_model['A']*np.sign(np.cos(self.params_model['Omega']*t))

		def drive_rot2(t):
			# problem: doesn't square to drive_rot!!!
			return self.params_model['A']**2 * 0.5*(1.0 - np.sign(np.sin(2.0*self.params_model['Omega']*t)) )


		mass_trap=[[1.0/(2*self.params_model['mw'])*(i-self.L//2)**2,i] for i in range(self.L)]
		hopping=[[-0.5*self.params_model['mw'],i,i+1] for i in range(self.L-1)]

		rot_drive_r=[[-0.5j*(2*(i-self.L//2)+1)*1.0/(2*self.params_model['mw']),i,i+1] for i in range(self.L-1)]
		rot_drive_l=[[+0.5j*(2*(i-self.L//2)+1)*1.0/(2*self.params_model['mw']),i,i+1] for i in range(self.L-1)]

		rot_drive2_0=[[0.5/(2*self.params_model['mw']*self.L),i] for i in range(self.L)] # identity term
		rot_drive2_1=[[-0.5*0.5/(2*self.params_model['mw']),i,i+2] for i in range(self.L-2)] # identity term

		static =[['n',mass_trap],['+-',hopping],['-+',hopping]]
		
		dynamic_rot=[['-+',rot_drive2_1,drive_rot2,[]],['+-',rot_drive2_1,drive_rot2,[]],['I',rot_drive2_0,drive_rot2,[]],
					 ['-+',rot_drive_r,drive_rot,[]],['+-',rot_drive_l,drive_rot,[]]
					]
					
		no_checks=dict(check_pcon=False,check_symm=False,check_herm=False)
		self.H=hamiltonian(static,dynamic_rot,basis=self.basis,**no_checks)

	def _create_unitaries(self):

		# preallocate unitary dict
		self.U=dict()

		if self.inf_frequency:
			for hx in self.allowed_states:
				self.U[(0,hx)] = exp_op(self.HF0 + hx*self.H1,a=-1j*self.dt).get_mat()
		else:
			for hx in self.allowed_states:
				for j in range(4):
					self.U[(j,hx)] = exp_op(self.H_list[j] + hx*self.H1,a=-1j*self.t_evolve.T/self.N_time_steps_per_period).get_mat()

			# compute Floquet Hamiltonian
			HF=Floquet({'H_list':self.H_list,'dt_list':self.dT_list},HF=True).HF
			self.HF=hamiltonian([HF],[],basis=self.basis)


	def _create_quantum_states(self):
		# initial state is GS of non-driven Hamiltonian
		# target is inverted position e'state of HF


		## Fourier transform: l,theta component
		self.theta = 2*np.pi/self.L*self.momenta
		self.FTltheta = 1.0/np.sqrt(self.L)*np.exp( -1j*np.outer( self.momenta, self.theta) )

		### INITIAL STATE
		# no contribution from drive
		E_i,V_i=self.H_i.eigsh(time=0.0,k=1,which='SA')
		self.psi_i=V_i[:,0]

		### TARGET STATE
		if self.target_state == 'Floquet':
			##### exact Floquet eigenstate at inverted position
			#
			# diagonalize infinite frequency Hamiltonian
			EF0,VF0 = self.HF0.eigh()
			# find index of eigenstate at inverted position
			ind_F0=np.where(np.argmax( np.abs(self.to_realspace(VF0))**2, axis=0 )==0 )[0][0]

			if self.inf_frequency:	

				self.psi_target= VF0[:,ind_F0] # L=20: (A,ind)=(2,2), (3,1), (4,1), (5,1)
			else:
				# compute exact Floquet states
				EF,VF = self.HF.eigh() #np.linalg.eigh(self.HF)
				# find index of state with largest overlap
				ind_F=np.argmax(np.abs( VF0[:,ind_F0].dot(VF) )**2)

				self.psi_target=VF[:,ind_F] # L=20: (A,ind)=(4,11), (3,8) , (13,2)
		elif self.target_state == 'Gauss':
			##### Gaussian state localized at inverted position
			#
			# width (HO length) of state of HO eigenstate approximated at inveted position for the potential of HF0
			a=(0.5*self.params_model['A']**2 - self.params_model['mw']**2)**(-1.0/4.0)
			
			gaussian=np.exp(-np.cos(self.theta)/a)
			self.psi_target=self.to_anglespace(gaussian/np.linalg.norm(gaussian))
		else:
			raise NotImplementedError


	def _create_registry(self):
		"""creates registry for quantum measurements, see environment.react()"""
		self.registry=algebraic_dict(0,3)
		
		

	def cost_function(self,state_current,actions_visited):
		'''
			self.registry attribute is added to model in Q_learning.py
		'''

		# if env is stochastic, do random actions with probability self.stochasticity_level
		if self.stochastic_env:
			actions_visited=np.array(actions_visited)
			random_nums=random.uniform(0.0,1.0,size=len(actions_visited))
			rand_inds,=np.where(self.stochasticity_level>random_nums)
			actions_visited[rand_inds]=random.choice(self.allowed_states,size=len(rand_inds))
			
		# evolve model
		for self.psi in self._evolve_quantum_state(actions_visited): pass


		# measure
		r=self.measure()

		if self.measure_mode is 'stochastic':
			
			# keep count of positive outcomes of final RL states
			if r>0:
				self.registry[0,state_current,0]+=1
			# keep count of all outcomes of final RL state
			self.registry[0,state_current,1]+=1
			
			# compute running probability
			R_current=self.registry.evaluate(0,state_current)
			running_p=R_current[0]/R_current[1]
			# compute error to be within 2.0 sigma (formula below available on Wikipeda)
			self.std_error=2.0*np.sqrt( running_p*(1.0-running_p)/R_current[1] ) 
			self.registry.set_value(0,state_current,2,self.std_error)
			return running_p
		else:
			return r


	def to_realspace(self,psi_l):
		return self.FTltheta.conj().T.dot(psi_l)

	def to_anglespace(self,psi_theta):
		return self.FTltheta.dot(psi_theta)


	def evolve_quantum_state(self,actions):
		# evolve system state
		for j,hx in enumerate(actions):
			# compute bang index within period
			if self.N_time_steps_per_period !=1:
				a=(j%self.N_time_steps_per_period)//(self.N_time_steps_per_period//4)
				self.psi=self.U[(a,hx)].dot(self.psi)
			else:
				self.psi=self.U[(0,hx)].dot(self.psi)
