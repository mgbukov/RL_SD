import sys,os
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec

import functools

from scipy.stats import ortho_group #,unitary_group

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
		self._create_EOMs()
		self._create_EOM_dict()
		self._create_states()
		self._create_registry()
		
		self.counter=0

	def reset(self):
		# init auxiliary state variables
		
		if self.initial_state_noise:
			# random Gaussian noise in the initial position and momentum
			self.psi=self.psi_i+self.noise_level*np.random.normal(0.0,1.0,size=(2,) )
		else:
			self.psi=self.psi_i

	def measure(self):

		#self.current_reward = np.abs(( self.psi[0] + np.pi) % (2 * np.pi ) - np.pi )**2/np.pi**2 - np.abs(self.psi[1])**2
		self.current_reward = (( self.psi[0] + np.pi) % (2 * np.pi ) - np.pi )**2/np.pi**2  - 4.0*(self.psi[1])**2
		#print(self.psi,self.current_reward, (( self.psi[0] + np.pi) % (2 * np.pi ) - np.pi )**2/np.pi**2, 4.0*(self.psi[1])**2)
		
		if self.measure_mode=='deterministic':
			return self.current_reward
		else:
			return self.current_reward+np.random.normal(0.0,0.05)

	def _evolve_quantum_state(self,actions):
		# evolve system state
		for j,hx in enumerate(actions):
			# compute bang index within period
			if self.N_time_steps_per_period !=1:
				a=(j%self.N_time_steps_per_period)//(self.N_time_steps_per_period//4)
				yield evolve(self.psi,self.t_evolve[j],self.t_evolve[j+1],self.U[(a,hx)],real=True,atol=1E-14,rtol=1E-14)
			else:
				yield evolve(self.psi,self.t_evolve[j],self.t_evolve[j+1],self.U[(0,hx)],real=True,atol=1E-14,rtol=1E-14)
			

	def _set_seed(self,seed):
		random.seed(seed) # for cpp RNG
		np.random.seed(seed) # for drawing random states using scipy.stats
			

	def _create_EOMs(self):


		# define Hamilton's EOM in the rot frame (one for each bang step)

		def Hamilton_EOM_rot(t,z,mw,A,Omega,h):

			theta_dot = ( z[1] - A*np.sign(np.cos(Omega*t))*np.sin(z[0]) )/mw

			p_dot = -mw*np.sin(z[0]) \
					+A/(mw)*np.sign(np.cos(Omega*t))*z[1]*np.cos(z[0]) \
					-A**2/(4.0*mw)*( 1.0 - np.sign(np.sin(2*Omega*t)) )*np.sin(2.0*z[0]) \
					-h*np.cos(z[0])
			
			return np.array([theta_dot,p_dot])



		def Hamilton_EOM_rot_0(t,z,mw,A,Omega,h):

			theta_dot = ( z[1] - A*np.sin(z[0]) )/mw

			p_dot = -mw*np.sin(z[0]) \
					+A/(mw)*z[1]*np.cos(z[0]) \
					-h*np.cos(z[0])
			
			return np.array([theta_dot,p_dot])

		def Hamilton_EOM_rot_1(t,z,mw,A,Omega,h):

			theta_dot = ( z[1] + A*np.sin(z[0]) )/mw

			p_dot = -mw*np.sin(z[0]) \
					-A/(mw)*z[1]*np.cos(z[0]) \
					-A**2/(2.0*mw)*np.sin(2.0*z[0]) \
					-h*np.cos(z[0])
			
			return np.array([theta_dot,p_dot])

		def Hamilton_EOM_rot_2(t,z,mw,A,Omega,h):

			theta_dot = ( z[1] + A*np.sin(z[0]) )/mw

			p_dot = -mw*np.sin(z[0]) \
					-A/(mw)*z[1]*np.cos(z[0]) \
					-h*np.cos(z[0])
			
			return np.array([theta_dot,p_dot])

		def Hamilton_EOM_rot_3(t,z,mw,A,Omega,h):

			theta_dot = ( z[1] - A*np.sin(z[0]) )/mw

			p_dot = -mw*np.sin(z[0]) \
					+A/(mw)*z[1]*np.cos(z[0]) \
					-A**2/(2.0*mw)*np.sin(2.0*z[0]) \
					-h*np.cos(z[0])
			
			return np.array([theta_dot,p_dot])



		def Hamilton_EOM_HF0(t,z,mw,A,Omega,h):

			theta_dot = z[1]/mw
			p_dot = - mw*np.sin(z[0]) - A**2/(4.0*mw)*np.sin(2*z[0]) - h*np.cos(z[0])

			return np.array([theta_dot,p_dot])



		def Hamilton_EOM_H0(t,z,mw,A,Omega,h):

			theta_dot = z[1]/mw
			p_dot = - mw*np.sin(z[0])

			return np.array([theta_dot,p_dot])

		
		# define continuous functio implementation of the step drive
		self.EOM_rot=Hamilton_EOM_rot
		self.EOM_HF0=Hamilton_EOM_HF0
		self.EOM_H0=Hamilton_EOM_H0

		self.EOM_rot_list=[Hamilton_EOM_rot_0,Hamilton_EOM_rot_1,Hamilton_EOM_rot_2,Hamilton_EOM_rot_3]

		
	def _create_EOM_dict(self):
		# preallocate unitary dict
		self.U=dict()

		if self.inf_frequency:
			for hx in self.allowed_states:
				self.U[(0,hx)] = functools.partial( self.EOM_HF0, mw=self.params_model['mw'],A=self.params_model['A'],Omega=self.params_model['Omega'],h=hx )

		else:
			for hx in self.allowed_states:
				for j in range(4):		
					self.U[(j,hx)] = functools.partial( self.EOM_rot_list[j], mw=self.params_model['mw'],A=self.params_model['A'],Omega=self.params_model['Omega'],h=hx )


	def _create_states(self):
		# initial state is GS of non-driven Hamiltonian
		# target is inverted position e'state of HF

		self.psi_i=np.array([0.01,0.0])
		self.psi_target=np.array([np.pi,0.0])


	def _create_registry(self):
		"""creates registry for quantum measurements, see environment.react()"""
		self.registry=algebraic_dict(0,3)
		self.variance=0.0
				

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
			
			# keep count of all outcomes of final RL state
			self.registry[0,state_current,1]+=1

			# keep running mean estimate
			N=self.registry[0,state_current,1]
			# compute running mean estimate
			#self.registry[0,state_current,0]=(N*self.registry[0,state_current,0] + r)/(N+1.0)
			self.registry[0,state_current,0]*=N/(N+1.0)
			self.registry[0,state_current,0]+=r/(N+1.0)
			
			# compute error to be within 2.0 sigma, see table on https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
			#self.variance=(N*self.variance + (r-self.registry[0,state_current,0])**2 )/(N+1.0)
			self.variance*=N/(N+1.0)
			self.variance+=(r-self.registry[0,state_current,0])**2/(N+1.0)
			self.std_error=2.0*np.sqrt(self.variance)
			self.registry.set_value(0,state_current,2,self.std_error)
			return self.registry[0,state_current,0]
		else:
			return r


	def evolve_quantum_state(self,actions):
		# evolve system state
		for j,hx in enumerate(actions):
			# compute bang index within period
			if self.N_time_steps_per_period !=1:
				a=(j%self.N_time_steps_per_period)//(self.N_time_steps_per_period//4)
				self.psi=evolve(self.psi,self.t_evolve[j],self.t_evolve[j+1],self.U[(a,hx)],real=True,atol=1E-14,rtol=1E-14)
			else:
				self.psi=evolve(self.psi,self.t_evolve[j],self.t_evolve[j+1],self.U[(0,hx)],real=True,atol=1E-14,rtol=1E-14)

	def evolve_state_no_control(self,psi_0,times,free=False):
		# evolve system state

		EOM_args=(self.params_model['mw'], self.params_model['A'], self.params_model['Omega'], 0.0)

		if free:
			psi=evolve(psi_0,times[0],times,self.EOM_H0 ,f_params=EOM_args,real=True,atol=1E-14,rtol=1E-14)
		elif self.inf_frequency:
			psi=evolve(psi_0,times[0],times,self.EOM_HF0,f_params=EOM_args,real=True,atol=1E-14,rtol=1E-14)
		else:
			psi=evolve(psi_0,times[0],times,self.EOM_rot,f_params=EOM_args,real=True,atol=1E-14,rtol=1E-14)

		return psi
		


