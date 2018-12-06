import numpy as _np
import struct
from .Q_learning import Q_learning_python


class Q_learning():

	def __init__(self,seed,model,params_QL,params_tabular):


		# Q Learning algorithm parameters
		N_episodes=params_QL['N_episodes']
		N_iterations=params_QL['N_episodes']+params_QL['N_greedy_episodes']
		N_time_steps=params_tabular['N_time_steps']

		self._initialize_data_holders(N_episodes,N_iterations,N_time_steps)
	
		QL=Q_learning_python(seed,model,params_QL,params_tabular)

		self.episodic_return=QL.episodic_return
		self.episodic_return_true=QL.episodic_return_true
		self.return_running_ave=QL.return_running_ave
		self.return_true_ave=QL.return_true_ave
		self.episodic_protocol=QL.episodic_protocol
		self.actions_best_encountered=QL.actions_best_encountered
		self.actions_taken=QL.actions_taken
		
		self.return_best_encountered=self.return_best_encountered.squeeze()
		self.states_best_encountered=self.states_best_encountered.squeeze()

	
	def _initialize_data_holders(self,N_episodes,N_iterations,N_time_steps):

		self.episodic_return=_np.zeros((N_iterations,),dtype=_np.float64)
		self.episodic_return_true=_np.zeros_like(self.episodic_return)
		self.return_running_ave=_np.zeros_like(self.episodic_return)
		self.return_true_ave=_np.zeros_like(self.episodic_return)
		
		# protocols stored as integers
		self.episodic_protocol=_np.zeros((N_iterations,N_time_steps),dtype=_np.int8)

		
		self.actions_taken=_np.zeros((N_time_steps,),dtype=_np.float64)
		self.actions_best_encountered=_np.zeros((N_time_steps,),dtype=_np.float64)

		self.return_best_encountered=_np.array(-1.0,ndmin=1,dtype=_np.float64)

		self.states_best_encountered=_np.zeros((N_time_steps,),dtype=_np.int8)
		#self.states_best_encountered[0]=bytes('', 'utf-8') 
		

	def byte_to_str(self,protocol):
		# protocol is a list of int8 encodings of the protocol strings, as returned by cython code
		return struct.pack('%sB' % len(protocol), *protocol)

	

