from Q_learning_cython.source_Qstruct import algebraic_dict


class Tabular(object):
	""" Defines tabular Q function. """
	def __init__(self,params_tabular):

		# parameters
		self.N_time_steps=params_tabular['N_time_steps']
		self.N_actions=params_tabular['N_actions']

		self._create_Q_tabular()
		self._create_trace()
		#self._create_registry()
		

	def _create_Q_tabular(self):
		# object of custom algebraic_disct class
		self.Q_tabular=algebraic_dict(self.N_time_steps,self.N_actions) 

	def _create_trace(self):
		"""eligibility trace used for credit assignment, c.f. algorithm TD(lambda)"""
		self.trace=algebraic_dict(self.N_time_steps,self.N_actions) 
	
	'''	
	def _create_registry(self):
		"""creates registry for quantum measurements, see environment.react()"""
		self.registry=algebraic_dict(0,3)
	'''	



