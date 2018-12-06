# distutils: language=c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

import numpy as _np
cimport numpy as _np

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator import preincrement as inc
from libcpp.unordered_map cimport unordered_map
from libc.math cimport signbit

from libc.stdint cimport uint64_t


cdef extern from "environment_object_template.h":
	cdef cppclass environment_object[T]:
		environment_object() except +
		environment_object(int,T) except +
		T initial,previous,current
		vector[T] visited
		void reset(T)



# environment class
cdef class environment:
	
	# declare all self's
	cdef unordered_map[double,int] actions_dict

	cdef vector[double] all_actions
	cdef vector[int] avail_inds
	
	cdef double reward
	#cdef public string protocol_str
	
	# update when everything is cythonized
	cdef environment_object[string] * state
	cdef environment_object[double] * action

	# wrap int128 for arithmetics to avoid segfaults
	cdef int _episode_step
	
	cdef object model
	cdef unordered_map[double,string] action_state_dict

	
	def __init__(self, object model, unordered_map[double,string] action_state_dict):

		self.state = new environment_object[string](model.N_time_steps,'-')
		self.action = new environment_object[double](model.N_time_steps,0.0)
		
		# create physical model
		self.model=model
		# actions-state dictionary
		self.action_state_dict=action_state_dict

		# reset environment
		self.reset()
		# compute actions dictionary
		self._init_environment()

	
	@property
	def episode_step(self):
		return self._episode_step


	cdef void _init_environment(self):
		""" One-tipe operations of environment class."""

		cdef int i 
		cdef double a

		self.all_actions=self.model.allowed_states
	
		self.avail_inds=range(len(self.all_actions))

		for i,a in enumerate(self.all_actions):
			self.actions_dict[a]=i


	
	cpdef tuple available_actions(self):
		""" Compute indices of available actions from current state"""	
		pass #return self.all_actions, self.avail_inds
	

	cpdef void reset(self,bool visited=True):
		'''Resets environment.'''

		self.model.reset()

		self.state.reset('')
		self.action.reset(0.0)

		#print(self.state.current)
		#print(self.action.current)
		#exit()

		self.reward=0.0

		self._episode_step=0
		

		#self.protocol_str=''



	cpdef void take_action(self,int indA,bool last_time_step_bool):
		'''Take action, observe reward and move over to next state. '''

		cdef double action=self.all_actions[indA]
		
		# update previous
		self.state.previous=self.state.current
		self.action.previous=self.action.current

		# update current
		self.action.current=action

		# environment's reaction
		self._react()

		# update visited lists
		self.action.visited.push_back(action)
		self.state.visited.push_back(self.state.current)
		# evaluate reward
		if last_time_step_bool:
			#exit()
			self.reward=self.model.cost_function(self.state.current,self.action.visited)
			
		# increment time step
		inc(self._episode_step)


	cdef void _react(self):
		""" Reaction of environment to action chosen by agent. 

		Computes integer corresponding to the protocol sequence (up to the present time)
		and assigns it to state.current.

		"""

		self.state.current+=self.action_state_dict[self.action.current]

		'''
		if not signbit(self.action.current):
			self.state.current|=self._one64<<self._episode_step64
		'''	
	
		


	