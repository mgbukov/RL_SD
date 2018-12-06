#distutils: language=c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

from __future__ import division

import numpy as _np
cimport numpy as _np

from libcpp.cast cimport reinterpret_cast
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator import preincrement as inc

from libc.math cimport sqrt,log,exp 


include "./source_Qstruct/Q_struct_cpp.pyx"
include "./source_env/environment.pyx"
include "../../cpp_python_RNG/cpp_RNG.pyx"


cdef extern from "greedy_action.h":
	vector[int] greedy_indices(vector[double]) nogil

cdef extern from "greedy_action.h":
	vector[double] evaluate_vector(vector[double],vector[int]) nogil
	vector[int] evaluate_vector(vector[int],vector[int]) nogil



cdef void set_v2_into_v1(double *v1,vector[double] v2) nogil:
	cdef unsigned int j, stop

	j=0
	stop=v2.size()

	while j<stop:
		v1[j]=v2[j]
		inc(j)


cdef class Q_learning_cython:

	# initit
	cdef public unsigned int N_episodes,N_greedy_episodes,N_repeat,N_replay,N_iterations,replay_frequency
	cdef public double alpha,lmbda,beta_RL_i,beta_RL_f

	cdef public object model
	cdef public environment env

	# initialize_algorithm
	cdef double * actions_taken
	cdef double * actions_best_encountered
	cdef double * return_best_encountered
	cdef string states_best_encountered
	

	# training and test loops
	cdef public unsigned int episode, ep, greedy_episode
	cdef double beta_RL,std_error_best
	cdef int _j

	# run_episode
	
	# define poiners to data holders
	cdef double * episodic_return
	cdef double * episodic_return_true
	cdef double * return_running_ave
	cdef double * return_true_ave
	cdef char * episodic_protocol

	# Q_structs
	cdef unsigned int N_time_steps, N_actions
	cdef algebraic_dict Q_tabular,trace

	# RNG
	cdef RNG_c * random

	# exploration
	cdef object exploration_schedule

	cdef int j


	def __init__(
				self,int seed,object model,dict params_QL,dict params_tabular,
				double[:] episodic_return,
				double[:] episodic_return_true,
				double[:] return_running_ave,
				double[:] return_true_ave,
				char[:,::1] episodic_protocol, # c-ordered
				double[:] actions_best_encountered,
				double[:] actions_taken,
				double[:] return_best_encountered,
				char[:] states_best_encountered,
		):
		
		# sefine pointers to data holders
		self.episodic_return=&episodic_return[0]
		self.episodic_return_true=&episodic_return_true[0]
		self.return_running_ave=&return_running_ave[0]
		self.return_true_ave=&return_true_ave[0]
		self.episodic_protocol=&episodic_protocol[0,0]

		self.actions_taken=&actions_taken[0]
		self.actions_best_encountered=&actions_best_encountered[0]
		self.return_best_encountered=&return_best_encountered[0]
		
		
		# define random numbe generator
		self.random=new RNG_c()
		
		# set random generator seed
		self.random.seed(seed)


		# Q-function object class
		self.N_time_steps=params_tabular['N_time_steps']
		self.N_actions=params_tabular['N_actions']

		# create environment
		self.env=environment(model,params_QL['action_state_dict'])
		self.model=model

		# Q Learning algorithm parameters
		self.N_episodes=params_QL['N_episodes']
		self.N_greedy_episodes=params_QL['N_greedy_episodes']
		self.N_repeat=params_QL['N_repeat']
		self.N_replay=params_QL['N_replay']
		self.replay_frequency=params_QL['replay_frequency']
		self.N_iterations=self.N_episodes+self.N_greedy_episodes

		self.alpha=params_QL['alpha']
		self.lmbda=params_QL['lmbda']
		
		self.exploration_schedule=params_QL['exploration_schedule']
		self.beta_RL_i=params_QL['beta_RL_i']
		self.beta_RL_f=params_QL['beta_RL_f']

	
		
		# initialize Q-learning algorithm

		self.initialize_algorithm()

		##### TRAIN STAGE #####

		for episode in range(self.N_episodes):
			# run exploratory episode
			self.episode=episode

			# set exploration temperature
			self.beta_RL=self.exploration_schedule(self.episode)

			# run episode
			self.run_episode(self.alpha,greedy=False)

			######## QUANTUM MEASUREMENT ########

			if self.env.model.measure_mode == 'stochastic':

				# update best encountered actions
				#create an auziliary function to release gil
				if self.episode==0:
					#self.actions_best_encountered=self.env.action.visited
					set_v2_into_v1(self.actions_best_encountered,self.env.action.visited) 
					
				# REPEAT episode for statistics w/o learning
				# put meas in model
				if self.model.std_error<1E-12 or self.model.std_error>=1E-2:
					for _ in range(self.N_repeat):
						self.repeat_episode()

				# run episode after repeats and evaluate it
				self.run_episode(self.alpha,greedy=False,mode='repeat')
				self.compute_stats(self.episode)
				

				# REPLAY and learn best encountered policy
				if self.episode%self.replay_frequency==0 and self.episode>0: 

					std_error_best=self.model.registry[0,self.states_best_encountered,2]
					if 1E-12<= std_error_best <= (1E-1 if self.episode<0.9*self.N_episodes else 1E-2):
					
						for j in range(self.N_replay):
							# replays carried out with unit learning rate
							self.run_episode(1.0,greedy=True,mode='replay')
							self.return_best_encountered[0] = self.env.reward 
						
					# print stats
					self.print_stats(mode='replay')

				
			######## CLASSICAL MEASUREMENT ########

			else:

				# evaluate episode
				self.compute_stats(self.episode)
				
				if self.episode%self.replay_frequency==0 and self.episode>0:
					for j in range(self.N_time_steps):
						self.run_episode(self.alpha,greedy=True,mode='replay')
						self.return_best_encountered[0] = self.env.reward 

					# print stats
					self.print_stats(mode='replay')

				

		##### TEST STAGE #####
		# import test model params
		self.model.initial_state_noise=params_QL['test_model_params']['initial_state_noise']
		self.model.stochastic_env=params_QL['test_model_params']['stochastic_env']

		# run w/o exploration to get the Q function convergent
		for ep in range(self.N_greedy_episodes):
			greedy_episode = ep + self.N_episodes
			self.run_episode(self.alpha,greedy=True)
			self.compute_stats(greedy_episode)

		# update best encountered protocol
		with nogil:
			for j in range(self.N_time_steps):
				states_best_encountered[j]=self.states_best_encountered[j] # try w/o loop over j's
		


	cdef void repeat_episode(self):
		""" Runs a sequence of taken actions to gains statistics. """
		self.env.model.reset()
		self.env.model.cost_function(self.env.state.current,self.env.action.visited)


	cdef void run_episode(self,double alpha,bool greedy=False,string mode=string('')):
		""" Runs an episode of the Q Learning algorithm """
		cdef unsigned int time_step, indA
		cdef bool last_time_step_bool
		cdef double delta_t=0.0
		
		# initialize episode
		self.initialize_episode()

		
		# compute Q function in initial state for all actions
		Q=self.Q_tabular.evaluate_c(0, self.env.state.current)


		# episode 
		for time_step in range(self.N_time_steps):

			
			last_time_step_bool=time_step == self.N_time_steps-1

			# pick action				
			indA=self.choose_action(Q,time_step,greedy=greedy,mode=mode)

			
			# take action, observe new state and reward
			self.take_action(indA,time_step,last_time_step_bool)
		
			###############################################################
			#####   from now on: state.current --> state.previous   #######
			###############################################################
			

			# fire trace for observed (s,a) pair
			self.trace.set_value_c(time_step,self.env.state.previous,indA,alpha)


			# Q learning update rule; GD error in time
			delta_t = - Q[indA] # error in time
			

			# check if new state is terminal
			if last_time_step_bool: 
				delta_t += self.env.reward
				self.trace*=delta_t
				self.Q_tabular += self.trace
				break

			### move on to next state ###
			
			
			# t-dependent Watkins Q learning
			Q=self.Q_tabular.evaluate_c(time_step+1, self.env.state.current)
			
			# update GS error in time
			delta_t += _np.max(Q) # Bellman error
			'''
			# update Q function and traces: if-statement below to make use of in-lace multiplication
			self.Q_tabular += delta_t*self.trace
			self.trace *= self.lmbda # <--- change to accommodate RETRACE(lmbda)
			'''
			if abs(delta_t)<=1E-32:
				self.trace *= self.lmbda # <--- change to accommodate RETRACE(lmbda)
			else:
				self.trace *= delta_t
				self.Q_tabular += self.trace
				# update traces
				self.trace *= self.lmbda/delta_t # <--- change to accommodate RETRACE(lmbda)
			
		#self.print_stats(mode=mode)
		#exit()

	cdef int choose_action(self, vector[double] Q,int time_step,bool greedy=False,string mode=string('')) nogil:
		""" Chooses action from current Q function according to Boltzmann exploration. """

		cdef int indA,indA_greedy
		cdef double A,A_greedy,random_num
		cdef vector[double] avail_actions
		cdef vector[int] inds_max


		if mode == 'repeat':
			A = self.actions_taken[time_step]
			indA = self.env.actions_dict[A]
		
		elif mode == '':

			# get indices of all available actions from current state of environment
			avail_actions=self.env.all_actions

			# greedy action is any random action our of the available actions which maximize Q
			#A_greedy = avail_actions[self.random.choice(_np.argwhere(Q[avail_inds]==_np.amax(Q[avail_inds])).flatten() ) ]
			inds_max=greedy_indices(Q)
			A_greedy = avail_actions[self.random.choice(inds_max)]
			indA_greedy=self.env.actions_dict[A_greedy]
			### eps-greedy exploration
			if not greedy or time_step%2==1:
				random_num=self.random.uniform(0.0,1.0)
				if random_num>self.beta_RL/self.beta_RL_f:
					A = self.random.choice(avail_actions)
				else:
					A = A_greedy
			else:
				# pick greedy action
				A = A_greedy
		
			# find the index of A
			indA = self.env.actions_dict[A]

			# reset traces if A is exploratory
			if indA != indA_greedy:
				self.trace.reset_c()
			
		elif mode == 'replay':
			A = self.actions_best_encountered[time_step]
			indA = self.env.actions_dict[A]

		return indA

	cdef void take_action(self,int indA,int time_step,bool last_time_step_bool):
		# move environment one step forward
		self.env.take_action(indA,last_time_step_bool)
		self.actions_taken[time_step]=self.env.action.current


	cdef void _create_Q_tabular(self):
		# object of custom algebraic_disct class
		self.Q_tabular=algebraic_dict(self.N_time_steps,self.N_actions) 

	cdef void _create_trace(self):
		"""eligibility trace used for credit assignment, c.f. algorithm TD(lambda)"""
		self.trace=algebraic_dict(self.N_time_steps,self.N_actions) 



	cdef void initialize_episode(self):
		""" Initializes learning episode """
		# reset environment
		self.env.reset()
		# reset trace and usage vector
		self.trace.reset_c()


	cdef void initialize_algorithm(self):

		# creat Q-function and trace Q_struct variables
		self._create_Q_tabular()
		self._create_trace()

		# expected return
		self.return_best_encountered[0]=-1.0
		self.states_best_encountered=string('')
		

	cdef void compute_stats(self,int episode):
		"""Computes average reward, running reward, etc."""

		# store current return
		self.episodic_return[episode]=self.env.reward
		self.episodic_return_true[episode]=self.env.model.current_reward

		# compute running average of return
		self.return_running_ave[episode]=1.0/(episode+1)*(self.env.reward + episode*self.return_running_ave[episode-1])
		self.return_true_ave[episode]=1.0/(episode+1)*(self.env.model.current_reward + episode*self.return_true_ave[episode-1])
		
		# store current protocol
		cdef int j
		with nogil:
			for j in range(self.N_time_steps):
				self.episodic_protocol[episode*self.N_time_steps + j]=self.env.state.current[j] # convert to character pointer, do loop
		

		# update best encountered quantities
		if self.model.measure_mode == 'deterministic':
			self.update_best_encountered_classical(episode)
		else:
			self.update_best_encountered_quantum(episode)

	cdef void update_best_encountered_quantum(self,int episode):
		cdef int j

		if self.return_best_encountered[0] < self.env.reward and 1E-12 <= self.model.std_error <= (1E-1 if episode<0.9*self.N_episodes else 1E-2) or episode == 0:

			self.return_best_encountered[0] = self.env.reward 
			#self.actions_best_encountered = self.env.action.visited
			set_v2_into_v1(self.actions_best_encountered,self.env.action.visited) 

			self.states_best_encountered=self.env.state.current
			
	cdef void update_best_encountered_classical(self,int episode) nogil:
		cdef int j

		if self.return_best_encountered[0] < self.env.reward or episode == 0:

			self.return_best_encountered[0] = self.env.reward
			
			#self.actions_best_encountered = self.env.action.visited
			set_v2_into_v1(self.actions_best_encountered,self.env.action.visited) 

			self.states_best_encountered=self.env.state.current
		


	def print_stats(self,mode='None'):

		print(str(mode)+"_ep=%i_r=%0.4f_rbest=%0.4f_p=%0.4f_betaRL=%0.2f_samples=%i_E=%0.3f_state=%s" 
			%(self.episode, 
				self.env.reward, 
				self.return_best_encountered[0], 
				self.env.model.current_reward, 
				self.beta_RL, #self.eps, 
				self.model.registry[0,self.env.state.current,1],
				self.model.registry[0,self.env.state.current,2],
				self.env.state.current,
			 )
			)
		print('')
