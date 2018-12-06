import sys,os
import numpy as np

#import numpy.random as random
#'''
rng_path = os.path.join(os.path.expanduser('../'),"cpp_python_RNG/")
sys.path.insert(0,rng_path)
from cpp_RNG import cpp_RNG
random=cpp_RNG()
#'''


from .environment import environment
from Q_learning_cython.source_Qstruct import algebraic_dict


np.set_printoptions(precision=4,suppress=True) #


class Q_learning_python():

	def __init__(self,seed,model,params_QL,params_tabular):
		# set random generator seed
		random.seed(seed)

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

		self.N_time_steps=params_tabular['N_time_steps']
		self.N_actions=params_tabular['N_actions']

		# create environment
		self.env=environment(model,params_QL['action_state_dict'])
		self.model=model


		# initialize Q-learning algorithm
		self.initialize_algorithm()

		##### TRAIN STAGE #####

		for episode in range(self.N_episodes):
			# run exploratory episode
			self.episode=episode

			# set exploration temperature
			self.beta_RL=self.exploration_schedule(episode)

			# run episode
			self.run_episode(self.alpha,greedy=False)

			######## QUANTUM MEASUREMENT ########

			if self.env.model.measure_mode is 'stochastic':

				# update best encountered actions
				if self.episode==0:
					self.actions_best_encountered=self.env.action.visited 
					
				# REPEAT episode for statistics w/o learning
				if self.model.std_error<1E-12 or self.model.std_error>=1E-2:
					for _ in range(self.N_repeat):
						self.repeat_episode()

				# run episode after repeats and evaluate it
				self.run_episode(self.alpha,mode='repeat')
				self.compute_stats(self.episode)


				# REPLAY and learn best encountered policy
				if self.episode%self.replay_frequency==0 and self.episode>0: 

					std_error_best=self.model.registry[0,self.states_best_encountered,2]
					if 1E-12<= std_error_best <= (1E-1 if self.episode/self.N_episodes<0.9 else 1E-2):
					
						for j in range(self.N_replay):
							# replays carried out with unit learning rate
							self.run_episode(1.0,greedy=True,mode='replay')
							self.return_best_encountered = self.env.reward

							#R=self.model.registry[0,self.states_best_encountered,:]
							#print('j',j,R,R[0]/R[1], self.env.model.current_reward)
							
					# print stats
					self.print_stats(mode='replay')
				

			######## CLASSICAL MEASUREMENT ########

			else:

				# evaluate episode
				self.compute_stats(episode)
				
				if self.episode%self.replay_frequency==0 and self.episode>0:
					for j in range(self.N_time_steps):
						self.run_episode(self.alpha,greedy=True,mode='replay')
						self.return_best_encountered = self.env.reward 

					# print stats
					self.print_stats(mode='replay')

		##### TEST STAGE #####
		# import test model params
		self.model.initial_state_noise=params_QL['test_model_params']['initial_state_noise']
		self.model.stochastic_env=params_QL['test_model_params']['stochastic_env']

		# run w/o exploration to get the Q function convergent
		for episode in range(self.N_greedy_episodes):
			greedy_episode = episode + self.N_episodes
			self.run_episode(self.alpha,greedy=True)
			self.compute_stats(greedy_episode)


	
	def run_episode(self,alpha,greedy=False,mode=''):
		""" Runs an episode of the Q Learning algorithm """
	
		self.initialize_episode()

		# compute Q function in initial state for all actions
		Q=self.Q_tabular[0,self.env.state.current,:] # Q=self.evaluate_Q(self.feature_inds.current,time_step=0)
		
		delta_t=0.0
		# episode 
		for time_step in range(self.N_time_steps):
			
			last_time_step_bool=time_step == self.N_time_steps-1

			# pick action
			indA=self.choose_action(Q,time_step,greedy,mode=mode)
			
			# take action, observe new state and reward
			self.take_action(indA,time_step,last_time_step_bool)


			###############################################################
			#####   from now on: state.current --> state.previous   #######
			###############################################################
			

			# fire trace for observed (s,a) pair
			self.trace[time_step,self.env.state.previous,indA]=alpha #self.trace[time_step,self.feature_inds.previous,indA]=self.alpha*self.fire_trace
			
			# Q learning update rule; GD error in time
			delta_t = - Q[indA] # error in time


			# check if new state is terminal
			if last_time_step_bool: 
				delta_t += self.env.reward
				self.Q_tabular += delta_t*self.trace
				break

			### move on to next state ###
			
			
			# t-dependent Watkins Q learning
			Q=self.Q_tabular[time_step+1,self.env.state.current,:] #Q=self.evaluate_Q(self.feature_inds.current,time_step+1)
			
			# update GS error in time
			delta_t += np.max(Q) # Bellman error
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
		

	def repeat_episode(self):
		""" Runs a sequence of taken actions to gains statistics. """
		self.env.model.reset()
		self.env.model.cost_function(self.env.state.current,self.env.action.visited)

	
	def choose_action(self,Q,time_step,greedy=False,mode=''):
		""" Chooses action from current Q function according to Boltzmann exploration. """

		if mode is 'repeat':
			A = self.actions_taken[time_step]
			indA = self.env.actions_dict[A]
		
		elif mode is '':

			# get indices of all available actions from current state of environment
			avail_actions,avail_inds=self.env.available_actions()

			# greedy action is any random action our of the available actions which maximize Q
			inds_max=np.argwhere(Q[avail_inds]==np.amax(Q[avail_inds])).flatten()
			A_greedy = avail_actions[random.choice(inds_max)]
			indA_greedy=self.env.actions_dict[A_greedy]
			### eps-greedy exploration
			if not greedy or time_step%2==1:
				random_num=random.uniform(0.0,1.0)
				if random_num>self.beta_RL/self.beta_RL_f:
					A = random.choice(avail_actions)
				else:
					A = A_greedy
			else:
				# pick greedy action
				A = A_greedy
		
			# find the index of A
			indA = self.env.actions_dict[A]

			# reset traces if A is exploratory
			if indA is not indA_greedy:
				self.trace.reset()
			
		elif mode is 'replay':
			A = self.actions_best_encountered[time_step]
			indA = self.env.actions_dict[A]

		return indA

	def take_action(self,indA,time_step,last_time_step_bool):
		# move environment one step forward
		self.env.take_action(indA,last_time_step_bool)
		self.actions_taken[time_step]=self.env.action.current
		
	

	def initialize_algorithm(self):

		self._create_Q_tabular()
		self._create_trace()


		self.actions_taken=np.zeros((self.N_time_steps,),dtype=np.float64)
		self.actions_best_encountered=np.zeros((self.N_time_steps,),dtype=np.float64)

		# expected return
		self.return_best_encountered=-1.0
		self.states_best_encountered=0
		
		self.episodic_return=np.zeros((self.N_iterations,),dtype=np.float64)
		self.episodic_return_true=np.zeros_like(self.episodic_return)
		self.return_running_ave=np.zeros_like(self.episodic_return)
		self.return_true_ave=np.zeros_like(self.episodic_return)
		
		# protocols stored as integers
		self.episodic_protocol=np.zeros((self.N_iterations,),dtype='|S2')
		

	def _create_Q_tabular(self):
		# object of custom algebraic_disct class
		self.Q_tabular=algebraic_dict(self.N_time_steps,self.N_actions) 

	def _create_trace(self):
		"""eligibility trace used for credit assignment, c.f. algorithm TD(lambda)"""
		self.trace=algebraic_dict(self.N_time_steps,self.N_actions) 


	def initialize_episode(self):
		""" Initializes learning episode """

		# reset environment
		self.env.reset()
		# reset trace and usage vector
		self.trace.reset()

		

	def compute_stats(self,episode):
		"""Computes average reward, running reward, etc."""

		# store current return
		self.episodic_return[episode]=self.env.reward
		self.episodic_return_true[episode]=self.env.model.current_reward

		# compute running average of return
		self.return_running_ave[episode]=1.0/(episode+1)*(self.env.reward + episode*self.return_running_ave[episode-1])
		self.return_true_ave[episode]=1.0/(episode+1)*(self.env.model.current_reward + episode*self.return_true_ave[episode-1])
		
		# store current protocol
		self.episodic_protocol[episode]=self.env.state.current

		# update best encountered quantities
		if self.model.measure_mode is 'deterministic':
			self.update_best_encountered_classical()
		else:
			self.update_best_encountered_quantum()

	def update_best_encountered_quantum(self):

		#print(self.return_best_encountered, self.env.reward, self.return_best_encountered < self.env.reward)
		if self.return_best_encountered < self.env.reward and 1E-12 <= self.model.std_error <= (1E-1 if self.episode/self.N_episodes<0.9 else 1E-2) or self.episode is 0:

			self.return_best_encountered = self.env.reward 
			self.actions_best_encountered = self.env.action.visited

			self.states_best_encountered=self.env.state.current 

	def update_best_encountered_classical(self):

		if self.return_best_encountered < self.env.reward or self.episode is 0:

			self.return_best_encountered = self.env.reward 
			self.actions_best_encountered = self.env.action.visited

			self.states_best_encountered=self.env.state.current 

	###### print and plot functions ######
	def print_stats(self,mode=None):
		print(str(mode)+"_ep=%i_r=%0.4f_rbest=%0.4f_p=%0.4f_betaRL=%0.2f_samples=%i_E=%0.3f" 
			%(self.episode, 
				self.env.reward, 
				self.return_best_encountered, 
				self.env.model.current_reward, 
				self.beta_RL, #self.eps, 
				self.model.registry[0,self.env.state.current,1],
				self.model.registry[0,self.env.state.current,2],
			 )
			)
		print()
	

	def plot_stats(self):

		from matplotlib import pylab as plt


		plt.plot(range(self.N_iterations),self.episodic_return,'.r',markersize=1.0,label='data')
		plt.plot(range(self.N_iterations),self.return_running_ave,'b',label='running ave')
		plt.plot(range(self.N_iterations),self.return_true_ave,'g',label='true fid ave')
		plt.plot(range(self.N_episodes),1.0/self.beta_RL_f*self.exploration_schedule(np.arange(self.N_episodes)),'y',label='exploration scheme')
		plt.xlim([0,self.episode+self.N_greedy_episodes])
		plt.legend(loc='lower right')
		plt.show()



