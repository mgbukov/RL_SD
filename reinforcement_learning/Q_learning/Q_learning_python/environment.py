import numpy as np


class RL_object(object):

	def __init__(self):

		self.initial=None
		self.previous=None
		self.current=None
		self.visited=[]


class environment(object):
	def __init__(self,model,action_state_dict):

		# create physical model
		self.model=model
		# actions-state dictionary
		self.action_state_dict=action_state_dict
		# reset environment
		self.reset()
		# compute actions dictionary
		self._init_environment()

		
	def _init_environment(self):

		self.avail_inds=range(len(self.all_actions))

		self.actions_dict={}
		for i,a in enumerate(self.all_actions):
			self.actions_dict[a]=i
	
	def available_actions(self,lower=None,upper=None):
		""" Compute indices of available actions from current state"""
		'''
		avail_inds=np.where(np.logical_and(self.action.current+self.all_actions>=lower,
											self.action.current+self.all_actions<=upper) 
											)[0]
		avail_actions=self.all_actions[avail_inds]
		'''
		
		return self.all_actions, self.avail_inds

	def reset(self,visited=True):
		# resets environment
		self.model.reset()

		self.state=RL_object()
		self.action=RL_object()
		
		self.state.initial=bytes('', 'utf-8') 
		self.state.current=self.state.initial

		self.action.current=0.0

		self.reward=0.0

		self.all_actions=self.model.allowed_states
	
		self.episode_step=0

		#self.protocol_str=bytes('', 'utf-8')


	def take_action(self,indA,last_time_step_bool):

		action=self.all_actions[indA]
		
		# update previous
		self.state.previous=self.state.current
		self.action.previous=self.action.current

		# update current
		self.action.current=action

		# environment's reaction
		self._react()

		# update visited lists
		self.action.visited.append(action)
		self.state.visited.append(self.state.current)

		# evaluate reward
		if last_time_step_bool:
			self.reward=self.model.cost_function(self.state.current,self.action.visited)

		# increment time step
		self.episode_step+=1


	def _react(self):
		""" reaction of environment and computation of reward. """

		# update state

		self.state.current+=self.action_state_dict[self.action.current]


		'''
		protocol_str=str(int( not np.signbit(self.action.current) )) #str(int(self.action.current>0))
		
		self.protocol_str=protocol_str+self.protocol_str
		#self.protocol_str=self.protocol_str+protocol_str
		
		self.state.current=int(self.protocol_str,2) 
		'''

	
	