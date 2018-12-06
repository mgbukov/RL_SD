import numpy as np
import pickle
from models import model_kapitza_quantum, model_kapitza_classical
import sys,os

from quspin.tools.Floquet import Floquet_t_vec
from search_algos import apply_RL

###########################################

save=False #True
model_type='quantum' #'classical' #


seed=int(sys.argv[1])  # random number generator seed

Omega=float(sys.argv[2]) # 6*2*np.pi # drive frequency

N_time_steps_per_period=int(sys.argv[3]) # time steps per period: muliple of 4
N_periods=int(sys.argv[4]) # number of periods
t_evolve=Floquet_t_vec(Omega,N_periods,len_T=N_time_steps_per_period)  #np.linspace(T/N_time_steps,T,N_time_steps) # protocol time

N_time_steps=N_time_steps_per_period*N_periods # protocol time steps

# allowed bang sizes
h_max=4.0
allowed_states=np.sort([-h_max,0.0,h_max]) # Kapitza

meas_mode_int=int(sys.argv[5]) # toggles quantum (1) vs classical (0) measurement

# action dictionary: encodes action values as strings from which RL states are built
as_dict={-h_max:bytes('-','utf-8'),0.0:bytes('0','utf-8'),h_max:bytes('+','utf-8')}

# set noise level in initial state
if model_type=='quantum':
	eta=0.31
elif model_type=='classical':
	eta=0.1

# define model params
params_model=dict( 
	### Kapitza
	L=21, # number of states
	mw=1.0, # mass times bare frequency
	Omega=Omega, # drive frequency
	A=2.0, # drive amplitude
	basis_kwargs=dict(sps=2,Nb=1), # basis arguments
	inf_frequency=0, # toggles infinite vs finite frequency
	
	### general
	t_evolve=t_evolve, # evolution duration
	allowed_states=allowed_states,
	measure_mode=['deterministic', 'stochastic'][meas_mode_int],
	initial_state_noise=int(sys.argv[6]), # noisy initial state
	noise_level=eta, # noise in initial state is sqrt(noise_level): psi <-- psi + noise_level*random_perturbation
	stochastic_env=int(sys.argv[7]), # stochastic actions
	stochasticity_level=1.0/N_time_steps, # probability to take random action
	target_state=['Floquet','Gauss'][int(sys.argv[8])], # target state: inverted exact Floquet eigenstate or Gaussian at inverted position,
	model_type=model_type, # classical vs quantum model
	)


# define tabular params
params_tabular=dict(
	N_time_steps=N_time_steps,
	N_actions=len(allowed_states),
	)


# define Q-learning params
params_QL=dict(
	N_episodes=2000, # train agent
	N_greedy_episodes=100, # test agent: episodes after learning is over
	test_model_params=dict(initial_state_noise=int(sys.argv[6]),stochastic_env=int(sys.argv[7])), # model parameters for test stage
	N_repeat=50, # repetition episodes (for quantum measurements only)
	replay_frequency=50, # every so many episodes replays are performed 
	N_replay=100, # replay episodes to facilitate learning
	alpha=0.1, # learning rate
	lmbda=0.6, # TD(lambda)
	beta_RL_i=10, # initial inverse RL temperature (annealed eps-greedy exploration!)
	beta_RL_f=50, # final inverse RL temperature (annealed eps-greedy exploration!)
	action_state_dict=as_dict, # actions string precision
	exploration_schedule='exp', # exploration schedule
	)

def exploration_schedule(episode):
		# linear
		#return (params_QL['beta_RL_f'] - params_QL['beta_RL_i'])*episode/params_QL['N_episodes'] + params_QL['beta_RL_i']
		# sqrt
		#return np.sqrt( (params_QL['beta_RL_f']**2 - params_QL['beta_RL_i']**2)*episode/params_QL['N_episodes'] + params_QL['beta_RL_i']**2 )
		# cubic root
		#return ( (params_QL['beta_RL_f']**3 - params_QL['beta_RL_i']**3)*episode/params_QL['N_episodes'] + params_QL['beta_RL_i']**3 )**(1.0/3.0)
		# quintic root
		#return ( (params_QL['beta_RL_f']**5 - params_QL['beta_RL_i']**5)*episode/params_QL['N_episodes'] + params_QL['beta_RL_i']**5 )**(1.0/5.0)
		# logarithmic
		#return np.log( ( np.exp(params_QL['beta_RL_f']) - np.exp(params_QL['beta_RL_i']) )*episode/params_QL['N_episodes'] + np.exp(params_QL['beta_RL_i']) )
		# exponential
		return  (params_QL['beta_RL_i'] - params_QL['beta_RL_f'])*np.exp(-10*episode/params_QL['N_episodes'] ) + params_QL['beta_RL_f']



params_QL['exploration_schedule']=exploration_schedule


########################################


# create model
if model_type=='quantum':
	model=model_kapitza_quantum(seed,params_model)
elif model_type=='classical':
	model=model_kapitza_classical(seed,params_model)
else:
	raise NotImplementedError
model_data='_target-'+params_model['target_state']+'_inffreq=%i_Nperiods=%i_Omega=%0.2f_A=%0.2f_hmax=%0.2f_L=%i'%(params_model['inf_frequency'],N_periods,params_model['Omega'],params_model['A'],h_max,params_model['L'])

################
# run algorithm
################

# RL
save_tuple=(seed,int(params_model['stochastic_env']),int(params_model['initial_state_noise']),params_QL['N_episodes'],N_time_steps)
save_file='QL_data-'+model_type+'_kapitza-'+params_model['measure_mode']+'-seed=%i_stochenv_%i_noise_%i_Nepisodes=%i_nsteps=%i'%save_tuple+model_data+'.pkl'

apply_RL(model,params_QL,params_model,params_tabular,save=save,file_name=save_file)


