import numpy as np 
from environment import environment

import sys,os
quspin_path = os.path.join("../")
sys.path.insert(0,quspin_path)

from model import model


T=2.0 #float(sys.argv[4]) # protocol duration
N_time_steps=40 # protocol time steps
t_evolve=np.linspace(T/N_time_steps,T,N_time_steps) # protocol time

allowed_states=np.sort([-4.0,4.0])

# define model params
params_model=dict(
	L=1, # system size
	J=-1.0, # zz interaction strength
	hx=-1.0, # x field strength
	hz=-1.0, # x field strength
	t_evolve=t_evolve, # evolution duration
	basis_kwargs=dict(pauli=False), # basis keayward arguments
	allowed_states=allowed_states,
	measure_mode='quantum', # 'classical' #
	initial_state_noise=0, #int(sys.argv[2]), # noisy initial state
	noise_level=0.1, # noise in initial state
	stochastic_env=0,#int(sys.argv[3]), # stochastic actions
	stochasticity_level=1.0/N_time_steps # probability to take random action
	)


model=model(params_model)

env=environment(model)

all_actions, avail_inds = env.available_actions()
print(all_actions,avail_inds)





