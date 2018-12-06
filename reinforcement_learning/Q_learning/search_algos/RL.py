import numpy as np
import pickle
import matplotlib.pyplot as plt
import os,sys

#from Q_learning_cython import Q_learning
from Q_learning_python import Q_learning



def apply_RL(model,params_QL,params_model,params_tabular,save=False,file_name=''):

	# apply tabular Q-Learning algorithm
	QL=Q_learning(model.seed,model,params_QL,params_tabular)


	# read in local directory path
	str1=os.getcwd()
	str2=str1.split('\\')
	n=len(str2)
	my_dir = str2[n-1]

	# check if directory exists and create it if not
	initial_state_noise = params_QL['test_model_params']['initial_state_noise']
	stochastic_env=params_QL['test_model_params']['stochastic_env']
	
	save_dir = my_dir+"/data/test_noise-{0:d}_stochenv-{1:d}/".format(initial_state_noise,stochastic_env)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	if save:
		pickle.dump([QL,params_QL,params_model,params_tabular], open(save_dir+file_name, "wb" ) )

