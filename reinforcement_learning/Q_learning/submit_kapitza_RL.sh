#!bin/bash

let 'target_state = 0' # system size 0: Floquet, 1: quasi-Gaussian
let 'test_noise = 0' # initial state noise during test stage
let 'test_stoch_env = 1' # random actions during test stage

for stoch_env in 0 1
do
	test_stoch_env=${stoch_env}

	for noise in 0 1
	do
		test_noise=${noise}

		for N_periods in 15 # 3 15
		do
			for N_time_steps_per_period in 8 # quantum (1) vs classical (0) measurement
			do
				for meas_mode_int in 1 # 0 1 # quantum (1) vs classical (0) measurement
				do
					for Omega in 10.0 #6.0 8.0 10.0 12.0 14.0 # quantum (1) vs classical (0) measurement
					do

						echo "#!/bin/bash -l" >> submission.sh

						echo "#$ -t 1-100" >> submission.sh # ask for jobs in the qsub array: determines range of SGE_TASK_ID (inclusive)
						echo "#$ -P fheating" >> submission.sh
						echo "#$ -N jobRL_${N_periods}_${N_time_steps_per_period}" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
						echo "#$ -l h_rt=72:00:00" >> submission.sh 
						echo "#$ -m n" >> submission.sh

						echo ~/.conda/envs/quspin/bin/python main_kapitza.py \${SGE_TASK_ID} ${Omega} ${N_time_steps_per_period} ${N_periods} ${meas_mode_int} ${noise} ${stoch_env} ${target_state} ${test_noise} ${test_stoch_env} >> submission.sh

						qsub submission.sh
						rm submission.sh

					done
				done
			done
		done
	done
done

