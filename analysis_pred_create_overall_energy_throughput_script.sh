path=/home/dladmin/Documents/arthurma/shared/dlabb


mpc_dir=/home/dladmin/Documents/arthurma/experiments/simple_resnetv218_eval


for entry in "$mpc_dir"/*
do
   python ${path}/analysis_pred_create_overall_energy_throughput.py $entry 100 2 0.0
done


mpc_dir=/home/dladmin/Documents/arthurma/experiments/simple_dqn_eval


for entry in "$mpc_dir"/*
do
   python ${path}/analysis_pred_create_overall_energy_throughput.py $entry 100 2 0.0
done



