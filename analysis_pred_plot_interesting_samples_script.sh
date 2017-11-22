path=/home/dladmin/Documents/arthurma/shared/dlabb


mpc_dir=/home/dladmin/Documents/arthurma/experiments/simple_resnetv218_eval


for entry in "$mpc_dir"/*
do
   python ${path}/analysis_pred_perform_mpc.py $entry
done





