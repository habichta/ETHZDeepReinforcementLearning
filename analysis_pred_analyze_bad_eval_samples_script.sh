path=/home/dladmin/Documents/arthurma/shared/dlabb


mpc_dir=/home/dladmin/Documents/arthurma/experiments/simple_dqn_eval


for entry in "$mpc_dir"/*
do
   var="$var $entry"
done


python ${path}/analysis_pred_analyze_bad_eval_samples.py $var

