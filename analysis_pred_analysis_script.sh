path=/home/habichta/dlabb


mpc_dir=/local/habichta/keras_experiment/balanced_16_224_pertrained_with_irradiance_31/eval

set_path=/home/habichta/dlabb/abb_deeplearning_keras/Source_code/models/validation_list.out


for entry in "$mpc_dir"/*
do
   python ${path}/analysis_pred_perform_mpc.py $entry 100 2 31 ${set_path}
   python ${path}/analysis_pred_create_overall_energy_throughput.py $entry 100 2 0.0
   python ${path}/analysis_pred_plot_mpc.py $entry 100 2 --save True --plot False
   python ${path}/analysis_pred_plot_interesting_samples.py $entry
   #bad_var="$bad_var $entry"
done

#python ${path}/analysis_pred_analyze_bad_eval_samples.py $bad_var

#mpc_dir=/local/habichta/keras_experiment/balanced_16_224_pertrained_no_irradiance_mean_mae/eval


#for entry in "$mpc_dir"/*
#do
  # python ${path}/analysis_pred_perform_mpc.py $entry 100 2 11 ${set_path}
  # python ${path}/analysis_pred_create_overall_energy_throughput.py $entry 100 2 0.0
  # python  ${path}/analysis_pred_plot_mpc.py $entry 100 2 --save True --plot False
  # python  ${path}/analysis_pred_plot_interesting_samples.py $entry
  # bad_var="$bad_var $entry"
#done


#python ${path}/analysis_pred_analyze_bad_eval_samples.py $bad_var


