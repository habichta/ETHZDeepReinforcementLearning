path=/home/dladmin/Documents/arthurma/shared/dlabb/
rl_path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_rl_algorithms/DDDQN/

cpk_path=/home/dladmin/Documents/arthurma/rf/run_hl_mse_tau_00001_e_100_followirr_guidedexp08_sunspot_action7_hard_nopt/-
eval_path=/home/dladmin/Documents/arthurma/rf/eval/run_hl_mse_tau_00001_e_100_followirr_guidedexp08_sunspot_action7_hard_nopt_



for cpk in 45738
do
   eval_dir=${eval_path}${cpk}_TRAIN
   cpk_dir=${cpk_path}${cpk}
   python ${rl_path}testing_simple.py --eval_dir=${eval_dir} --pretrained_checkpoint_path=${cpk_dir}
   python ${path}reinforce_pred_create_overall_energy_throughput.py ${eval_dir} 0.0
done





