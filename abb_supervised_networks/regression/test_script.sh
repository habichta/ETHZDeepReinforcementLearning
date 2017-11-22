#!/bin/sh

path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/runs/RESNET10_irr_nopool_mse_fine_EVAL/

dlabb_p=/home/dladmin/Documents/arthurma/shared/dlabb/


#cpk_path=/home/dladmin/Documents/arthurma/runs/RESNET10_irr_nopool_mse_fine/simple_2_balFalse_typeB_regression_resnetV218_irr_ollinear31_irr_kpnodropout_32_diffFalse_ih128_ch3_isFalse_imFalse_pool_layerFalse_lfabs_max_maxw0.5_optadam_2/-
image_h=128
image_w=128


for cpk_path in /home/dladmin/Documents/arthurma/runs/RESNET10_irr_nopool_mse_fine/simple_2_balFalse_typeB_regression_resnetV218_irr_ollinear31_irr_kpnodropout_32_diffFalse_ih128_ch3_isFalse_imFalse_pool_layerFalse_lfabs_max_maxw0.5_optadam_2/-
do
for cpk_dir in 89292 2000907 357168
do
for img_ch in 3
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 1
do
for bal in False
do
for bal_type in B
do
for loss_function in abs_max
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218_irr
do
for output_layer in linear31_irr
do
for pool_layer in False
do
for diff in False
do
for img_std in False
do
for img_mean in False
do
for print_outlier in 40000
do

e_dir=${output_path}simple_${img_num}_bal${bal}_type${bal_type}_${network_architecture}_ol${output_layer}_kp${keep_prob}_${batch_size}_diff${diff}_ih${image_h}_ch${img_ch}_is${img_std}_im${img_std}_pool_layer${pool_layer}_lf${loss_function}_maxw${abs_max_weight}_outl${print_outlier}_${cpk_dir}/

python ${path}nn_regression_prediction_low_memory_fine.py --cpk_dir=${cpk_path}${cpk_dir} --eval_dir=${e_dir} --print_outlier=${print_outlier}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture} --irr_pooling_layer=${pool_layer} --output_layer=${output_layer} --test_batch_size=${batch_size} --difference_images=${diff}  --image_channels=${img_ch} --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std} --image_subtract_mean=${img_mean}

python ${dlabb_p}analysis_pred_perform_mpc.py ${e_dir} 100 1 31


done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done


for cpk_path in /home/dladmin/Documents/arthurma/runs/RESNET10_irr_nopool_mse_fine/simple_2_balFalse_typeB_regression_resnetV218_irr_ollinear31_irr_kpnodropout_32_diffFalse_ih128_ch3_isFalse_imFalse_pool_layerFalse_lfmse_maxw0.5_optadam_2/-
do
for cpk_dir in 89292 2000907 357168
do
for img_ch in 3
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 1
do
for bal in False
do
for bal_type in B
do
for loss_function in mse
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218_irr
do
for output_layer in linear31_irr
do
for pool_layer in False
do
for diff in False
do
for img_std in False
do
for img_mean in False
do
for print_outlier in 40000
do

e_dir=${output_path}simple_${img_num}_bal${bal}_type${bal_type}_${network_architecture}_ol${output_layer}_kp${keep_prob}_${batch_size}_diff${diff}_ih${image_h}_ch${img_ch}_is${img_std}_im${img_std}_pool_layer${pool_layer}_lf${loss_function}_maxw${abs_max_weight}_outl${print_outlier}_${cpk_dir}/

python ${path}nn_regression_prediction_low_memory_fine.py --cpk_dir=${cpk_path}${cpk_dir} --eval_dir=${e_dir} --print_outlier=${print_outlier}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture} --irr_pooling_layer=${pool_layer} --output_layer=${output_layer} --test_batch_size=${batch_size} --difference_images=${diff}  --image_channels=${img_ch} --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std} --image_subtract_mean=${img_mean}

python ${dlabb_p}analysis_pred_perform_mpc.py ${e_dir} 100 1 31


done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

