#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/experiments/simple_resnetv218_eval/


gpu_memory_usage=0.3
image_h=84
image_w=84

#############ABS##########

for cpk_path in /home/dladmin/Documents/arthurma/experiments/simple_resnetv218/regression_resnetV218_bottleneck_linear11_in-2_ih-84_bs-32_kp-0.5_bal-False_is-False_lf-abs_maxw-0.5/-
do
for cpk_dir in 66969 178584 379491
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 1
do
for bal in False
do
for loss_function in abs
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218_bottleneck
do
for output_layer in linear11
do
for img_std in False
do
for print_outlier in 200
do

e_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}_${cpk_dir}/

python ${path}nn_regression_prediction_low_memory_fine.py   --cpk_dir=${cpk_path}${cpk_dir} --eval_dir=${e_dir} --print_outlier=${print_outlier} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture}  --output_layer=${output_layer} --test_batch_size=${batch_size}  --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std}  --per_process_gpu_memory_fraction${gpu_memory_usage}

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


#############ABS_MAX##########

for cpk_path in /home/dladmin/Documents/arthurma/experiments/simple_resnetv218/regression_resnetV218_bottleneck_linear11_in-2_ih-84_bs-32_kp-0.5_bal-False_is-False_lf-abs_max_maxw-0.5/-
do
for cpk_dir in 89292 223230 379491
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 1
do
for bal in False
do
for loss_function in abs_max
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218_bottleneck
do
for output_layer in linear11
do
for img_std in False
do
for print_outlier in 250
do

e_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}_${cpk_dir}/

python ${path}nn_regression_prediction_low_memory_fine.py   --cpk_dir=${cpk_path}${cpk_dir} --eval_dir=${e_dir} --print_outlier=${print_outlier} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture}  --output_layer=${output_layer} --test_batch_size=${batch_size}  --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std}  --per_process_gpu_memory_fraction${gpu_memory_usage}

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


#############MSE##########

for cpk_path in /home/dladmin/Documents/arthurma/experiments/simple_resnetv218/regression_resnetV218_bottleneck_linear11_in-2_ih-84_bs-32_kp-0.5_bal-False_is-False_lf-abs_max_maxw-0.5/-
do
for cpk_dir in 89292 223230 491106
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 1
do
for bal in False
do
for loss_function in mse
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218_bottleneck
do
for output_layer in linear11
do
for img_std in False
do
for print_outlier in 50000
do

e_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}_${cpk_dir}/

python ${path}nn_regression_prediction_low_memory_fine.py   --cpk_dir=${cpk_path}${cpk_dir} --eval_dir=${e_dir} --print_outlier=${print_outlier} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture}  --output_layer=${output_layer} --test_batch_size=${batch_size}  --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std}  --per_process_gpu_memory_fraction${gpu_memory_usage}

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
