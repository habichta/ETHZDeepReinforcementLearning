#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/experiments/simple_dqn/



gpu_memory_usage=1.0
image_h=84
image_w=84
val_steps=6000
epochs=25


for img_num in 2
do
for keep_prob in 0.5
do
for batch_size in 32
do
for bal in False
do
for loss_function in abs abs_max mse
do
for abs_max_weight in 0.5
do
for network_architecture in regression_simple_dqnnet regression_simple_dqnnet_do
do
for output_layer in identity
do
for img_std in False
do
for optimizer in adam
do
python ${path}nn_regression_prediction_low_memory_fine.py   --train_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}/ --optimizer=${optimizer} --validation_every_n_steps=${val_steps} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture}  --output_layer=${output_layer} --train_batch_size=${batch_size}  --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std}  --per_process_gpu_memory_fraction${gpu_memory_usage}
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


output_path=/home/dladmin/Documents/arthurma/experiments/simple_resnetv218/

image_h=84
image_w=84
val_steps=12000
epochs=25


for img_num in 2
do
for keep_prob in 0.5
do
for batch_size in 32
do
for bal in False
do
for loss_function in abs abs_max mse
do
for abs_max_weight in 0.5
do
for network_architecture in regression_resnetV218
do
for output_layer in linear11
do
for img_std in False
do
for optimizer in adam
do
python ${path}nn_regression_low_memory_train_validation_fine.py   --num_epochs=${epochs} --train_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}/ --optimizer=${optimizer} --validation_every_n_steps=${val_steps} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture}  --output_layer=${output_layer} --train_batch_size=${batch_size}  --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std}  --per_process_gpu_memory_fraction${gpu_memory_usage}
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
