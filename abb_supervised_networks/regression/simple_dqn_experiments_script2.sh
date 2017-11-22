#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/experiments/simple_dqn/



gpu_memory_usage=0.4
image_h=84
image_w=84
val_steps=6000
epochs=30


for img_num in 2
do
for keep_prob in 0.5
do
for batch_size in 32
do
for bal in False
do
for loss_function in abs
do
for abs_max_weight in 0.5
do
for network_architecture in regression_simple_dqnnet
do
for output_layer in identity
do
for img_std in False
do
for optimizer in adam
do
python ${path}nn_regression_low_memory_train_validation_fine.py   --num_epochs=${epochs} --pretrained_checkpoint_path=/home/dladmin/Documents/arthurma/experiments/simple_dqn/regression_simple_dqnnet_identity_in-2_ih-84_bs-32_kp-0.5_bal-False_is-False_lf-abs_maxw-0.5/-178584 --train_dir=${output_path}${network_architecture}_${output_layer}_in-${img_num}_ih-${image_h}_bs-${batch_size}_kp-${keep_prob}_bal-${bal}_is-${img_std}_lf-${loss_function}_maxw-${abs_max_weight}_2/ --optimizer=${optimizer} --validation_every_n_steps=${val_steps} --balance_training_data=${bal}  --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
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


