#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT4/
cpk_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT2/simple_False_2_regression_simple_dqnnet_do_32/-89292


gpu_memory_usage=0.3
image_h=128
image_w=128
val_steps=12000



for img_ch in 3
do
for img_num in 2
do
for keep_prob in 0.5
do
for batch_size in 32
do
for bal in True False
do
for network_architecture in  regression_simple_dqnnet_do
do
for diff in False
do

if [ "$bal" == "True" ]
then
cpk_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT41/simple_2_True_regression_simple_dqnnet_do_kp0.5_32_diffFalse_ch3/-200907
output_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT42/
else
cpk_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT2/simple_False_2_regression_simple_dqnnet_do_32/-89292
output_path=/home/dladmin/Documents/arthurma/runs/SIMPLE_SCRIPT4/
fi

python ${path}nn_regression_low_memory_train_validation.py --pretrained_checkpoint_path=${cpk_path} --validation_every_n_steps=${val_steps} --num_epochs=30 --train_dir=${output_path}simple_${img_num}_${bal}_${network_architecture}_kp${keep_prob}_${batch_size}_diff${diff}_ch${img_ch}/ --balance_training_data=${bal} --image_num_per_sample=${img_num} --network_architecture=${network_architecture} --train_batch_size=${batch_size} --difference_images=${diff} --dqn_keep_prob=${keep_prob}  --image_channels=${img_ch} --image_height_resize=${image_h} --image_width_resize=${image_w} --per_process_gpu_memory_fraction${gpu_memory_usage}
done
done
done
done
done
done
done

