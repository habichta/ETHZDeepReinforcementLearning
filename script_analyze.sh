#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/
output_path=/home/dladmin/Documents/arthurma/runs/RESNET8_irr_nopool_absmax_84_EVAL/


#regression_simple_dqnnet


gpu_memory_usage=0.2
image_h=84
image_w=84
val_steps=12000
#val_steps=2000

#--pretrained_checkpoint_path=${cpk_path}
#--dqn_keep_prob=${keep_prob}

#save_checkpoint_steps=13695
#--save_checkpoint_steps=13695


for cpk_dir in 44646 89292 133938 200907 267876 290199
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
for output_layer in linear11_irr
do
for pool_layer in False
do
for diff in False
do
for img_std in False
do
for img_mean in False
do
for print_outlier in 200
do
dir=${output_path}simple_${img_num}_bal${bal}_type${bal_type}_${network_architecture}_ol${output_layer}_kp${keep_prob}_${batch_size}_diff${diff}_ch${img_ch}_is${img_std}_im${img_std}_pool_layer${pool_layer}_lf${loss_function}_maxw${abs_max_weight}_outl${print_outlier}_${cpk_dir}/
python ${path}pred_visualize_interesting_samples.py ${dir}
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
