#!/bin/sh


path=/home/dladmin/Documents/arthurma/shared/dlabb/abb_supervised_networks/regression/
output_path=/home/dladmin/Documents/arthurma/runs/RESNET10_irr_nopool_mse_fine/
cpk_path=/home/dladmin/Documents/arthurma/runs/RESNET9_irr_nopool_mse/simple_2_balFalse_typeB_regression_resnetV218_irr_ollinear11_irr_kpnodropout_32_diffFalse_ih128_ch3_isFalse_imFalse_pool_layerFalse_lfmse_maxw0.5_HARD/-312522


#regression_simple_dqnnet


gpu_memory_usage=1.0
image_h=128
image_w=128
val_steps=12000
#val_steps=2000

#--pretrained_checkpoint_path=${cpk_path}
#--dqn_keep_prob=${keep_prob}

#save_checkpoint_steps=13695
#--save_checkpoint_steps=13695

for img_ch in 3
do
for img_num in 2
do
for keep_prob in nodropout
do
for batch_size in 32
do
for bal in False
do
for bal_type in B
do
for loss_function in mse abs_max
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
for optimizer in adam
do
python ${path}nn_regression_low_memory_train_validation_fine.py   --num_epochs=20 --train_dir=${output_path}simple_${img_num}_bal${bal}_type${bal_type}_${network_architecture}_ol${output_layer}_kp${keep_prob}_${batch_size}_diff${diff}_ih${image_h}_ch${img_ch}_is${img_std}_im${img_std}_pool_layer${pool_layer}_lf${loss_function}_maxw${abs_max_weight}_opt${optimizer}_2/ --optimizer=${optimizer} --validation_every_n_steps=${val_steps} --balance_training_data=${bal} --loss_weight_type=${bal_type} --loss_function=${loss_function} --abs_max_weight=${abs_max_weight} \
--image_num_per_sample=${img_num} --network_architecture=${network_architecture} --irr_pooling_layer=${pool_layer} --output_layer=${output_layer} --train_batch_size=${batch_size} --difference_images=${diff}  --image_channels=${img_ch} --image_height_resize=${image_h} --image_width_resize=${image_w} --image_standardization=${img_std} --image_subtract_mean=${img_mean} --per_process_gpu_memory_fraction${gpu_memory_usage}
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

