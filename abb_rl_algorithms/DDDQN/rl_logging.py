

import numpy as np
import datetime as dt
import tensorflow as tf
import os,csv
slim = tf.contrib.slim



def save_statistics(train_writer, episodes_reward_list, episodes_mean_max_q_value_list, episodes_mean_chosen_q_value_list=None, episodes_mean_batch_reward_list=None, episode_mean_action_q_value_list=None,step=1, action_counter=None, set="training", write_path=None):
    print("Epoch statistics for: " + str(set))

    summary = tf.Summary()
    if len(episodes_mean_max_q_value_list) > 0 and len(episodes_reward_list)>0:
        total_mean_reward = sum(episodes_reward_list) / len(episodes_reward_list)
        total_median_reward = np.median(np.array(episodes_reward_list))
        total_std_reward = np.std(np.array(episodes_reward_list))
        total_perc_reward_list = list()
        for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
            total_perc_reward_list.append(np.percentile(episodes_reward_list,q=i))


        total_mean_max_q_value = sum(episodes_mean_max_q_value_list) / len(episodes_mean_max_q_value_list)
        total_median_max_q_value = np.median(np.array(episodes_mean_max_q_value_list))
        total_std_max_q_value = np.std(np.array(episodes_mean_max_q_value_list))
        total_perc_max_q_value_list = list()
        for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
            total_perc_max_q_value_list.append(np.percentile(episodes_mean_max_q_value_list, q=i))

        summary.value.add(tag=str(set) + "_total_mean_step_reward", simple_value=total_mean_reward)
        summary.value.add(tag=str(set) + "_total_median_step_reward", simple_value=total_median_reward)
        summary.value.add(tag=str(set) + "_total_std_step_reward", simple_value=total_std_reward)
        summary.value.add(tag=str(set) + "_total_mean_max_step_q", simple_value=total_mean_max_q_value)
        summary.value.add(tag=str(set) + "_total_median_max_step_q", simple_value=total_median_max_q_value)
        summary.value.add(tag=str(set) + "_total_std_max_step_q", simple_value=total_std_max_q_value)


        if episodes_mean_batch_reward_list is not None:
            total_mean_batch_reward_value = sum(episodes_mean_batch_reward_list) / len(episodes_mean_batch_reward_list)
            total_median_batch_reward_value = np.median(np.array(episodes_mean_batch_reward_list))
            total_std_batch_reward_value = np.std(np.array(episodes_mean_batch_reward_list))
            total_perc_batch_reward_value_list = list()
            for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                total_perc_batch_reward_value_list.append(np.percentile(episodes_mean_batch_reward_list, q=i))

            summary.value.add(tag=str(set) + "_total_mean_step_batch_reward",
                              simple_value=total_mean_batch_reward_value)
            summary.value.add(tag=str(set) + "_total_median_step_batch_reward",
                              simple_value=total_median_batch_reward_value)
            summary.value.add(tag=str(set) + "_total_std_step_batch_reward", simple_value=total_std_batch_reward_value)

        if episodes_mean_chosen_q_value_list is not None:
            total_mean_chosen_q_value = sum(episodes_mean_chosen_q_value_list) / len(episodes_mean_chosen_q_value_list)
            total_median_chosen_q_value = np.median(np.array(episodes_mean_chosen_q_value_list))
            total_std_chosen_q_value = np.std(np.array(episodes_mean_chosen_q_value_list))
            total_perc_chosen_q_value_list = list()
            for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                total_perc_chosen_q_value_list.append(np.percentile(episodes_mean_chosen_q_value_list, q=i))

            summary.value.add(tag=str(set) + "_total_mean_chosen_step_q", simple_value=total_mean_chosen_q_value)
            summary.value.add(tag=str(set) + "_total_median_chosen_step_q", simple_value=total_median_chosen_q_value)
            summary.value.add(tag=str(set) + "_total_std_chosen_step_q", simple_value=total_std_chosen_q_value)

        if episode_mean_action_q_value_list is not None:
            total_mean_action_q_values = np.mean(np.array(episode_mean_action_q_value_list),axis=0)
            total_median_action_q_values = np.median(np.array(episode_mean_action_q_value_list),axis=0)

            for i in range(total_mean_action_q_values.shape[0]):
                summary.value.add(tag=str(set) + "_mean_action_{}_step_q".format(str(i)), simple_value=total_mean_action_q_values[i])
                summary.value.add(tag=str(set) + "_median_action_{}_step_q".format(str(i)), simple_value=total_median_action_q_values[i])

        print("#### Reward:")
        print("Total mean step reward:", total_mean_reward)
        print("Total median step reward:", total_median_reward)
        print("Total std. step reward:", total_std_reward)
        print("Total step reward percentiles:", str(total_perc_reward_list))
        print("#### Maximum Action-Value (Q):")
        print("Total mean max step Q:", total_mean_max_q_value)
        print("Total median max step Q:", total_median_max_q_value)
        print("Total std. max step Q:", total_std_max_q_value)
        print("Total step Q max percentiles:", str(total_perc_max_q_value_list))


        if action_counter is not None:
            for elem,cnt in action_counter.items():
                summary.value.add(tag=str(set) + "_action_count_"+str(elem), simple_value=cnt)


        train_writer.add_summary(summary, int(step))
        train_writer.flush()


        if write_path is not None:
            with open(os.path.join(write_path, str(set)+"_"+str(step)+"_results.csv"), "w+") as f:
                w = csv.writer(f)
                w.writerow([str(dt.datetime.now())])
                for key, val in [("Total mean step reward:", str(total_mean_reward)),("Total median step reward:",str(total_median_reward)),("Total median step reward:", str(total_median_reward)),("Total step reward percentiles:", str(total_perc_reward_list)),("Total mean step Q:", str(total_mean_max_q_value)),("Total median step Q:", str(total_median_max_q_value)),("Total std. step Q:", str(total_std_max_q_value)),("Total step Q percentiles:", str(total_perc_max_q_value_list))]:
                    w.writerow([key, val])

    else:
        print("No logging since either reward and q-value list were empty")



def step_log(train_writer,epoch,episode_nr, num_episode_per_epoch,total_steps,episode_steps,total_loss,reward, mean_max_q_value, mean_chosen_q_value, sec_per_batch,learning_rate,epsilon):
    time_now = str(dt.datetime.now())

    try:
        print(
            "####################################################################################")
        print("Time: " + time_now + ", Epoch: " + str(epoch) + ", Episode Nr/Next Epoch: " + str(episode_nr)+"/"+str(epoch*num_episode_per_epoch) + ", Total steps:" + str(total_steps) + ", Episode step: " + str(episode_steps) + \
               ", Minibatch Huberloss: " + \
              "{:.6f}".format(total_loss) + ", Step Reward: " + str(reward)  +", Mean Max-Action-Value (Q): " +str(mean_max_q_value)+", Mean Chosen-Action-Value: "+str(mean_chosen_q_value) + ", Epsilon: " +str(epsilon) +
              ", sec/Batch: " + "{:.2f}".format(
                  sec_per_batch))
    except Exception as e:
        print("Error printing information")

    summary = tf.Summary()
    summary.value.add(tag="learning_rate", simple_value=float(learning_rate))
    summary.value.add(tag="epsilon", simple_value=epsilon)
    summary.value.add(tag="total_loss", simple_value=total_loss)
    train_writer.add_summary(summary, int(total_steps))




def _save_gradient_stats(train_writer,gradients,learning_rate,step):


    ratio_statistics = list()
    grad_norm_statistics=list()
    grad_mean_statistics=list()
    grad_max_statistics = list()
    for grad, var in gradients:
        grad_step = np.linalg.norm(grad*-learning_rate)
        var_norm = np.linalg.norm(var)
        if var_norm > 0:
            wg_ratio = grad_step / var_norm
            ratio_statistics.append((wg_ratio))
            grad_norm_statistics.append(np.linalg.norm(grad))
            grad_mean_statistics.append(np.mean(grad))
            grad_max_statistics.append(np.max(grad))

    mean_wg_ratio = sum(ratio_statistics) / len(ratio_statistics)
    median_wg_ratio = np.median(ratio_statistics)
    max_wg_ratio = max(ratio_statistics)
    min_wg_ratio = min(ratio_statistics)

    max_grad_norm = max(grad_norm_statistics)
    mean_grad_norm = sum(grad_norm_statistics)/len(grad_norm_statistics)
    median_grad_norm = np.median(grad_norm_statistics)
    min_grad_norm = min(grad_norm_statistics)

    mean_grad = np.mean(grad_mean_statistics)
    max_grad =np.max(grad_max_statistics)
    min_grad = np.max(grad_max_statistics)
    median_grad = np.max(grad_max_statistics)

    summary_gwratio = tf.Summary()
    summary_gwratio.value.add(tag="gradient_ratio_mean_wg", simple_value=mean_wg_ratio)
    summary_gwratio.value.add(tag="gradient_ratio_median_wg", simple_value=median_wg_ratio)
    summary_gwratio.value.add(tag="gradient_ratio_max_wg", simple_value=max_wg_ratio)
    summary_gwratio.value.add(tag="gradient_ratio_min_wg", simple_value=min_wg_ratio)

    summary_gwratio.value.add(tag="gradient_norm_max", simple_value=max_grad_norm)
    summary_gwratio.value.add(tag="gradient_norm_mean", simple_value=mean_grad_norm)
    summary_gwratio.value.add(tag="gradient_norm_median", simple_value=median_grad_norm)
    summary_gwratio.value.add(tag="gradient_norm_min", simple_value=min_grad_norm)

    summary_gwratio.value.add(tag="gradient_mean", simple_value=mean_grad)
    summary_gwratio.value.add(tag="gradient_max", simple_value=max_grad)
    summary_gwratio.value.add(tag="gradient_min", simple_value=min_grad)
    summary_gwratio.value.add(tag="gradient_median", simple_value=median_grad)

    train_writer.add_summary(summary_gwratio, step)



def create_summaries(network,img_sequence_length):
    summaries = set()
    for end_point in network.end_points:
        x = network.end_points[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))

    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))


    for grad, var in network.gradients:
        summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))

    _weight_image_summary(summaries, network.first_layer_weights, img_sequence_length, scope="")
    merged_train = tf.summary.merge(list(summaries), name='train_summary_op')

    return summaries,merged_train



def _weight_image_summary(summaries, weights,img_sequence_length, scope=""):
    # visualization of first convolutional layer
    # weights is of shape (length,width,depth,filters), z.b. (8,8,6,64) for two images with 3 channels each


    if weights is not None:
        split_nr = img_sequence_length

        split_tensors = tf.split(weights, split_nr, axis=2, name="split")  # list of [(8,8,3,64),(8,8,3,64),...]
        filter_cols = list()
        for split in split_tensors:
            padded_filters = tf.pad(split, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]),
                                    mode='CONSTANT')  # filter to 10x10x3x64

            padded_filters_shape = padded_filters.get_shape().as_list()  # 10x10x3x64
            trsp_pf = tf.transpose(padded_filters, perm=[3, 0, 1, 2])  # 64x10x10x3
            filter_col = tf.reshape(trsp_pf, shape=[1, -1, padded_filters_shape[1],
                                                    padded_filters_shape[2]])  # 1x64x10x10x3 => 1x640x10x3

            filter_cols.append(filter_col)

        stacked_slices = tf.stack(filter_cols)  # 3x1x640x10x3

        trsp_ss = tf.transpose(stacked_slices, perm=[1, 2, 0, 3, 4])

        trsp_ss_shape = trsp_ss.get_shape().as_list()  # 1x640x3x10x3

        weight_image = tf.reshape(trsp_ss, shape=[1, trsp_ss_shape[1], -1, trsp_ss_shape[4]])  # 1x640x30x3
        summaries.add(tf.summary.image(tensor=weight_image, name="weights"))




def save_summaries(sess,merge_op,feed_dict,train_writer,gradients,learning_rate,step):
    if gradients is not None:
        _save_gradient_stats( train_writer, gradients, learning_rate, step)
    summary = sess.run(merge_op,feed_dict=feed_dict)
    train_writer.add_summary(summary, step)



