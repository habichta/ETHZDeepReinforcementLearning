'''
Created on Mon Jan 18 11:32:24 2017
Module contains several network models
- Loads model input parameters from params.json
- Uses input generator to feed the network

@author: maverick
'''

from __future__ import print_function
import numpy as np

import itertools as it

np.random.seed(1337)  # for reproducibility
nrs = np.random.RandomState(1337)

from PIL import Image
import inputgenerator as ig
import json, sys
from keras.utils.vis_utils import plot_model
import resnet
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K
from shutil import copyfile
import re
import unicodedata

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
config.gpu_options.per_process_gpu_memory_fraction = 0.2

"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""


set_session(tf.Session(config=config))


def folder_name(value):
    return "".join([c for c in value if c.isalpha() or c.isdigit() or c == ' ']).rstrip()

def pearson_correlation(y_true, y_pred):




    fsp = y_pred - K.mean(y_pred,axis=-1,keepdims=True)
    fst = y_true - K.mean(y_true,axis=-1, keepdims=True)

    corr = K.mean((K.sum((fsp)*(fst),axis=-1))) / K.mean((
    K.sqrt(K.sum(K.square(y_pred - K.mean(y_pred,axis=-1,keepdims=True)),axis=-1) * K.sum(K.square(y_true - K.mean(y_true,axis=-1,keepdims=True)),axis=-1))))

    return corr


def pearson_correlation_f(y_true, y_pred):

    y_true,_ = tf.split(y_true[:,1:],2,axis=1)
    y_pred, _ = tf.split(y_pred[:,1:], 2, axis=1)


    fsp = y_pred - K.mean(y_pred,axis=-1,keepdims=True)
    fst = y_true - K.mean(y_true,axis=-1, keepdims=True)

    corr = K.mean((K.sum((fsp)*(fst),axis=-1))) / K.mean((
    K.sqrt(K.sum(K.square(y_pred - K.mean(y_pred,axis=-1,keepdims=True)),axis=-1) * K.sum(K.square(y_true - K.mean(y_true,axis=-1,keepdims=True)),axis=-1))))

    return corr



def pearson_correlation_l(y_true, y_pred):

    _,y_true = tf.split(y_true[:,1:],2,axis=1)
    _,y_pred = tf.split(y_pred[:,1:], 2, axis=1)


    fsp = y_pred - K.mean(y_pred,axis=-1,keepdims=True)
    fst = y_true - K.mean(y_true,axis=-1, keepdims=True)

    corr = K.mean((K.sum((fsp)*(fst),axis=-1))) / K.mean((
    K.sqrt(K.sum(K.square(y_pred - K.mean(y_pred,axis=-1,keepdims=True)),axis=-1) * K.sum(K.square(y_true - K.mean(y_true,axis=-1,keepdims=True)),axis=-1))))

    return corr

def r_squared(y_true,y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))



def kl_divergence(y_true, y_pred):

    y_pred = K.clip(y_pred,0.0000001,10000)

    true_s = K.sum(y_true,keepdims=True,axis=-1)

    p_s = K.sum(y_pred, keepdims=True, axis=-1)

    dist_t = y_true/true_s
    dist_p = y_pred/p_s

    div = dist_t/dist_p
    KL_div = K.mean(K.sum(dist_t * K.log(div),axis=-1))

    return KL_div



def rmse(y_true, y_pred):

    rmse = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    return rmse

def max_error(y_true, y_pred):

    me = K.mean(K.max(y_pred - y_true,axis=-1,keepdims=True))

    return me


def max_batch_error(y_true, y_pred):

    mbe = K.max(y_pred - y_true)

    return mbe



def mean_true(y_true, y_pred):

    return K.mean(y_true)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def normalized_mae_loss(y_true, y_pred):

    n = K.mean(K.abs(y_pred-y_true),axis=-1)
    d = K.mean(y_true,axis=-1)
    n_loss = K.mean((n/(d+K.epsilon()))) # mean over batches

    return n_loss


if __name__ == "__main__":
    json_file_p = sys.argv[1]
    #json_file_p = "/media/nox/OS/Linux/Documents/Masterarbeit/shared/dlabb/abb_deeplearning_keras/Source_code/models/experiment_settings.json"
    json_file = open(json_file_p, "r")
    #    json_file = open('/home/maverick/knet/models/params.json',"r")
    params = json.load(json_file)

    experiment_name = params['experiment_name']
    experiment_folder = params['experiment_folder']
    batch_size = 1
    nb_classes = params['nb_classes']
    nb_epoch = params['nb_epoch']
    img_rows = params['img_rows']
    img_cols = params['img_cols']
    start_date = params['start_date']
    end_date = params['end_date']
    over_sample = params['over_sample']
    label_name = params['label_name']
    sequence_length = params['sequence_length']
    sequence_stride = params['sequence_stride']
    balanced = params['balanced']
    order = params['order']
    sky_mask_file = params['sky_mask']
    data_file = params['data_file']
    label_file = params['label_file']
    root_dir = params['root_dir']
    input_irradiance = params['input_irradiance']
    prediction_resolution = params['prediction_resolution']
    cpk_path = params['cpk_path']
    test_file = params['test_file']

    seq_channels = sequence_length*3
    labels30 = ["IRR" + str(i) for i in range(31)] #20sec forecast frequency
    labels10 = ["IRR" + str(i) for i in range(0, 31, 3)]#60sec forecast frequency

    pred30 =  ["P" + str(i) for i in range(31)]
    pred10 =  ["P" + str(i) for i in range(0, 11)]

    l30 = ["L" + str(i) for i in range(31)]
    l10 = ["L" + str(i) for i in range(0, 11)]

    if prediction_resolution == 30:
        labels = labels30
        pred_col=pred30
        label_col = l30
    else:
        labels = labels10
        pred_col=pred10
        label_col = l10

    #CHANGE THIS

    nr_labels = len(labels)


    master_df, train_df, validation_df, test_df, sky_mask = ig.download_metadata(
        start_date=start_date,
        end_date=end_date,
        data_file=data_file,
        label_file=label_file,
        sky_mask_file=sky_mask_file,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        nb_classes=nb_classes,
        over_sample=over_sample,
        label_name=label_name,
        balanced=balanced)

    train_batch_nr_per_epoch = (len(train_df.index) // batch_size)
    validation_batch_nr_per_epoch = (len(validation_df.index) // batch_size)
    test_batch_nr_per_epoch = (len(test_df.index) // batch_size)

    print(validation_batch_nr_per_epoch)






    training_data_generator = ig.generate_data(
        set_df=train_df,
        master_df=master_df,
        root_dir=root_dir,
        img_rows=img_rows,
        img_cols=img_cols,
        batch_size=batch_size,
        order=order,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        sky_mask=sky_mask,
        labels=labels
    )

    validation_data_generator = ig.generate_data(
        set_df=validation_df,
        master_df=master_df,
        root_dir=root_dir,
        img_rows=img_rows,
        img_cols=img_cols,
        batch_size=batch_size,
        order=order,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        sky_mask=sky_mask,
        labels=labels
    )

    test_data_generator = ig.generate_data(
        set_df=test_df,
        master_df=master_df,
        root_dir=root_dir,
        img_rows=img_rows,
        img_cols=img_cols,
        batch_size=batch_size,
        order=order,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        sky_mask=sky_mask,
        labels=labels
    )


    if test_file == "test":
        data_generator = test_data_generator
        data_file = test_df
        steps = test_batch_nr_per_epoch


    elif test_file == "validation":
        data_generator = validation_data_generator
        data_file = validation_df
        steps = validation_batch_nr_per_epoch

    elif test_file == "train":
        data_generator = training_data_generator
        data_file = train_df
        steps = train_batch_nr_per_epoch

    else:
        raise ValueError("Illegal test_file setting")



    experiment_root = os.path.join(experiment_folder, experiment_name,"eval")
    os.mkdir(experiment_root)

    experiment_path = os.path.join(experiment_root,folder_name(str(os.path.basename(cpk_path))))

    os.mkdir(experiment_path)





    ################RES NET######################


    resnet_model = resnet.ResnetBuilder.build_resnet_18((seq_channels, img_rows, img_cols),(sequence_length,),
                                                        nr_labels,input_irradiance)  # (img_rows*sequence_length) should be used, for normal sequence (3 color channels)
    resnet_model.compile(loss='mean_absolute_error',
                         optimizer='adam',
                         metrics=[pearson_correlation,pearson_correlation_f,pearson_correlation_l,rmse,mean_true,mean_pred,kl_divergence,max_batch_error,max_error,r_squared,normalized_mae_loss])

    #plot_model(resnet_model, show_shapes=True,to_file=os.path.join(experiment_path,'logs',"model.png"))
    print(resnet_model.summary())
    # pre-initialized weights
    resnet_model.load_weights(cpk_path)
    predict = resnet_model.predict_generator(generator=data_generator, steps=steps,verbose=1,max_queue_size=1,workers=1,use_multiprocessing=False)



    label_df = data_file[labels]
    label_df.columns = label_col



    predict_df = pd.DataFrame(data=predict,index=label_df.index,columns=pred_col)


    result_df = pd.concat([predict_df,label_df],axis=1).sort_index()




    result_df.to_csv(os.path.join(experiment_path,'eval_predictions.csv'))
    print("Predict shape:", predict.shape)



    np.save(os.path.join(experiment_path,'predict_model.npy'), predict)
    print(predict)



