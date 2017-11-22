'''
Created on Mon Jan 18 11:32:24 2017
Module contains several network models
- Loads model input parameters from params.json
- Uses input generator to feed the network

@author: maverick
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD
from PIL import Image
import inputgenerator as ig
import json, sys
from keras.utils.visualize_util import plot
import resnet
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score
import os


    

if __name__== "__main__":
    
    json_file = sys.argv[1]
    json_file = open(json_file, "r")
#    json_file = open('/home/maverick/knet/models/params.json',"r")
    params = json.load(json_file)

    batch_size = params['batch_size']
    nb_classes = params['nb_classes']
    nb_epoch = params['nb_epoch']
    data_augmentation = True
    img_channels = params['img_channels'] # Images are RGB.
    # input image dimensions
    img_rows = params['img_rows']
    img_cols = params['img_cols']
    
    # start_date = params['start_date']
    # end_date = params['end_date']
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    slice_by_hour = params['slice_by_hour']
    start_hour = params['start_hour']
    end_hour = params['end_hour']
    test_size = params['test_size']
    over_sample = params['over_sample']
    label_name = params['label_name']
    sequence_mode = params['sequence_mode']
    sequence_length = params['sequence_length']
    overlap = params['overlap']
    balanced = params['balanced']
    order = params['order']

    sky_mask = params['sky_mask']
    label_mask = params['label_mask']
    meta_file = params['meta_file']
    root_dir = params['root_dir']



    #initializes sky mask and class label mask to be added to the input images
    sky_mask = Image.open(sky_mask)
    label_mask = {0 : label_mask+"0.png",
           1 : label_mask+"1.png",
           2 : label_mask+"2.png",
           3 : label_mask+"3.png",
           }
    for i in range(4):
        label_mask[i] = Image.open(label_mask[i])

        
    
    #initializes sequence_length to be 1 in single image generation cases
    sequence_length = sequence_length if sequence_mode else 1
    
        
      
    
    files_test, _, labels_test,_ = ig.download_metadata(
                                    start_date = start_date,
                                    end_date = end_date, 
                                    meta_file = meta_file, 
                                    root_dir = root_dir,
                                    slice_by_hour = slice_by_hour,
                                    start_hour = start_hour,
                                    end_hour = end_hour,
                                    test_size = 1,
                                    nb_classes = nb_classes,
                                    over_sample = over_sample,
                                    label_name = label_name,
                                    sequence_mode = sequence_mode,
                                    sequence_length = sequence_length,
                                    overlap = overlap,
                                    balanced = False)    
    
    
    testing_data_generator = ig.generate_testing_data(
                                    file_list = files_test,
                                    labels = labels_test,     
                                    img_rows = img_rows,
                                    img_cols = img_cols,
                                    batch_size = batch_size,
                                    order = order,
                                    nb_classes = nb_classes,
                                    sequence_mode = sequence_mode,
                                    sequence_length = sequence_length,
                                    overlap = overlap,
                                    sky_mask=sky_mask,
                                    label_mask=label_mask)
    
    test_label_generator = ig.generate_testing_data(
                                    file_list = files_test,
                                    labels = labels_test,     
                                    img_rows = img_rows,
                                    img_cols = img_cols,
                                    batch_size = batch_size,
                                    order = order,
                                    nb_classes = nb_classes,
                                    sequence_mode = sequence_mode,
                                    sequence_length = sequence_length,
                                    overlap = overlap,
                                    sky_mask=sky_mask,
                                    label_mask=label_mask)


    # samples_per_epoch = len(files_train)
    # validation_samples = len(files_validate)
    test_samples = len(files_test)

    print ("test samples - " + str(test_samples))
    
  
    ################RES NET######################

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=30)
    csv_logger = CSVLogger('checkpoint/CAV_resnet18_201516_nv_seq2.csv')
     # checkpoint for every improved epoch
#    filepath="checkpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint only the best epoch
    filepath="checkpoint/CAV_RS18_201516_nv_seq2_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    # tensorboard visualization
    visualizer = TensorBoard(log_dir='checkpoint/logs', histogram_freq=1, write_graph=True)
    
    resnet_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes) # (img_rows*sequence_length) should be used, for normal sequence (3 color channels)
    resnet_model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
    
    
    plot(resnet_model, show_shapes=True,to_file="checkpoint/resnet_model")   
    # print (resnet_model.summary())    


    resnet_model.load_weights("/home/pdinesh/knet-euryale/euryale/checkpoint/exp_45_JT18_c4_retrain/JT_RS18_6chan_4CLASS_retrain_day-day_E68_0.8621_weights.best.hdf5")
    predict = resnet_model.predict_generator(testing_data_generator,test_samples)

    test_generator_volume = test_samples/batch_size+2
    predict_labels = []
    for x,y in test_label_generator:
        test_generator_volume -=1
        if(test_generator_volume > 0):
#            print (y.shape)
            predict_labels.append(y)
        else:
            break
    predict_labels = np.vstack(predict_labels)
    
    def compute_pred_score(predict, predict_labels):
        right = 0
        wrong = 0
        for i in range(-1,len(predict)):
            if((predict[i].argmax(axis=0))== (predict_labels[i].argmax(axis=0))):
                right +=1
            else:
                wrong +=1
        return right,wrong

        
    right,wrong = compute_pred_score(predict,predict_labels)
#    accuracy_score(predict_labels, predict, normalize=False)
    
    test_pred = np.argmax(predict, axis=1)
    test_true = np.argmax(predict_labels, axis=1)
    print (accuracy_score(test_true, test_pred))
    print ("right - "+str(right)+ ", wrong - "+ str(wrong))
    print (start_date)

    
    
    np.save('/home/pdinesh/knet-euryale/models/checkpoint/daily_MS_4/'+start_date+'_predict-model.npy', predict)
    np.save('/home/pdinesh/knet-euryale/models/checkpoint/daily_MS_4/'+start_date+'_predict-true.npy', predict_labels)
    
    
    
    
        
#    test_labels = 
    
#    evaluate = resnet_model.evaluate_generator(testing_data_generator,validation_samples)
    
#    print("%s: %.2f%%" % (resnet_model.metrics_names[1], evaluate[1]*100))



    
##############################################################
