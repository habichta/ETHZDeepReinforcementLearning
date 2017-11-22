from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.optimizers import SGD


def get_vgg(input_shape, nb_classes):
    vgg = Sequential()
    vgg.add(ZeroPadding2D((1,1),input_shape=input_shape))
    vgg.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(64, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))
    
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(128, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))
    
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(256, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))
    
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))
    
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(ZeroPadding2D((1,1)))
    vgg.add(Convolution2D(512, 3, 3, activation='relu'))
    vgg.add(MaxPooling2D((2,2), strides=(2,2)))
    
    vgg.add(Flatten())
    vgg.add(Dense(4096, activation='relu'))
    vgg.add(Dropout(0.5))
    vgg.add(Dense(4096, activation='relu'))
    vgg.add(Dropout(0.5))
    vgg.add(Dense(nb_classes, activation='softmax'))
    return vgg



batch_size = 32
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
input_shape = X_train.shape[1:]



vgg = get_vgg(input_shape,nb_classes)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
vgg.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

vgg.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)