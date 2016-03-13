'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''
from __future__ import print_function

import numpy as np
np.random.seed(314)

import json

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json

batch_size = 32
nb_classes = 10
nb_epoch = 80
data_augmentation = False
experiment_name = "question3"
load_experiment_name = "question2"

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

with open('{0}_net.json'.format(load_experiment_name), 'r') as f:
    old_model = model_from_json(f.read())
old_model.load_weights('{0}_weights.h5'.format(load_experiment_name))

model = Sequential()
model.add(AveragePooling2D(pool_size=(2,2), input_shape=(img_channels, img_rows, img_cols)))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

for i in range(16):
	model.layers[i].set_weights(old_model.layers[i].get_weights())
model.layers[17].set_weights(old_model.layers[16].get_weights())
model.layers[20].set_weights(old_model.layers[18].get_weights())

# # remove softmax
# softmax_layer = model.layers.pop()
# softmax_weights = model.params.pop()
# 
# # remove last dense layer
# dense2_layer = model.layers.pop()
# dense2_weights = model.params.pop()
# 
# # remove relu
# relu_layer = model.layers.pop()
# relu_weights = model.params.pop()
# 
# # remove second to last dense layer
# dense1_layer = model.layers.pop()
# dense1_weights = model.params.pop()
# 
# # add dropout
# model.add(Dropout(0.5))
# 
# # add back removed layers
# model.add(dense1_layer)
# model.add(relu_layer)
# 
# # add dropout
# model.add(Dropout(0.5))
# 
# # add back removed layers
# model.add(dense2_layer)
# model.add(softmax_layer)

# # let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, show_accuracy=True,
              validation_data=(X_test, Y_test), shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, Y_test),
                        nb_worker=1)

# print and save history
with open('{0}_history.json'.format(experiment_name), 'w') as f:
    json.dump(hist.history, f, sort_keys=True, indent=4, separators=(',',': '))

# save model
json_string = model.to_json()
with open('{0}_net.json'.format(experiment_name), 'w') as f:
    f.write(json_string)
model.save_weights('{0}_weights.h5'.format(experiment_name))


