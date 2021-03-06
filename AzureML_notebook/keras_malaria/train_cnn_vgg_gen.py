# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:57:15 2019

@author: edumu
"""
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import argparse
import os

import matplotlib.pyplot as plt

import keras
#from keras.models import Sequential, model_from_json
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D

from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD

#from keras.optimizers import RMSprop
from keras.callbacks import Callback

import tensorflow as tf

from azureml.core import Run

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--x_filename', type=str, dest='x_filename', help='Filename with training data')
parser.add_argument('--y_filename', type=str, dest='y_filename', help='Filename with label data')
parser.add_argument('--training_size', type=str, dest='training_size', help='Size of training dataset')
parser.add_argument('--n_epochs', type=int, dest='n_epochs', help='Number of epochs')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', help='Learning rate optimizer')

args = parser.parse_args()

data_folder = args.data_folder

print('training dataset is stored here:', data_folder)
#Load Data
# Load npz file containing image arrays
x_npz = np.load(data_folder+'/'+args.x_filename)
x = x_npz['arr_0']
# Load binary encoded labels for Lung Infiltrations: 0=Not_infiltration 1=Infiltration
y_npz = np.load(data_folder+'/'+args.y_filename)
y = y_npz['arr_0']

# Create traiinf and validation datasets
from sklearn.model_selection import train_test_split

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x,y, test_size=0.2, random_state=1, stratify=y)
# Second split the 20% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)

training_set_size = X_train.shape[0]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')

# Global variables and parameters
img_width, img_height = 128, 128
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
n_epochs = args.n_epochs
batch_size = args.batch_size
learning_rate=args.learning_rate

print('nb_train_samples = ',nb_train_samples)
print('nb_validation_samples = ',nb_validation_samples)

# load model
model = VGG16(include_top=False, input_shape=(img_width, img_height, 3))
# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# compile model
opt = SGD(lr=learning_rate, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Creating the image generator for data augmentation
from keras.preprocessing.image import ImageDataGenerator

# define data preparation
shift = 0.2

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True, rotation_range=90,
                                  width_shift_range=shift, height_shift_range=shift)
valtest_datagen = ImageDataGenerator(rescale=1. / 255)

#Apply the transformation or data augmentation previously defined
train_generator = train_datagen.flow(np.array(X_train), y_train, batch_size=batch_size)
validation_generator = valtest_datagen.flow(np.array(X_val), y_val, batch_size=batch_size)
test_generator = valtest_datagen.flow(np.array(X_test), y_test, batch_size=batch_size)

# start an Azure ML run
run = Run.get_context()


class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])

#Train the model and save the parameters
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=n_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[LogRunMetrics()]
)

#score = model.evaluate(X_test, y_test, verbose=0)
score=model.evaluate_generator(test_generator, steps=len(X_test)//batch_size, verbose=0)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

# log a single value
run.log("Training size", args.training_size)
print('Training size:', args.training_size)

fig = plt.figure(figsize=(25, 25))
ax = fig.add_subplot(131)
ax.set_title('Acc vs Loss ({} epochs)'.format(n_epochs), fontsize=14)
ax.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
ax.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
ax.legend(fontsize=8)
ax.grid(True)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

ax=fig.add_subplot(132)
ax.plot(epochs, acc, 'blue', label='Training acc')
ax.plot(epochs, val_acc, 'red', label='Validation acc')
ax.set_title('Training and validation accuracy')
ax.legend()

ax=fig.add_subplot(133)
ax.plot(epochs, loss, 'blue', label='Training loss')
ax.plot(epochs, val_loss, 'red', label='Validation loss')
ax.set_title('Training and validation loss')
ax.legend()
                 
# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model_generator.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model_generator.h5')
print("model saved in ./outputs/model folder")
