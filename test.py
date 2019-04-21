from __future__ import print_function

import keras

import input
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling1D,MaxPooling2D
from keras.optimizers import SGD, Adam
import os
from keras.utils import to_categorical

from keras import backend as K


X_train = input.traindata
Y_train = to_categorical(input.trainlabel)
X_test = input.testdata
Y_test = to_categorical(input.testlabel)
X = input.train_input

batch_size = 128
num_classes = 10
epochs = 20
learning_rate = 0.01


print('x_train shape:', X_train.shape)
print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')
print(X.shape, 'whole samples')

model = Sequential()
model.add(Conv1D (kernel_size = (3), filters = 2, input_shape=X.shape[1:], activation='relu'))
print(model.input_shape)
print(model.output_shape)
model.add(MaxPooling1D(pool_size = (2), strides=(1)))
print(model.output_shape)
#model.add(keras.layers.core.Reshape([20,-1,1]))
#print(model.output_shape)
#model.add(Conv2D (kernel_size = (2,3), filters = 4, activation='relu'))
#print(model.output_shape)
#model.add(MaxPooling2D(pool_size = (1,2), strides=(1,2)))
print(model.output_shape)
model.add(Flatten())

model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd=SGD(lr=learning_rate)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer= Adam(), metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128,
                        epochs=20, verbose=1, shuffle=True)
model.evaluate(X_test, Y_test, verbose=0)
model_json = model.to_json()

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)
