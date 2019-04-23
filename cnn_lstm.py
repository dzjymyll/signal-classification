from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
import input


# Embedding
max_features = 20000
maxlen = 24
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 10

print('Loading data...')

x_train = input.traindata
y_train = input.trainlabel
x_test = input.testdata
y_test = input.testlabel
X = input.train_input

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Build model...')

model = Sequential()

model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

ynew = model.predict_proba(input.New)

print('Class_1',ynew)

########################################class_2

x_train = input.traindata_2
y_train = input.trainlabel_2
x_test = input.testdata_2
y_test = input.testlabel_2

model_2 = Sequential()

model_2.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model_2.add(MaxPooling1D(pool_size=pool_size))
model_2.add(LSTM(lstm_output_size))
model_2.add(Dense(1))
model_2.add(Activation('sigmoid'))

model_2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model_2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))


score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

ynew = model_2.predict_proba(input.New)

print('Class_2',ynew)
