from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import datetime
import logging
from matplotlib import pyplot as plt
from IPython import display

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Response

import keras
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i=0
        self.x=[]
        self.losses=[]
        self.val_losses=[]
        self.acc=[]
        self.val_acc=[]
        self.logs=[]
        print("on_train_being")
    
    def on_epoch_end(self, epoch, logs={}):
        print("on epoch end", epoch)
        print(logs)

    def on_batch_end(self, batch, logs={}):
        print("on batch end", batch)
        print(logs)

def runBasic():
    plot_losses = PlotLearning()
    # Provide the output path and name for the plot
    filename = 'output/training_plot.jpg'
    # Find the number of classes
    num_classes = 10
    # Split the data into train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Preprocess data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train_cat = np.squeeze(keras.utils.to_categorical(y_train, num_classes))
    y_test_cat = np.squeeze(keras.utils.to_categorical(y_test, num_classes))

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    
    model.fit(x_train, y_train_cat,
         epochs=25,
         validation_data=(x_test, y_test_cat),
         callbacks=[plot_losses])

# instantiate the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

CORS(app, resources={r'/*': {'origins': '*'}}) 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

runBasic()

if __name__ == "__main__":
    print('Started')
    app.run(debug=True, use_reloader=False)
