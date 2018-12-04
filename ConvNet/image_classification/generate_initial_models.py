import keras

import util.emnist as emnist
import util.model as model_util

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

setname = 'balanced'
dataset = emnist.sets[setname]
classes = dataset['classes']
model_amounts = 5

filepath = './models/'
filebase = 'untrained_'

for i in range(0, model_amounts):

    #Layout our model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    #Compile our model
    model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adadelta(),
          metrics=['accuracy'])

    #Save our i'th model
    model_util.save(model, 'untrained', i)

"""
Notes:
    Small model configuration, we just create and compile each time for new
    inital weights. Probably a better way
"""
