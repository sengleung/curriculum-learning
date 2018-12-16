import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model_amounts = 5
categorical_classes = 47

filepath = './untrained/'
filebase = 'model_'

#Define a metric to see if one of top 2 predictions are correct (5 vs s, l vs i)
#//https://github.com/keras-team/keras/issues/8102
def top_2_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

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
    model.add(Dense(categorical_classes, activation='softmax'))

    #Compile our model
    model.compile(
        loss=keras.losses.categorical_crossentropy,
         optimizer=keras.optimizers.Adadelta(),
         metrics=[keras.metrics.categorical_accuracy, top_2_accuracy]
    )

    #Save our i'th model
    uri = filepath + filebase
    model.save(filepath + filebase + str(i) + '.h5')

"""
Notes:
    Small model configuration, we just create and compile each time for new
    inital weights. Probably a better way
"""
