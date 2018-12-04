import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

filepath = './models/'

def load(name, id=0):
    return load_model(filepath + name + '_' + str(id) + '.h5')

def save(model, name, id=0):
    model.save(filepath + name + '_' + str(id) + '.h5')
