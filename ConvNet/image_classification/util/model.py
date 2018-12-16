import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def load(name, filepath):
    return load_model(filepath + name + '.h5')

def save(model, filepath,  name):
    model.save(filepath + '/' + name + '.h5')
