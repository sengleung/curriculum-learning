import keras
import json
import emnist_file_loader as efl

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import random
import numpy as np

#Consts
digits_classes = 10
balanced_classes = 47
by_class_classes = 62
classes = balanced_classes

files = {
    'digits_train_data' : './datasets/emnist-mnist-train-images-idx3-ubyte',
    'digits_train_labels' : './datasets/emnist-mnist-train-labels-idx1-ubyte',
    'digits_test_data' : './datasets/emnist-mnist-test-images-idx3-ubyte',
    'digits_test_labels' : './datasets/emnist-mnist-test-labels-idx1-ubyte',
    'letters_train_data' : './datasets/emnist-letters-train-images-idx3-ubyte',
    'letters_train_labels' : './datasets/emnist-letters-train-labels-idx1-ubyte',
    'letters_test_data' : './datasets/emnist-letters-test-images-idx3-ubyte',
    'letters_test_labels' : './datasets/emnist-letters-test-label-idx1-ubyte',
    'balanced_train_data' : './datasets/emnist-balanced-train-images-idx3-ubyte',
    'balanced_train_labels' : './datasets/emnist-balanced-train-labels-idx1-ubyte',
    'balanced_test_data' : './datasets/emnist-balanced-test-images-idx3-ubyte',
    'balanced_test_labels' : './datasets/emnist-balanced-test-labels-idx1-ubyte',
    'byclass_train_data' : './datasets/emnist-byclass-train-images-idx3-ubyte',
    'byclass_train_labels' : './datasets/emnist-byclass-train-labels-idx1-ubyte',
    'byclass_test_data' : './datasets/emnist-byclass-test-images-idx3-ubyte',
    'byclass_test_labels' : './datasets/emnist-byclass-test-labels-idx1-ubyte'
}

def test_image(image, label):
    print("Label : " + str(label))
    rescaled = (image * 255).astype(np.uint8)
    plt.imshow(rescaled.reshape(28,28))
    plt.show()


def load_data(name, amount=-1):
    data_collection = dict()
    data_collection['x'] = efl.load_idx_images(files[name + '_train_data'], amount)
    data_collection['y'] = efl.load_idx_labels(files[name + '_train_labels'], amount)
    data_collection['test_x'] = efl.load_idx_images(files[name + '_test_data'], amount)
    data_collection['test_y'] = efl.load_idx_labels(files[name + '_test_labels'], amount)
    return data_collection

def get_model(id):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.add(Dropout(0.5))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.load_weights('./models/' + id)
    return model

def writeToJson(filename, results):
    with open(filename + '.json', 'w') as fp:
        json.dump(results, fp, indent=4)
