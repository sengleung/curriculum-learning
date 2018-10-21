import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

LABEL = 0
N_TRAINING = 200
N_TEST = 200
# data_train = pd.read_csv('fashion-mnist_train.csv')
TRAINING_DATA = 'fashion-mnist_test.csv'
TEST_DATA = 'fashion-mnist_test.csv'
IMG_ROWS = 28
IMG_COLS = 28
shape = (IMG_ROWS, IMG_COLS, 1)


def create_cnn(epochs=1):
    # Create model of Convolutional neural network
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=shape, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.4))

    classifier.add(Flatten())

    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=10, activation='sigmoid'))

    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit(X_train, y_train, epochs=epochs)

    return classifier


def real_classification(X_test, y_test):
    real_classes = np.zeros(len(X_test))
    y_test = y_test.astype('int')
    for i in range(len(X_test)):
        for j in range(0, 10):
            if(y_test[i][j] == 1):
                real_classes[i] = (j)
    return real_classes


def test_data(data_test):
    X_test = np.array(data_test.iloc[:, 1:])
    X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    X_test = X_test.astype('float32')
    X_test /= 255

    # test data     x -images y - results
    y_test = to_categorical(np.array(data_test.iloc[:, 0]))

    return X_test, y_test


def test_data(data_train):
    X_train = np.array(data_train.iloc[:, 1:])
    X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    y_train = to_categorical(np.array(data_train.iloc[:, 0]))  # training data

    return X_train, y_train


data_train = pd.read_csv(TRAINING_DATA, nrows=N_TRAINING)
data_test = pd.read_csv(TEST_DATA,  nrows=N_TEST)


# Test data

X_train, y_train = test_data(data_train)

X_test, y_test = test_data(data_test)

classifier = create_cnn(1)

# Predict training data

predicted_classes = classifier.predict_classes(X_test)

# Gather testing data real classification

real_classes = real_classification(X_test, y_test)

# Compare predicted classification with real classification

correct = 0
incorrects = set()
for x in range(len(X_train)):
    if(predicted_classes[x] == real_classes[x]):
        correct += 1
    else:
        incorrects.add(x)

print('The accuracy is', correct / len(X_test))

data_test_list = data_train.values


# easy = []
# hard = []
# for i in range(len(X_train)):
#     if i not in incorrects:
#         easy.append(data_test_list[i])
#     else:
#         hard.append(data_test_list[i])
#
# easy_data = pd.DataFrame(easy)
# hard_data = pd.DataFrame(hard)

# ------------------------------------------------------

# data_test = pd.read_csv('fashion-mnist_test.csv',  nrows=N_TEST)
# X_train = np.array(easy_data.iloc[:, 1:])
# y_train = to_categorical(np.array(easy_data.iloc[:, 0]))  # training data
# # X_test = np.array(data_test.iloc[:, 1:])
# # y_test = to_categorical(np.array(data_test.iloc[:, 0]))    # test data
#
# X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
# X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
#
# classifier = Sequential()
#
# classifier.add(Conv2D(32, (3, 3), input_shape=shape, activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.2))
#
# classifier.add(Conv2D(32, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# classifier.add(Dropout(0.4))
#
# classifier.add(Flatten())
#
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dense(units=10, activation='sigmoid'))
#
# classifier.compile(
#     optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# classifier.fit(X_train, y_train, epochs=5)
#
# correct = 0
# incorrects = set()
# for x in range(len(X_test)):
#     if(predicted_classes[x] == real_classes[x]):
#         correct += 1
#     else:
#         incorrects.add(x)
#
# print('The accuracy is', correct / len(X_test))
# data_test = pd.read_csv('fashion-mnist_test.csv',  nrows=N_TEST)
# X_train = np.array(hard_data.iloc[:, 1:])
# y_train = to_categorical(np.array(hard_data.iloc[:, 0]))  # training data
# X_test = np.array(data_test.iloc[:, 1:])
# y_test = to_categorical(np.array(data_test.iloc[:, 0]))    # test data
#
# X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
# X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
#
# predicted_classes = classifier.predict_classes(X_test)
#
# classifier.fit(X_train, y_train, epochs=5)
#
# correct = 0
# incorrects = set()
# for x in range(len(X_test)):
#     if(predicted_classes[x] == real_classes[x]):
#         correct += 1
#     else:
#         incorrects.add(x)
#
# print('The accuracy is', correct / len(X_test))
