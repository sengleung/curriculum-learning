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
N_TRAINING = 10000
N_TEST = 1000
EPOCHS = 1
TRAINING_DATA = 'fashion-mnist_train.csv'
TEST_DATA = 'fashion-mnist_test.csv'
IMG_ROWS = 28
IMG_COLS = 28
SHAPE = (IMG_ROWS, IMG_COLS, 1)


def create_cnn(x, y, epochs=EPOCHS):
    # Create model of Convolutional neural network
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape=SHAPE, activation='relu'))
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

    classifier.fit(x, y, epochs=epochs)

    return classifier


def format_data(data):
    x = np.array(data.iloc[:, 1:])
    x = x.reshape(x.shape[0], IMG_ROWS, IMG_COLS, 1)
    x = x.astype('float32')
    x /= 255

    y = to_categorical(np.array(data.iloc[:, 0]))  # training data

    return x, y


def real_classification(X_test, y_test):
    real_classes = np.zeros(len(X_test))
    y_test = y_test.astype('int')
    for i in range(len(X_test)):
        for j in range(0, 10):
            if(y_test[i][j] == 1):
                real_classes[i] = (j)
    return real_classes


def calculate_accuracy(predicted, real):
    correct = 0
    for i in range(len(real)):
        if predicted[i] == real[i]:
            correct += 1
    return correct / len(real)


def test_accuracy(name,  data_test, classifier):
    # Predict training data using classifier
    X_test, y_test = format_data(data_test)
    predicted_classes = classifier.predict_classes(X_test)

    # Real classification
    real_classes = real_classification(X_test, y_test)

    print(name + ':',
          'train:', N_TRAINING,
          'test:', N_TEST,
          'accuracy:', calculate_accuracy(predicted_classes, real_classes))


# Load data
data_train = pd.read_csv(TRAINING_DATA, nrows=N_TRAINING)
data_test = pd.read_csv(TEST_DATA, nrows=N_TEST)
curriculum_data = pd.read_csv(TRAINING_DATA).tail(N_TRAINING)

# Create training model without curriculum and calculate accuracy
X_train, y_train = format_data(data_train)
classifier = create_cnn(X_train, y_train)
test_accuracy('No curriculum', data_test, classifier)


# Classify latter half of data to group data into easy and hard
X_test, y_test = format_data(curriculum_data)
predicted_classes = classifier.predict_classes(X_test)
real_classes = real_classification(X_test, y_test)

correct = 0
incorrects = set()
for x in range(len(X_train)):
    if predicted_classes[x] == real_classes[x]:
        correct += 1
    else:
        incorrects.add(x)

print('Classifed curriculum', correct / len(X_test))

# Sort data with easy first and hard next
curriculum_data_list = curriculum_data.values

easy = []
hard = []
for i in range(len(X_train)):
    if i not in incorrects:
        easy.append(curriculum_data_list[i])
    else:
        hard.append(curriculum_data_list[i])

combined = easy + hard
curriculum = pd.DataFrame(combined)

# Create training model with curriculum and calculate accuracy
X_train, y_train = format_data(curriculum)
classifier2 = create_cnn(X_train, y_train)
test_accuracy('With curriculum', data_test, classifier2)
