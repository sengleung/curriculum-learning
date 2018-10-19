import keras
import emnist_file_loader as efl
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
from datalist import Datalist

#Consts
files = {
    'digits_train_data' : './datasets/emnist-mnist-train-images-idx3-ubyte',
    'digits_train_labels' : './datasets/emnist-mnist-train-labels-idx1-ubyte',
    'digits_test_data' : './datasets/emnist-mnist-test-images-idx3-ubyte',
    'digits_test_labels' : './datasets/emnist-mnist-test-labels-idx1-ubyte',
    'letters_train_data' : './datasets/emnist-letters-train-images-idx3-ubyte',
    'letters_train_labels' : './datasets/emnist-letters-train-labels-idx1-ubyte',
    'letters_test_data' : './datasets/emnist-letters-test-images-idx3-ubyte',
    'letters_test_labels' : './datasets/emnist-letters-test-label-idx1-ubyte'
}
number_of_digits_classes = 10
batch_size = 0
epochs = 0

#Loading in data
digits_training_examples = efl.load_idx_images(files['digits_train_data'])
digits_training_labels = efl.load_idx_labels(files['digits_train_labels'])
digits_training_data = Datalist(digits_training_examples, digits_training_labels)

digits_testing_examples = efl.load_idx_images(files['digits_test_data'])
digits_testing_labels = efl.load_idx_labels(files['digits_test_labels'])
digits_testing_data = Datalist(digits_testing_examples, digits_testing_labels)

batch_size = 128
epochs = 1

#Convert labels to target vectors
#Converts 6 -> [0,0,0,0,0,0,1,0,0,0], 2 -> [0,0,1,0,0,0,0,0,0,0]
digits_training_target_vectors = keras.utils.to_categorical(digits_training_labels, number_of_digits_classes)
digits_testing_target_vectors = keras.utils.to_categorical(digits_testing_labels, number_of_digits_classes)

#Setup model
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
model.add(Dense(number_of_digits_classes, activation='softmax'))

#Compile and test model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(digits_training_data.get_examples(), digits_training_target_vectors,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(digits_testing_data.get_examples(), digits_testing_target_vectors))

score = model.evaluate(digits_testing_data.get_examples(), digits_testing_target_vectors, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# #Display data
# print("Img 1: " + str(digits_training_data.get_label(2)) )
# print("Img 2: " + str(digits_testing_data.get_label(2)) )
#
# fig, axs = plt.subplots(1,2)
# axs[0].imshow(digits_training_data.get_example(2).reshape((28,28)) )
# axs[1].imshow(digits_testing_data.get_example(2).reshape((28,28)) )
# plt.show()
