import keras
import json
import emnist_file_loader as efl
import emnist_digit_sorter as eds
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


def load_data(name):
    data_collection = dict()
    data_collection['training_data'] = efl.load_idx_images(files[name + '_train_data'])
    data_collection['training_labels'] = efl.load_idx_labels(files[name + '_train_labels'])
    data_collection['testing_data'] = efl.load_idx_images(files[name + '_test_data'])
    data_collection['testing_labels'] = efl.load_idx_labels(files[name + '_test_labels'])
    return data_collection

def create_tasks(data, labels, classes, task_count, randomize_each_task=False):
    sorted_labeled_data = eds.emnist_digit_sort_by_mean_diff(data, labels, classes)
    tasks = list()

    for i in range(0, task_count):
        task = list()

        for label, images in sorted_labeled_data.items(): #iters through each class
            partition = images[i*len(images) // task_count : (i+1)*len(images) // task_count]
            labeled_partition = list(map(lambda image: (label, image), partition))
            if randomize_each_task:
                random.shuffle(labeled_partition)
            task.extend(labeled_partition)

        tasks.append(task)
    return tasks

def create_validation_split(data, labels, validation_split):
    d = dict() #Definitely a better name than d
    copy_data = data.copy()
    copy_labels = labels.copy()
    copy_labelled_data = list(zip(copy_data, copy_labels))
    random.shuffle(copy_labelled_data)
    validation_data = copy_labelled_data[0: int(validation_split*len(data))]
    training_data = copy_labelled_data[int(validation_split*len(data)): -1]
    d['training_data'], d['training_labels'] = zip(*training_data)
    d['validation_data'], d['validation_labels'] = zip(*validation_data)
    return d

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
    model.load_weights('./models/init_weights' + str(id) + ".h5")
    return model

def create_single_shot_sorted(tasks):
    sorted_data = list()
    sorted_labels = list()
    for task in tasks:
        copy = task.copy()
        random.shuffle(copy)
        labels, images = zip(*copy)
        sorted_data.extend(images)
        sorted_labels.extend(labels)
    return sorted_data, sorted_labels

def writeToJson(filename, results):
    with open(filename + '.json', 'w') as fp:
        json.dump(results, fp, indent=4)
