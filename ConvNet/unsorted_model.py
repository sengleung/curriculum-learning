import keras
import json
import random
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

from helper import get_model, load_data, create_validation_split, writeToJson

#Consts
digits_classes = 10
balanced_classes = 47
by_class_classes = 62
classes = balanced_classes

validation_split = 0.2
epochs = 2
batch_size = 128


model = get_model(0)
data_collection = load_data('balanced')

zipped_data = zip(data_collection['x'], data_collection['y'])
validation_split, training_split = split(zipped_data, validation_split_percent)
data_collection['y'], data_collection['x'] = unzip(training_split)
data_collection['val_x'], data_collection['val_y'] = unzip(validation_split)


training_data = np.asarray(data_collection['x'])
training_labels = data_collection['y']
training_y_vectors = keras.utils.to_categorical(training_labels, classes)

testing_data = np.asarray(data_collection['test_x'])
testing_labels = data_collection['test_y']
testing_y_vectors = keras.utils.to_categorical(testing_labels, classes)

validation_data = np.asarray(data_collection['val_x'])
validation_labels = data_collection['val_y']
validation_y_vectors = keras.utils.to_categorical(validation_labels, classes)

validation_scores = {
    "samples_looked_at" : [],
    "validation_loss" : [],
    "validation_accuracy" : []
}
evaluation_scores = {
    "samples_looked_at" : [],
    "evaluation_loss" : [],
    "evaluation_accuracy" : []
}



validation_score = model.fit(
    x=training_data,
    y=training_y_vectors,
    batch_size=batch_size,
    epochs=2,
    verbose=1,
    shuffle=True,
    validation_data=(validation_data, validation_y_vectors)
)

#     validation_scores['validation_loss'].append(validation_score.history['val_loss'][-1])
#     validation_scores['validation_accuracy'].append(validation_score.history['val_acc'][-1])
#     validation_scores['samples_looked_at'].append(samples_looked_at)
#
#     evaluation_score = model.evaluate(testing_data, testing_y_vectors, verbose=1)
#     evaluation_scores['evaluation_loss'].append(evaluation_score[0])
#     evaluation_scores['evaluation_accuracy'].append(evaluation_score[1])
#     evaluation_scores['samples_looked_at'].append(samples_looked_at)
#
# results = {
#     'validation' : validation_scores,
#     'evaluation' : evaluation_scores
# }
#
# writeToJson('./results/unsortedFull', results)
score = model.evaluate(testing_data, testing_y_vectors, verbose=1)
print(score)
