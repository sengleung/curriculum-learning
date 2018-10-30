import keras
import json
import random
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt

from helper import get_model, load_data, create_validation_split, create_tasks, writeToJson

#Consts
digits_classes = 10
balanced_classes = 47
by_class_classes = 62
classes = balanced_classes
task_count = 10

validation_split = 0.2
epochs = 1
batch_size = 128


model = get_model(0)
data_collection = load_data('balanced')

split = create_validation_split(
    data_collection['training_data'], data_collection['training_labels'], validation_split
)

tasks = create_tasks(
    data=np.asarray(split['training_data']),
    labels=split['training_labels'],
    classes=classes,
    task_count=task_count,
    randomize_each_task=True
)
#Flatten all tasks into one list
training_labelled_data = [data_point for task in tasks for data_point in task]
training_labels, training_data = zip(*training_labelled_data)

training_data = np.asarray(training_data)
training_y_vectors = keras.utils.to_categorical(training_labels, classes)

testing_data = np.asarray(data_collection['testing_data'])
testing_labels = data_collection['testing_labels']
testing_y_vectors = keras.utils.to_categorical(testing_labels, classes)

validation_data = np.asarray(split['validation_data'])
validation_labels = split['validation_labels']
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

for i in range(0,10):

    samples_looked_at = (i+1)*10000
    i = i%8
    train = training_data[i*10000:(i+1)*(10000)]
    labels = training_y_vectors[i*10000:(i+1)*(10000)]
    validation_score = model.fit(
        x=train,
        y=labels,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        shuffle=False,
        validation_data=(validation_data, validation_y_vectors)
    )

    validation_scores['validation_loss'].append(validation_score.history['val_loss'][-1])
    validation_scores['validation_accuracy'].append(validation_score.history['val_acc'][-1])
    validation_scores['samples_looked_at'].append(samples_looked_at)

    evaluation_score = model.evaluate(testing_data, testing_y_vectors, verbose=1)
    evaluation_scores['evaluation_loss'].append(evaluation_score[0])
    evaluation_scores['evaluation_accuracy'].append(evaluation_score[1])
    evaluation_scores['samples_looked_at'].append(samples_looked_at)

results = {
    'validation' : validation_scores,
    'evaluation' : evaluation_scores
}

writeToJson('./results/SortedFull', results)
score = model.evaluate(testing_data, testing_y_vectors, verbose=1)
