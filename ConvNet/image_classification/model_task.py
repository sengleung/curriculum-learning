import json
import numpy as np

from syllabus import Syllabus

import util.emnist as emnist
import util.data as data_util
import util.model as model_util
import results.results as results_util

#Load all configurations
model_configs = []
with open('./models/configurations.json', 'r') as fp:
    model_configs = json.load(fp)

epochs = 3
batch_size = 128
balanced_set = emnist.sets['balanced']
validation_split = 0.2

#Could probably do some stuff with model.metric_names if needs be
#https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
loss_index = 0
categorial_accuracy_index = 1
top_2_accuracy_index = 2

filepath = './models/untrained'

#Used for estimating completion percent
total = float(len(model_configs))
current = 0.0

#For every configuration
for model_config in model_configs:

    #Used for estimating completion percent
    current += 1.0
    percent = current/total

    name = model_config['name']
    model_id = model_config['id']
    task_count = model_config['task_count']
    distribution = model_config['distribution']

    print("\nmodel:\t" + name + "\t Complete: " + str(percent) + "%")

    results = []
    samples_seen = 0

    #Load in our modelPretend
    model_to_load = 'model_{0}'.format(model_id)
    model = model_util.load(model_to_load, filepath)

    #To prevent any possible corruption between cycles, just reload and
    #reprocess the data, including resorting
    data, classes = emnist.get(balanced_set)
    x, y, val_x, val_y = data_util.validation_split(data['x'], data['y'], validation_split)
    x, y = emnist.mean_sort(x, y, classes, task_count)

    #Convert test data to be used in model
    val_x, val_y = data_util.prep(val_x, val_y, classes)
    test_x, test_y = data_util.prep(data['test_x'], data['test_y'], classes)

    #Set up syllabus which handles splitting into tasks
    syllabus = Syllabus(
        training_data =(x,y),
        epochs=epochs,
        task_amount=task_count,
        weightings=distribution,
    )

    #Until we have finished training
    while not syllabus.training_complete():
        #Get our next task and prep it
        x, y = syllabus.next()
        x, y = data_util.prep(x, y, classes)
        print("epoch : {0}\ttask: {1}".format(syllabus.current_epoch, syllabus.current_task_index))

        model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=1,
            validation_data=(val_x, val_y)
        )

        #Evaluate our model
        score = model.evaluate(
            x=test_x,
            y=test_y,
            batch_size=batch_size
        )

        samples_seen += len(x)
        result_point = {
            "samples_seens" : samples_seen,
            "categorial_accuracy" : score[categorial_accuracy_index],
            "top_2_accuracy" : score[top_2_accuracy_index],
            "loss" : score[loss_index],
            "epoch" : syllabus.current_epoch,
            "task" : syllabus.current_task_index,
        }

        results.append(result_point)
        syllabus.task_finished()

    #We have finished training this model, save the results
    model_results = {
        "name" : name,
        "id" : model_id,
        "task_count" : task_count,
        "results" : results
    }
    results_util.save('./results/data', name, model_results)
