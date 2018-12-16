import keras
import json
import numpy as np
import util.emnist as emnist
import util.data as data_util
import util.model as model_util
import results.results as results_util

"""
We pretend unsorted model is doing it in tasks so we can compare it
to the models who are using tasks
"""

epochs = 3
batch_size = 128
model_ids = [0]
task_counts_to_compare_to = [5, 10, 25]
balanced_set = emnist.sets['balanced']
validation_split = 0.2

#Could probably do some stuff with model.metric_names if needs be
#https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
loss_index = 0
categorial_accuracy_index = 1
top_2_accuracy_index = 2

filepath = './models/untrained'

#For every model set up
for task_count in task_counts_to_compare_to:
    for model_id in model_ids:

        results = []
        samples_seen = 0

        #Load in our modelPretend
        model_to_load = 'model_{0}'.format(model_id)
        model = model_util.load(model_to_load, filepath)

        #To prevent any possible corruption between cycles, just reload and
        #reprocess the data
        data, classes = emnist.get(balanced_set)
        x, y, val_x, val_y = data_util.validation_split(data['x'], data['y'], validation_split)

        #Convert data to be used in model
        x, y = data_util.prep(x, y, classes)
        val_x, val_y = data_util.prep(val_x, val_y, classes)
        test_x, test_y = data_util.prep(data['test_x'], data['test_y'], classes)

        #Create Tasks for comparison sake, just going through unsorted data in chunks
        tasks_x = data_util.chunk(x, task_count)
        tasks_y = data_util.chunk(y, task_count)

        #Train our current model through all the tasks for every epoch
        for epoch in range(0, epochs):
            for task_index in range(0, task_count):

                #Train our model on the task
                model.fit(
                    x=tasks_x[task_index],
                    y=tasks_y[task_index],
                    batch_size=batch_size,
                    epochs=1,
                    validation_data=(val_x, val_y)
                )

                #Evaluate our model
                score = model.evaluate(
                    x=test_x,
                    y=test_y,
                    batch_size=128
                )

                samples_seen += len(tasks_x[task_index])
                result_point = {
                    "samples_seens" : samples_seen,
                    "categorial_accuracy" : score[categorial_accuracy_index],
                    "top_2_accuracy" : score[top_2_accuracy_index],
                    "loss" : score[loss_index],
                    "epoch" : epoch,
                    "task" : task_index,
                }

                results.append(result_point)

        #We have finished training this model, save the results
        name = "id{0}_t{1}_unsorted".format(model_id, task_count)
        model_results = {
            "name" : name,
            "id" : model_id,
            "task_count" : task_count,
            "results" : results
        }
        results_util.save('./results/data', name, model_results)
