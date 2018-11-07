import random
import numpy as np
from ml_util import *

class WeightedTaskSyllabus:
    """Trains a model 1 task at a time"""

    def __init__(self, data, weightings, difficulty_sorter, task_count,
        batch_size=128,
        validation_split=0.2,
        validation_data=None,
        pre_run=False,
        verbose_level=1):
        """
        data: tuple -> (x,y)
            the data to train on
        weightings: dict -> {'bw': _ , 'cw' : _ , 'fw' : _ }
            The waits with which to create tasks from
        difficulty_sorter: func(x,y) -> return [(x,y) , (x,y) , ... ]
            Given a list of data and its labels, return a list of tuples in
            order of difficulty
        task_count: int
            How many tasks to create out of the data
        batch_size: int
            The batch size for the trainer to use
        validation_split: float
            If no validation data is given, sets aside some training data as
            validation data
        validation_data: tuple -> (x,y)
            Data to validate the model on
        pre_run : bool
            Whether to run through the model once with unsorted data
        verbose_level: int
            logging variable

        """
        self.weightings = weightings
        self.verbose_level = verbose_level
        self.current_task = 0
        self._batch_size = batch_size
        self.pre_run=pre_run
        if validation_data:
            data['validation_x'] = validation_data[0]
            data['validation_y'] = validation_data[1]
            data['train_x'] = data[0]
            data['train_y'] = data[1]
        else:
            self.data = self._validation_split(data, validation_split)

        self.tasks = self._create_tasks(
            self.data['train_x'], self.data['train_y'], weightings, difficulty_sorter, task_count
        )

    def training_complete(self):
        """Returns if the training has complete"""
        return self.current_task >= len(self.tasks)

    def batch_size(self):
        """Returns the batchsize for the trainer to use"""
        return self._batch_size

    def next_train(self):
        """Returns the next task for the trainer to train the model on"""
        if self.pre_run:
            self.pre_run = False
            return self.data['train_x'], self.data['train_y']
        else:
            return unzip(self._current_task())

    def next_validation(self):
        """Returns the next set of samples for the trainer to validate on"""
        return self.data['validation_x'], self.data['validation_y']

    def on_task_complete(self, history, model):
        """Callback called by trainer once it has finished a round of training"""
        if self.verbose_level == 1:
            if self.pre_run:
                print("Completed prerun :  : \t" + str(history.history) )
            else:
                print("Completed task : " + str(self.current_task) + " : \t" + str(history.history) )
        self.current_task += 1

    def _current_task(self):
        """Returns the samples for the task the model is currently on"""
        return self._task(self.current_task)

    def _task(self, task_index):
        """Returns a certain task"""
        print("current_task", task_index)
        return self.tasks[task_index]

    def _sample(self, task_index, sample_index):
        """Returns a certain samples from a certain task"""
        return self._task(task_index)[sample_index]

    def _validation_split(self, data, validation_split):
        """Removes a random sample of the data to be used as validation data"""
        split_data = dict()
        labelled_data = list(zip(data[0], data[1]))
        random.shuffle(labelled_data)
        training_data, validation_data = split(labelled_data, (1-validation_split))
        split_data['train_x'], split_data['train_y'] = unzip(training_data)
        split_data['validation_x'], split_data['validation_y'] = unzip(validation_data)
        return split_data

    def _create_tasks(self, x, y, weightings, difficulty_sorter, task_count):
        """Segments the data into a certain amount of tasks"""
        data = difficulty_sorter(x,y) #given x,y -> return [(x,y) , (x,y)]
        tiered_data = chunk(data, task_count) #chunk data into N chunks
        tasks = list()
        print(tiered_data[0])
        if self.verbose_level == 1:
            print("Task weightings : ")

        for task_index in range(0, task_count): #For each task

            current_task_weight = weightings['cw']
            back_weights = list()
            forward_weights = list()

            #If previous tasks
            if task_index > 0:
                #Calculate weights for previous tasks
                previous_tasks = task_index
                back_parts = previous_tasks * (previous_tasks+1) / 2
                back_weight_per_part = weightings['bw'] / back_parts
                back_weights = [(x+1)*back_weight_per_part for x in range(0, task_index)]
            else: #No previous tasks
                current_task_weight += weightings['bw']

            #If future tasks
            if task_index < task_count - 1:
                #Calculate weights for forward tasks
                future_tasks = (task_count - task_index - 1)
                forward_parts = future_tasks * (future_tasks+1) / 2
                forward_weight_per_part = weightings['fw'] / forward_parts
                forward_weights = [(task_count-x)*forward_weight_per_part for x in range(task_index+1, task_count)]
            else: #No future tasks
                current_task_weight += weightings['fw']

            #Combine weights together into one list
            weights = list()
            weights.extend(back_weights)
            weights.append(current_task_weight)
            weights.extend(forward_weights)

            #Sample from each of our tasks
            task = sample_multiple(tiered_data, weights)

            if self.verbose_level == 1:
                print("Task " + str(task_index) + " Samples: " + str(len(task)) + " :\t", end='')
                print(np.round(weights,3))

            tasks.append(task)
        return tasks
