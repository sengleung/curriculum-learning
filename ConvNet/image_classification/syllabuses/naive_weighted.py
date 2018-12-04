import random
import numpy as np
import util.data as data_util

class NaiveWeightedSyllabus:
    """Trains a model 1 task at a time,
    including weighted samples from other tasks"""

    def __init__(self, training_data, task_amount, validation_data, weightings,
        preprocess_data=None,
        on_task_start=None,
        on_task_end=None,
        epochs=1,
        resample_each_epoch=True,
        batch_size=128):
        """Assumes data is presorted"""
        self.data = {
            'train_x' : training_data[0],
            'train_y' : training_data[1],
            'validation_x' : validation_data[0],
            'validation_y' : validation_data[1]
        }
        xs, ys = self._create_tasks(self.data['train_x'], self.data['train_y'],
                                        task_amount, weightings)
        self.tasks = {
            'x' : xs,
            'y' : ys,
            'length' : task_amount
        }
        self.epochs = epochs
        self.task_amount = task_amount
        self.current_task_index = 0
        self.current_epoch = 0
        self._batch_size = batch_size
        self._preprocess_data = preprocess_data

    def training_complete(self):
        return self._is_all_epochs_complete()

    def batch_size(self):
        return self._batch_size

    def next(self):
        x, y = self._get_task(self.current_task_index)
        val_x, val_y = self.data['validation_x'], self.data['validation_y']
        if self._preprocess_data: #If _preprocess_data callback is set
                x, y, val_x, val_y = self._preprocess_data(x, y, val_x, val_y)
        return x, y, val_x, val_y

    def task_starting(self):
        #Nothing
        return

    def task_finished(self, history, model):
        #If current epoch complete, reset task index and progress to next epoch
        self.current_task_index += 1
        if self.is_current_epoch_complete():
            self.current_epoch += 1
            self.current_task_index = 0

    def is_current_epoch_complete(self):
        return self.current_task_index >= self.task_amount

    def _get_task(self, index):
        return self.tasks['x'][index], self.tasks['y'][index]

    def _is_all_epochs_complete(self):
        return self.current_epoch >= self.epochs

    def _create_tasks(self, x, y, task_amount, weightings):
        tasks_x = []
        tasks_y = []
        xy = list(zip(x,y))
        chunks = data_util.chunk(xy, task_amount)
        for task_index in range(0, task_amount):
            cw = weightings['cw']
            fw = weightings['fw']
            bw = weightings['bw']

            back_weights = []
            forward_weights = []
            if task_index <= 0: #If we are on the first task
                cw += bw
            else:
                back_parts = list(range(1,task_index+1))
                total = sum(back_parts)
                weight_per = bw / total
                back_weights = [x*weight_per for x in back_parts]

            if task_index+1 >= task_amount: #If we are on the last task
                cw += fw
            else:
                forward_parts = list(range((task_amount-1) - task_index, 0, -1))
                total = sum(forward_parts)
                weight_per = fw / total
                forward_weights = [x*weight_per for x in forward_parts]

            weights = back_weights + [cw] + forward_weights
            task = data_util.sample_multiple(chunks, weights)
            task_x, task_y = data_util.unzip(task)
            tasks_x.append(task_x)
            tasks_y.append(task_y)
            print("Task: %d\t Samples: %d\t" % (task_index, len(task)), end='')
            print(np.round(weights,3))

        return tasks_x, tasks_y
