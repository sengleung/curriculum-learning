import random
import numpy as np
import util.data as data_util

class NaiveSyllabus:
    """Trains a model 1 task at a time"""

    def __init__(self, training_data, task_amount, validation_data,
        on_task_start=None,
        on_task_end=None,
        epochs=1,
        batch_size=128,
        verbose_level=1):
        """Assumes data is presorted"""
        self.data = {
            'train_x' : training_data[0],
            'train_y' : training_data[1],
            'validation_x' : validation_data[0],
            'validation_y' : validation_data[1]
        }
        self.tasks = {
            'x' : data_util.chunk(self.data['train_x'], task_amount),
            'y' : data_util.chunk(self.data['train_y'], task_amount),
            'length' : task_amount
        }
        self.epochs = epochs
        self.task_amount = task_amount
        self.current_task_index = 0
        self.current_epoch = 0
        self._batch_size = batch_size
        self.verbose_level = verbose_level
        self._preprocess_data = preprocess_data
        self._on_task_start = on_task_start
        self._on_task_end = on_task_end

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

    def task_starting(self, x, y, val_x, val_y):
        self._handle_on_task_start(x, y, val_x, val_y)

    def task_finished(self, history, model):
        #If current epoch complete, reset task index and progress to next epoch
        self.current_task_index += 1
        self._handle_on_task_end(history)
        if self.is_current_epoch_complete():
            self.current_epoch += 1
            self.current_task_index = 0

    def is_current_epoch_complete(self):
        return self.current_task_index >= self.task_amount

    def on_task_start(self, function):
        self._on_task_start = function

    def on_task_end(self, function):
        self._on_task_end = function

    def preprocess_data(self,function):
        self._preprocess_data = function

    def _handle_on_task_start(self, x, y, val_x, val_y):
        if self._on_task_start:
            self._on_task_start(x, y, val_x, val_y)

    def _handle_on_task_end(self, score):
        if self._on_task_end:
            self._on_task_end(score)

    def _get_task(self, index):
        return self.tasks['x'][index], self.tasks['y'][index]

    def _is_all_epochs_complete(self):
        return self.current_epoch >= self.epochs
