import keras
import random
import numpy as np
from helper import create_tasks

class TaskManager:

    def __init__(self, data, labels, classes ,epsilon=0.1,
        weightings={ 'bw' : 0.0, 'cw' : 1.0, 'fw' : 0.0 }, task_count=10, progression=1):
        self.data = data
        self.labels = labels
        self.classes = classes
        self.current_epoch = 1
        self.task_count = task_count
        self.weightings = weightings
        self.previous_score = 0
        self.tasks = create_tasks(self.data, self.labels, self.classes,
            task_count, randomize_each_task=True)
        self.current_task = 0
        self.finished = False
        self.samples = list()
        self.sample_labels = list()
        self.progression = progression
        self.epsilon = epsilon
        self.samples_looked_at = 0
        self.previous_history = None

    def final_acc(self, history):
        return history.history['val_acc'][-1]

    def should_progress(self, history):
        if not self.previous_history:
            return True
        else:
            acc_diff = self.final_acc(history) - self.final_acc(self.previous_history)
            print("Accuracy diff: " + str(acc_diff))
            return acc_diff < self.epsilon or acc_diff < 0

    def submit_model_score(self, history):
        should_progress = self.should_progress(history)
        self.previous_history = history
        if should_progress: #Finished a Task
            self.current_task += self.progression
            if self.current_task >= self.task_count: #Finished a progression through the tasks
                self.finished = True

    def is_finished(self):
        return self.finished

    def sample_from_task(self, task, sample_percent):
        amount = int(np.ceil(len(task) * sample_percent))
        result = random.sample(task.copy(), amount)
        return result

    def get_current_task_samples(self):
        bw = self.weightings['bw']
        fw = self.weightings['fw']
        cw = self.weightings['cw']

        sample_collection = list()
        print("Current task is : " + str(self.current_task))
        #Previous tasks
        back_parts = 0
        back_parts_weight = 0
        future_parts = 0
        future_parts_weight = 0
        if self.current_task != 0:
            amount = self.current_task
            back_parts = (amount*(amount + 1)) / 2
            back_parts_weight = bw / back_parts
            for i in range(0, self.current_task):
                sample_weight = back_parts_weight * (i+1)
                sample_collection.extend(self.sample_from_task(self.tasks[i], sample_weight))

        #Current Task
        if self.current_task == 0: #If there are no previous tasks
            sample_collection.extend(self.sample_from_task(self.tasks[self.current_task], cw + bw))
        elif self.current_task == self.task_count - 1: #If there are no future tasks
            sample_collection.extend(self.sample_from_task(self.tasks[self.current_task], cw + fw))
        else:
            sample_collection.extend(self.sample_from_task(self.tasks[self.current_task], cw))

        #Future tasks
        if self.current_task != self.task_count:
            amount = self.task_count - self.current_task
            future_parts = (amount*(amount + 1)) / 2
            future_parts_weight = fw / future_parts
            for i in range(self.current_task+1, self.task_count):
                sample_weight = future_parts_weight * (self.task_count-i)
                sample_collection.extend(self.sample_from_task(self.tasks[i], sample_weight))
        #Set task colelction
        random.shuffle(sample_collection)
        sample_labels, samples = zip(*sample_collection)
        self.samples_looked_at += len(sample_labels)
        return np.asarray(samples) , sample_labels
