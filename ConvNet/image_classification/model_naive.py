from trainer import ModelTrainer
from syllabuses.naive import NaiveSyllabus

import util.data as data_util
import util.model as model_util
import util.emnist as emnist
import numpy as np

setname = 'balanced'
model_ID = 0
model_name = "naive"
task_amount = 10

#Load data and model
balanced_set = emnist.sets['balanced']
data, classes = emnist.get(balanced_set, amount=1000)
model = model_util.load("untrained" , model_ID)

#Load data and split it
x, y, val_x, val_y = data_util.validation_split(data['x'], data['y'], 0.2)
test_x, test_y = data_util.prep(data['test_x'], data['test_y'], classes)

#function for preprocessing data before being fed to model
def preprocess_data(x, y, val_x, val_y):
    x, y = data_util.prep(x, y, classes)
    val_x, val_y = data_util.prep(val_x, val_y, classes)
    return x, y, val_x, val_y

syllabus = NaiveSyllabus(
    training_data=(x, y),
    task_amount=task_amount,
    validation_data=(val_x, val_y),
    preprocess_data=preprocess_data,
    epochs=1
)

trainer = ModelTrainer(
    model,
    syllabus,
    on_task_end=None,
    on_task_start=None
)

#Train
trainer.train()

#Save trained model
model_util.save(model, model_name, model_ID)


"""
print("Epoch :%d\t Task :%d\t Samples Seen :%d"
        % (syllabus.current_epoch, syllabus.current_task_index, samples_seen))
    print("Evaluation | Loss : %f\tAcc : %f" % (score[0], score[1]))
"""
