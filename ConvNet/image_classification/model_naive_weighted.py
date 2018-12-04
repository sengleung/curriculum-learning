from syllabuses.naive_weighted import NaiveWeightedSyllabus
from trainer import ModelTrainer

import util.data as data_util
import util.model as model_util
import util.emnist as emnist
import numpy as np

setname = 'balanced'
model_id = 0
model_name = 'balanced'
task_amount = 10
epochs = 1
weightings = { 'bw' : 0.15, 'cw' : 0.7, 'fw' : 0.15 }

balanced_set = emnist.sets['balanced']
data, classes = emnist.get(balanced_set, amount=1000)

model = model_util.load("untrained", model_id)
x, y, val_x, val_y = data_util.validation_split(data['x'], data['y'], 0.2)
test_x, test_y = data_util.prep(data['test_x'], data['test_y'], classes)

def preprocess_data(x, y, val_x, val_y):
    x, y = data_util.prep(x, y, classes)
    val_x, val_y = data_util.prep(val_x, val_y, classes)
    return x, y, val_x, val_y

syllabus = NaiveWeightedSyllabus(
    training_data=(x, y),
    task_amount=task_amount,
    validation_data=(val_x, val_y),
    weightings=weightings,
    preprocess_data=preprocess_data,
    epochs=1
)

trainer = ModelTrainer(
    model,
    syllabus
)

trainer.train()

model_util.save(model, model_name, model_id)
