import keras
import numpy as np

from helper import get_model, load_data
from model_trainer import ModelTrainer
from WeightedTaskSyllabus import WeightedTaskSyllabus
from difficulty_sorters import emnist_difficulty_sort

#Consts
digits_classes = 10
balanced_classes = 47
by_class_classes = 62
classes = balanced_classes

task_count = 10
difficulty_sections = 15
weightings = { "bw" : 0.15, "cw" : 0.7, "fw" : 0.15}
model_name = "init_weights0.h5"

#Loading model and data
model = get_model(model_name)
data_collection = load_data('balanced')
data_collection['test_y'] = keras.utils.to_categorical(data_collection['test_y'])

#Setting up a syllabus
syl = WeightedTaskSyllabus(
    data=(data_collection['x'], data_collection['y']),
    weightings=weightings,
    validation_split=0.2,
    difficulty_sorter=emnist_difficulty_sort(balanced_classes, difficulty_sections),
    task_count=task_count,
    pre_run=False
)

#Create a model trainer with our syllabus
trainer = ModelTrainer(model, syl, verbose_level=1)

#Create a callback for our model trainer and pass it in
def preprocess_data(data, syllabus, model):
    data['x'] = np.asarray(data['x'])
    data['y'] = keras.utils.to_categorical(data['y'], classes)
    data['val_x'] = np.asarray(data['val_x'])
    data['val_y'] = keras.utils.to_categorical(data['val_y'], classes)

trainer.on_task_start(preprocess_data)

#Use the model trainer
trainer.train()

trainer.model.save("./models/models_task_based_model_trained_once")
#evaluate
# evaluation_score = model.evaluate(data_collection['test_x'], data_collection['test_y'], verbose=1)
# print(evaluation_score)
