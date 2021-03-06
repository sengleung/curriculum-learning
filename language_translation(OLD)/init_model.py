import keras
import numpy as np
from model_trainer import ModelTrainer
from WeightedTaskSyllabus import WeightedTaskSyllabus
from prep_model import get_model
from organise_data import get_data, sort_data, split_data, load_clean_sentences
from encode_data import language_translation_encode_data

task_count = 10
difficulty_sections = 15
weightings = { "bw" : 0.15, "cw" : 0.7, "fw" : 0.15}
weight_file = "init_weights0.h5"

#Loading model and data
model = get_model(weight_file)
print("Training model - French to English")
all_pairs = load_clean_sentences('allpairs.pkl')
test_pairs = load_clean_sentences('testpairs.pkl')
training_pairs = load_clean_sentences('trainingpairs.pkl')

# print("tp[0] = ", training_pairs[:,0])
# print("tp[1] =", training_pairs[:,1])


#Setting up a syllabus
syl = WeightedTaskSyllabus(
    data=(training_pairs[:,0], training_pairs[:,1]),
    weightings=weightings,
    validation_split=0.1,
    difficulty_sorter=sort_data,
    task_count=task_count,
    pre_run=False
)

#Create a model trainer with our syllabus
trainer = ModelTrainer(model, syl, verbose_level=1)

#Create a callback for our model trainer and pass it in
def preprocess_data(data, syllabus, model):
    data['x'], data['y'], data['val_x'], data['val_y'] = language_translation_encode_data(data['x'], data['y'], data['val_x'], data['val_y'])

trainer.on_task_start(preprocess_data)

#Use the model trainer
trainer.train()

print("model saved")
trainer.model.save("./models/trained_model.h5")
