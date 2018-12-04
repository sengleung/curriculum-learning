import util.data as data_util
import util.model as model_util
import util.emnist as emnist
import numpy as np

from trainer import ModelTrainer
from syllabuses.naive_weighted import NaiveWeightedSyllabus

###
#Model setup
###
def process_data(x,y, classes):
    return np.asarray(x), data_util.to_categorical(y, classes)

setname = 'balanced'
model_id = 0
load_model_name = 'untrained_' + setname
save_model_name = 'untrained_' + setname + 'naiveWeighted'
task_amount = 10
weightings = { 'bw' : 0.15, 'cw' : 0.7, 'fw' : 0.15 }

balanced_set = emnist.sets['balanced']
data, classes = emnist.get(balanced_set, amount=1000)

model = model_util.create(classes)
model = model_util.compile(model)
model = model_util.load_weights(model, load_model_name, model_id)

x, y, val_x, val_y = data_util.validation_split(data['x'], data['y'], 0.2)
test_x, test_y = process_data(data['test_x'], data['test_y'], classes)

###
#Result vars
###
samples_seen = 0

###
#Callbacks
###
def preprocess_data(x, y, val_x, val_y):
    x, y = process_data(x, y, classes)
    val_x, val_y = process_data(val_x, val_y, classes)
    return x, y, val_x, val_y

def on_task_start(x, y, val_x, val_y):
    global samples_seen
    samples_seen += len(x)
    print("Epoch :%d\t Task :%d\t Samples Seen :%d"
        % (syllabus.current_epoch, syllabus.current_task_index, samples_seen))

def on_task_end(score):
    if syllabus.is_current_epoch_complete():
        score = model.evaluate(test_x, test_y, verbose=0)
        print("Evaluation | Loss : %f\tAcc : %f" % (score[0], score[1]))
###
#Syllabus setup
###
syllabus = NaiveWeightedSyllabus(
    training_data=(x, y),
    task_amount=task_amount,
    validation_data=(val_x, val_y),
    weightings=weightings,
    preprocess_data=preprocess_data,
    on_task_start=on_task_start,
    on_task_end=on_task_end,
    epochs=3
)

###
#Setup trainer
###
trainer = ModelTrainer(model, syllabus, verbose_level=0)
trainer.train()
