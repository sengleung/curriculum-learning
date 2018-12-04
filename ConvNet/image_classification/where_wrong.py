from helper import *
from ml_util import *
from functools import reduce

from keras.layers.core import Lambda
from keras import backend as K


import matplotlib.pyplot as plt
import keras
import numpy as np

classes = 47
classes_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

index_dropout1 = 3
index_dropout2 = 6
index_softmax = 7

model1 = keras.models.load_model("./models/models_task_based_model_trained_once")
model2 = get_model_with_weights("init_weights0.h5")
data = load_data('balanced', 1000)

#Get the scatter of 100 stochastic forward passes
def dropout_layer1(model):
    return model.layers[index_dropout1]

def dropout_layer2(model):
    return model.layers[index_dropout2]

def softmax_layer(model):
    return model.layers[index_softmax]

#Keeps dropout happening during prediction
def replace_dropout_layers(model):
    model.layers[index_dropout1] = Lambda(lambda x: K.dropout(x, level=0.25))
    model.layers[index_dropout2] = Lambda(lambda x: K.dropout(x, level=0.50))
    return model

def filter_data(x,y,label):
    filtered_x = list()
    filtered_y = list()
    for i in range(0,len(x)):
        if y[i] == label:
            filtered_x.append(x[i])
            filtered_y.append(y[i])
    return filtered_x, filtered_y

def copy(item, amount):
    return [item] * amount

xs, ys = filter_data(data['x'],data['y'], 5)
# model1 = replace_dropout_layers(model1)
softmax_layer1 = softmax_layer(model1)
# print(model1.summary())
#
sm = K.function([model1.input, K.learning_phase()], [softmax_layer1.output])
first = copy(xs[0], 100)
results = sm([first, 1])

inputs = results[0]
outputs = results[0]
x_ins = []
y_ins = []
x_outs = []
y_outs = []
for ins in inputs:
    for i, v in enumerate(ins):
        x_ins.append(i)
        y_ins.append(v)
for outs in outputs:
    for i, v in enumerate(outs):
        x_outs.append(i)
        y_outs.append(v)



plt.plot(x_ins,y_ins, '_')
plt.xticks(np.arange(classes), list(classes_string))
plt.show()


plt.plot(x_outs,y_outs, '_')
plt.xticks(np.arange(classes), list(classes_string))
plt.show()
