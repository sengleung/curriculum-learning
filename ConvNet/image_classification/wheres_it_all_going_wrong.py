from helper import *
from ml_util import *
from functools import reduce

import matplotlib.pyplot as plt
import keras
import numpy as np

classes = 47
model1 = keras.models.load_model("./models/models_task_based_model_trained_once")
model2 = get_model_with_weights("init_weights0.h5")

data = load_data('balanced', 1000)

data['test_y_vector'] = keras.utils.to_categorical(data['test_y'], classes)
object_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
index = np.arange(len(object_string))
bw = 0.25


def general_uncertainty_calc(array):
    regulizer = 1 / np.max(array)
    answer = reduce(lambda x,y: x + (regulizer*y)**2, array, 0)
    return answer - 1

def plot_uncertainty_of_a_class(model, x, y, label):

    #Filter based on label
    data = list()
    ids = list()
    for i in range(0, len(x)):
        if y[i] == label:
            data.append(x[i])
            ids.append(i)

    indexes = np.arange(len(data))

    p = model.predict_classes(np.asarray(data))
    pv = model.predict(np.asarray(data))

    results = (p == label)

    probabilties = np.zeros(len(pv))
    certainties_correct = np.zeros(len(pv))
    certainties_incorrect = np.zeros(len(pv))

    for i,p in enumerate(pv):
        correct = results[i]
        probabilties[i] = p[label]
        if correct:
            certainties_correct[i] = general_uncertainty_calc(p)
        else:
            certainties_incorrect[i] = general_uncertainty_calc(p)


    plt.bar(indexes+0.5, certainties_incorrect, 0.3, color='r', label='Incorrectly Guessed')
    plt.bar(indexes+0.25, probabilties, 0.3, color='g', label='Probabilty of label')
    plt.bar(indexes, certainties_correct, 0.3, color='b', label='Correctly Guessed')
    plt.title("Uncertainty Score in guessing : " + str(label))
    plt.xticks(indexes+0.35, ids)
    plt.xlabel("Index of the " + str(label) + " in dataset")
    plt.ylabel("Uncertainty Score")
    plt.legend()
    plt.show()

    for i in range(0, len(pv)):
        pd = pv[i]
        correct = results[i]
        uncertainty = general_uncertainty_calc(pd)
        plt.title("Example : " + str(ids[i]) + " with Uncertainty : " + str(uncertainty))

        if correct:
            plt.bar(index, pd, 0.4, color='b', label='Probabilities of labels: Correct')
        else:
            plt.bar(index, pd, 0.4, color='r', label='Probabilities of labels: Incorrect')

        plt.xticks(index, list(object_string))
        plt.xlabel("Probabilities of classes")
        plt.legend()
        plt.show()
# predictions = model.predict(data['test_x'])
# indices = [i for i,v in enumerate(predictions) if predictions[i] != data['test_y'][i]]
# test1 = predictions[0]



plot_uncertainty_of_a_class(model1, data['test_x'], data['test_y'], 5)
plot_uncertainty_of_a_class(model2, data['test_x'], data['test_y'], 5)
# plt.bar(index, corrects, bw, color='b', label='Correct')
# plt.bar(index+bw, incorrects, bw, color='r', label='Incorrect')
# plt.bar(index+bw*2, certainties, bw, color='g', label='Certainties')
# plt.xticks(index, list(object_string))
# plt.legend()
# plt.show()


#
# for i in range(0, 1000):
#     if data['test_y'][i] == 0:
#         test_image(data['test_x'][i], data['test_y'][i])



# incorrects = np.nonzero(model.predict_class(data['test_x']).reshape((-1,)) != data['test_y'])
# print(len(incorrects))
