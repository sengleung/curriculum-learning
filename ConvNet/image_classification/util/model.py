import keras
from keras.models import load_model

#Needed for loading the model from a saved file
#https://github.com/keras-team/keras/issues/3911
def top_2_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def load(name, filepath):
    full = filepath + '/'  + name + '.h5'
    print("Loading model " + full)
    return load_model(
        full,
        custom_objects={'top_2_accuracy': top_2_accuracy}
    )


def save(model, filepath,  name):
    full = filepath + '/'  + name + '.h5'
    print("Saving model " + full)
    model.save(full)
