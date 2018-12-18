import random
import math
import numpy as np
import keras

#Maybe round instead of ceil
def sample(data, percent):
    return random.sample(data, int(math.ceil(len(data)*percent)))

def split(data, percent):
    divider = int(len(data)*percent)
    return data[0:divider], data[divider:-1]

def validation_split(x,y, percent):
    val_x, x = split(x, percent)
    val_y, y = split(y, percent)
    return x, y, val_x, val_y

def chunk(data, chunks):
    chunked_data = list()
    chunk_size = len(data) // chunks
    for i in range(0,chunks):
        chunked_data.append(data[i*chunk_size:(i+1)*chunk_size])
    return chunked_data

def sample_multiple(chunks, sample_weights):
    samples = list()
    for i in range(0, len(sample_weights)):
        samples.extend(sample(chunks[i], sample_weights[i]))
    return samples

def unzip(zipped_item):
    return list(map(lambda tup: list(tup), zip(*zipped_item)))

def tag(items, tag):
    return list(map(lambda item: (item, tag) ,items))

def to_categorical(labels, classes):
    return keras.utils.to_categorical(labels, classes)

def prep(x, y, classes):
    return np.asarray(x), to_categorical(y, classes)
