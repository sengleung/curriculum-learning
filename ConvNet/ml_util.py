import random
import math
import numpy as np

#Maybe round instead of ceil
def sample(data, percent):
    return random.sample(data, int(math.ceil(len(data)*percent)))

def split(data, percent):
    divider = int(len(data)*percent)
    return data[0:divider], data[divider:-1]

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
