import numpy as np
from numpy import array
from pickle import load
import random

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
#from ml_util import sample, split

## shuffle and then split data into test & training data
def split(data, percent):
    divider = int(len(data)*percent)
    return data[0:divider], data[divider:-1]

#Maybe round instead of ceil
def sample(data, percent, removal=False):
    if removal:
        random.shuffle(data)
        return split(data, percent)
    else:
        return random.sample(data, int(math.ceil(len(data)*percent)))

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))


### <---- DATA PREPARATION ---->####

def get_data():
	clean_english = load_clean_sentences('data/clean_english.pkl')
	clean_french = load_clean_sentences('data/clean_french.pkl')
	clean_pairs = np.column_stack((clean_french,clean_english))
	### local test
	clean_english = clean_english[:10000]
	clean_french = clean_french[:10000]
	clean_pairs = clean_pairs[:10000]
	return clean_french, clean_english, clean_pairs

def get_new_test():
    clean_english = load_clean_sentences('data/clean_english.pkl')
    clean_french = load_clean_sentences('data/clean_french.pkl')
    clean_pairs = np.column_stack((clean_french,clean_english))
    ### local test
    new_all_pairs = clean_pairs[10000:11000]
    clean_english = clean_english[10000:11000]
    clean_french = clean_french[10000:11000]

    test_pairs, training_pairs = split_data(clean_french, clean_english)
    return test_pairs, new_all_pairs


## split dataset into test and training data
def split_data(x, y):
	clean_pairs = np.column_stack((x, y))
	temp_cp = clean_pairs
	## Divide data into test and sample data
	test_pairs, training_pairs = sample(temp_cp, 0.1, True)
	print("Test sample = ", test_pairs.shape, "pairs")
	print("Training sample = ", training_pairs.shape, "pairs")
	return test_pairs, training_pairs


## sort dataset based on average length of bilinugal pair
def sort_data(x):
	training_pairs = x #np.column_stack((x, y))

	## Sort training data by average length of bilingual pair
	tp_list = list(training_pairs)
	tp_list.sort(key=(lambda z: (len(z[0])+len(z[1]))/2))
	sorted_training_pairs = array(tp_list)

	return sorted_training_pairs


### <----- DATA ENCODING ---------> ###

# fit a tokenizer
# map words to integers (needed for modelling)
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length in words
def max_length(lines):
	return max(len(line.split()) for line in lines)

## Input and Output needs to be encoded to integers and padded to max phrase length
## Word embedding for input sequences

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

## Model will predict probability of each word in the vocabulary as output - so one-hot

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
