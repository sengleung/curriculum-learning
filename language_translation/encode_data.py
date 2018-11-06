import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from pickle import load

# fit a tokenizer
# map words to integers (needed for modelling)
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
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

def language_translation_encode_data(train_x, train_y, test_x, test_y):
    ### switch parameters to reverse translation
    clean_x = all_pairs[:,0]
    clean_y = all_pairs[:,1]

    ## PREP MODEL TOKENIZER

    # prep english tokeniser
    x_tokenizer = create_tokenizer(clean_x)
    x_vocab_size = len(x_tokenizer.word_index) +1
    x_length = max_length(clean_x)
    # prep french tokeniser
    y_tokenizer = create_tokenizer(clean_y)
    y_vocab_size = len(fr_tokenizer.word_index) +1
    y_length = max_length(clean_y)


    ## PREP TRAINING DATA

    # prepare training data
    trainX = encode_sequences(x_tokenizer, x_length, train_x)
    trainY = encode_sequences(y_tokenizer, y_length, train_y)
    trainY = encode_output(trainY, y_vocab_size)
    # prepare validation data
    testX = encode_sequences(x_tokenizer, x_length, test_x)
    testY = encode_sequences(y_tokenizer, y_length, test_y)
    testY = encode_output(testY, y_vocab_size)
    return trainX, trainY, testX, testY
