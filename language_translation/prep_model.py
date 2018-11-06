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
from organise_data import get_data

# fit a tokenizer
# map words to integers (needed for modelling)
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model


### <---- main function ---->####
def get_model(weight_id):
    clean_x, clean_y, all_pairs = get_data()

    ## PREP MODEL TOKENIZER

    # prep english tokeniser
    x_tokenizer = create_tokenizer(clean_x)
    x_vocab_size = len(x_tokenizer.word_index) +1
    x_length = max_length(clean_x)
    # prep french tokeniser
    y_tokenizer = create_tokenizer(clean_y)
    y_vocab_size = len(fr_tokenizer.word_index) +1
    y_length = max_length(clean_y)

    ## DEFINE MODEL

    # define model
    model = define_model(x_vocab_size, y_vocab_size, x_length, y_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.load_weights('./model_weights/weights' + weight_id)

    return model
