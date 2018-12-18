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
from organise_data import get_data, split_data
from encode_data import create_tokenizer, max_length
from pickle import dump

def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

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
	x, y, all_pairs = get_data()
	print("Prepping model - French to English")
	test_pairs, training_pairs = split_data(x, y)
	save_clean_data(all_pairs, 'allpairs.pkl')
	save_clean_data(test_pairs, 'testpairs.pkl')
	save_clean_data(training_pairs, 'trainingpairs.pkl')


	clean_x = x
	clean_y = y
	# prep english tokeniser
	x_tokenizer = create_tokenizer(clean_x)
	x_vocab_size = len(x_tokenizer.word_index) +1
	x_length = max_length(clean_x)
	# prep french tokeniser
	y_tokenizer = create_tokenizer(clean_y)
	y_vocab_size = len(y_tokenizer.word_index) +1
	y_length = max_length(clean_y)
	## DEFINE MODEL

	# define model
	model = define_model(x_vocab_size, y_vocab_size, x_length, y_length, 256)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.load_weights('./model_weights/weights/' + weight_id)
	return model
