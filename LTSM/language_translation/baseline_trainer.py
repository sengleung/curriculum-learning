import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from organise_data import create_tokenizer, max_length, encode_sequences, encode_output
from clean_data import load_clean_sentences
from pickle import load
import random


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

### <---- MAIN ---->####
all_pairs = load_clean_sentences('data/allpairs.pkl')
test_pairs = load_clean_sentences('data/testpairs.pkl')
sorted_training_pairs = load_clean_sentences('data/trainingpairs.pkl')
validation_set = load_clean_sentences('data/validationpairs.pkl')

clean_x = clean_french
clean_y = clean_english
train_x = sorted_training_pairs[:,0]
train_y = sorted_training_pairs[:,1]
validation_x = validation_set[:,0]
validation_y = validation_set[:,1]
test_x = test_pairs[:,0]
test_y = test_pairs[:,1]

## PREP MODEL TOKENIZER
print("Training Baseline Model\n\n")
print("Training Dataset= ", sorted_training_pairs.shape)
print("Validation Dataset= ", validation_set.shape)
# prep english tokeniser
x_tokenizer = create_tokenizer(clean_x)
x_vocab_size = len(x_tokenizer.word_index) +1
x_length = max_length(clean_x)
print('\n\nEnglish Vocabulary Size: %d' % x_vocab_size)
print('English Max Length: %d' % (x_length))
# prep french tokeniser
y_tokenizer = create_tokenizer(clean_y)
y_vocab_size = len(y_tokenizer.word_index) +1
y_length = max_length(clean_y)
print('French Vocabulary Size: %d' % y_vocab_size)
print('French Max Length: %d' % (y_length))
print("\n\n")

## PREP TRAINING DATA

# prepare training data
trainX = encode_sequences(x_tokenizer, x_length, train_x)
trainY = encode_sequences(y_tokenizer, y_length, train_y)
trainY = encode_output(trainY, y_vocab_size)
# prepare validation data
validationX = encode_sequences(x_tokenizer, x_length, validation_x)
validationY = encode_sequences(y_tokenizer, y_length, validation_y)
validationY = encode_output(validationY, y_vocab_size)

## DEFINE MODEL
model = load_model("./untrained/model_0.h5")

#print(model.summary())
plot_model(model, to_file='baseline_model.png', show_shapes=True)

## TRAIN MODEL
# fit model - 30 epochs, batch_size of 64 examples
filename = './trained/baseline_model.h5'
#
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(validationX, validationY), callbacks=[checkpoint], verbose=2)
