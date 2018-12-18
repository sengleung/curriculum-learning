from pickle import load
import numpy as np
from numpy import array
from numpy import argmax

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from organise_data import create_tokenizer, max_length, encode_sequences, get_new_test
from clean_data import load_clean_sentences

## < ----- FUNCTIONS ------ > ###

## FOR EVALUATION ##

# 2 STEPS
# 1. Generate a translated output sequence
# Inference: model can predict the entire output sequence in one-shot manner..
#### translation = model.predict(source, verbose=0) ###
# This will be sequence of of integers to enumerate and lookup in the tokenizer to map back to words

# to perform REVERSE MAPPING
# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# to perform RM for each integer in the translation and return result as string of words
# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# to repeat this for each source phrase in a dataset and compare the predicted result to expected target phrase in french
# also calculate BLEU score
# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
	    # translate encoded source text
	    source = source.reshape((1, source.shape[0]))
	    translation = predict_sequence(model, tokenizer, source)
	    raw_src, raw_target = raw_dataset[i]
	    if i < 10:
	        print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
	    actual.append(raw_target.split())
	    predicted.append(translation.split())
	##    calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	# print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	# print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

	# 2. Repeat this process for many input examples and summarising the skill of the model across multiple cases

## < ----- MAIN ------ > ####

## LOAD & ORGANISE DATA
all_pairs = load_clean_sentences('data/allpairs.pkl')
test_pairs = load_clean_sentences('data/testpairs.pkl')
sorted_training_pairs = load_clean_sentences('data/trainingpairs.pkl')
clean_x = all_pairs[:,0]
clean_y = all_pairs[:,1]
train_x = sorted_training_pairs[:,0]
train_y = sorted_training_pairs[:,1]
test_x = test_pairs[:,0]
test_y = test_pairs[:,1]

# prep french tokeniser
x_tokenizer = create_tokenizer(clean_x)
x_vocab_size = len(x_tokenizer.word_index) +1
x_length = max_length(clean_x)
# prep english tokeniser
y_tokenizer = create_tokenizer(clean_y)
y_vocab_size = len(y_tokenizer.word_index) +1
y_length = max_length(clean_y)
# prep data
trainX = encode_sequences(x_tokenizer, x_length, train_x)
testX = encode_sequences(x_tokenizer, x_length, test_x)


for i in range(0,18):
	# load model
	model = load_model('./trained/model'+str(i)+'.h5')

	print('TEST - MODEL '+str(i))
	evaluate_model(model, y_tokenizer, testX, test_pairs)
