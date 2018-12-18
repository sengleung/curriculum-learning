from pickle import load
import numpy as np
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from organise_data import get_data, split_data, load_clean_sentences
from encode_data import create_tokenizer, max_length, encode_sequences

### < ----- FUNCTIONS ------ > ###

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

def maxIndex(list):
	m = np.amax(list)
	max_pred = 0
	max_index = 0
	for i in range(0,len(list)):
		if list[i] > max_pred:
			max_pred = list[i]
			max_index = i
	return max_index


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
    # calculate BLEU score
    # print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    # print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    # print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    # print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# 2. Repeat this process for many input examples and summarising the skill of the model across multiple cases


## Evaluate pre-trained model
def eval_model(test_pairs, training_pairs, all_pairs):

	clean_x = all_pairs[:,0]
	clean_y = all_pairs[:,1]
	train_x = training_pairs[:,0]
	train_y = training_pairs[:,1]
	test_x = test_pairs[:,0]
	test_y = test_pairs[:,1]
	print("Evaluating Translation Model")
	print("Input = French Sentences\nOutput = English Translation")
	# prep english tokeniser
	x_tokenizer = create_tokenizer(clean_x)
	x_vocab_size = len(x_tokenizer.word_index) +1
	x_length = max_length(clean_x)
	print('French, input size = ', clean_x.shape )
	print('French, vocabulary size: %d' % x_vocab_size)
	print('French, max words in a sentence: %d' % (x_length))
	# prep french tokeniser
	y_tokenizer = create_tokenizer(clean_y)
	y_vocab_size = len(y_tokenizer.word_index) +1
	y_length = max_length(clean_y)
	print('English, input size = ', clean_y.shape)
	print('English, vocabulary size: %d' % y_vocab_size)
	print('English, max words in a sentence: %d' % (y_length))
	# prep data
	trainX = encode_sequences(x_tokenizer, x_length, train_x)
	testX = encode_sequences(x_tokenizer, x_length, test_x)
	# load model
	model = load_model('models/trained_model.h5')
	# test training sequences
	evaluate_model(model, y_tokenizer, trainX, training_pairs)

	# test test sequences
	print('test')
	evaluate_model(model, y_tokenizer, testX, test_pairs)



## < ----- MAIN ------ > ####

all_pairs = load_clean_sentences('allpairs.pkl')
test_pairs = load_clean_sentences('testpairs.pkl')
training_pairs = load_clean_sentences('trainingpairs.pkl')

eval_model(test_pairs, training_pairs, all_pairs)
