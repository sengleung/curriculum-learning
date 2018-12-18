import json
import numpy as np
from numpy import array

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from organise_data import get_data, split_data, sort_data, create_tokenizer, max_length, encode_sequences, encode_output
from clean_data import save_clean_data, load_clean_sentences
from pickle import load

from syllabus import Syllabus

import results.results as results_util


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model


#Load all configurations
model_configs = []
with open('./models/configurations.json', 'r') as fp:
    model_configs = json.load(fp)

epochs = 3
batch_size = 64

#Could probably do some stuff with model.metric_names if needs be
#https://stackoverflow.com/questions/51299836/what-values-are-returned-from-model-evaluate-in-keras
loss_index = 0
categorial_accuracy_index = 1
top_2_accuracy_index = 2

filepath = './models/untrained'

#Used for estimating completion percent
total = float(len(model_configs))
current = 0.0
model_number=0

#For every configuration
for model_config in model_configs:

	#Used for estimating completion percent
	current += 1.0
	percent = current/total

	name = model_config['name']
	model_id = model_config['id']
	task_count = model_config['task_count']
	distribution = model_config['distribution']

	print("\nmodel:\t" + name + "\t Complete: " + str(percent) + "%")

	results = []
	samples_seen = 0

    ### <---- MAIN ---->####
	all_pairs = load_clean_sentences('data/allpairs.pkl')
	test_pairs = load_clean_sentences('data/testpairs.pkl')
	sorted_training_pairs = load_clean_sentences('data/trainingpairs.pkl')
	validation_set = load_clean_sentences('data/validationpairs.pkl')

	clean_x =  all_pairs[:,0]
	clean_y =  all_pairs[:,1]
	train_x = sorted_training_pairs[:,0]
	train_y = sorted_training_pairs[:,1]
	validation_x = validation_set[:,0]
	validation_y = validation_set[:,1]
	test_x = test_pairs[:,0]
	test_y = test_pairs[:,1]

    ## PREP MODEL TOKENIZER

    # prep english tokeniser
	x_tokenizer = create_tokenizer(clean_x)
	x_vocab_size = len(x_tokenizer.word_index) +1
	x_length = max_length(clean_x)
	print('English Vocabulary Size: %d' % x_vocab_size)
	print('English Max Length: %d' % (x_length))
	# prep french tokeniser
	y_tokenizer = create_tokenizer(clean_y)
	y_vocab_size = len(y_tokenizer.word_index) +1
	y_length = max_length(clean_y)
	print('French Vocabulary Size: %d' % y_vocab_size)
	print('French Max Length: %d' % (y_length))

	## PREP TRAINING DATA

	# prepare training data
	trainX = encode_sequences(x_tokenizer, x_length, train_x)
	trainY = encode_sequences(y_tokenizer, y_length, train_y)
	trainY = encode_output(trainY, y_vocab_size)
	# prepare validation data
	validationX = encode_sequences(x_tokenizer, x_length, validation_x)
	validationY = encode_sequences(y_tokenizer, y_length, validation_y)
	validationY = encode_output(validationY, y_vocab_size)

	testX = encode_sequences(x_tokenizer, x_length, test_x)
	testY = encode_sequences(y_tokenizer, y_length, test_y)
	testY = encode_output(testY, y_vocab_size)
	# define model
	model = load_model("./untrained/model_0.h5")

	filename = './trained/model'+str(model_number)+'.h5'
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    #Set up syllabus which handles splitting into tasks
	syllabus = Syllabus(
        training_data =(trainX,trainY),
        epochs=epochs,
        task_amount=task_count,
        weightings=distribution,
    )

    #Until we have finished training
	while not syllabus.training_complete():
        #Get our next task and prep it
	    x, y = syllabus.next()
	    print("epoch : {0}\ttask: {1}".format(syllabus.current_epoch, syllabus.current_task_index))

	    model.fit(
	        np.asarray(x),
	        np.asarray(y),
	        batch_size=batch_size,
	        epochs=1,
	        validation_data=(validationX, validationY),
	        callbacks=[checkpoint],
	        verbose=2
	    )

	    #Evaluate our model
	    score = model.evaluate(
	        x=testX,
	        y=testY,
	        batch_size=batch_size
	    )

	    samples_seen += len(x)
	    result_point = {
	        "samples_seens" : samples_seen,
	        "epoch" : syllabus.current_epoch,
	        "task" : syllabus.current_task_index,
	    }

	    results.append(result_point)
	    syllabus.task_finished()

	#We have finished training this model, save the results
	model_results = {
	    "name" : name,
	    "id" : model_id,
	    "task_count" : task_count,
	    "results" : results
	}
	results_util.save('./results/data', name, model_results)
	model_number = model_number+1
