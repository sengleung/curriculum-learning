import numpy as np
from numpy import array
from pickle import load
import random
from ml_util import sample, split

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))


### <---- main function ---->####
## load clean pairs
def get_data():
	clean_english = load_clean_sentences('clean_data/data/clean_english.pkl')
	clean_french = load_clean_sentences('clean_data/data/clean_french.pkl')
	clean_pairs = np.column_stack((clean_english,clean_french))
	### local test
	clean_english = clean_english[:1000]
	clean_french = clean_french[:1000]
	clean_pairs = clean_pairs[:1000]
	return clean_english, clean_french, clean_pairs


## split dataset into test and training data
def split_data(x, y):
	clean_pairs = np.column_stack((x, y))
	temp_cp = clean_pairs
	## Divide data into test and sample data
	test_pairs, training_pairs = sample(temp_cp, 0.01, True)
	return test_pairs, training_pairs


## sort dataset into test and training data
def sort_data(x, y):
	training_pairs = np.column_stack((x, y))

	## Sort training data by average length of bilingual pair
	tp_list = list(training_pairs)
	tp_list.sort(key=(lambda z: (len(z[0])+len(z[1]))/2))
	sorted_training_pairs = array(tp_list)
	sorted_training_tuples =[]
	for i in range(0,len(sorted_training_pairs)):
		sorted_training_tuples.append((sorted_training_pairs[i][0], sorted_training_pairs[i][1]))
	# print("SORTED ", sorted_training_tuples[0:20])
	return sorted_training_tuples

# x, y, all = get_data()
# test_pairs, training_pairs = split_data(x, y)
# strp = sort_data(training_pairs[0], training_pairs[1])
# print(strp)