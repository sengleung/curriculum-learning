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
	clean_pairs = np.column_stack((clean_french,clean_english))
	### local test
	clean_english = clean_english[:10000]
	clean_french = clean_french[:10000]
	clean_pairs = clean_pairs[:10000]
	return clean_french, clean_english, clean_pairs


## split dataset into test and training data
def split_data(x, y):
	clean_pairs = np.column_stack((x, y))
	temp_cp = clean_pairs
	## Divide data into test and sample data
	test_pairs, training_pairs = sample(temp_cp, 0.01, True)
	print("Test sample = ", test_pairs.shape, "pairs")
	print("Training sample = ", training_pairs.shape, "pairs")
	return test_pairs, training_pairs


## sort dataset based on average length of bilinugal pair
def sort_data(x, y):
	training_pairs = np.column_stack((x, y))

	## Sort training data by average length of bilingual pair
	tp_list = list(training_pairs)
	tp_list.sort(key=(lambda z: (len(z[0])+len(z[1]))/2))
	sorted_training_pairs = array(tp_list)

	# format for syllabus
	sorted_training_tuples =[]
	for i in range(0,len(sorted_training_pairs)):
		sorted_training_tuples.append((sorted_training_pairs[i][0], sorted_training_pairs[i][1]))
	# print("SORTED ", sorted_training_tuples[0:20])
	return sorted_training_tuples
