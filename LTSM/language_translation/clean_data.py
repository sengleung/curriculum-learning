# import pandas
import numpy as np
from numpy import array
# import tensorflow as tf
# from tensorflow import keras
## for data cleaning
import re
import string
from unicodedata import normalize
from pickle import load
from pickle import dump


# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs

# clean a list of lines
def clean_pairs(lines):
	## format as string list to return correct output
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return array(cleaned)

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

### <---- MAIN ---->####

## LOAD DATA

doc = load_doc('data/eng-fra.txt')

## CLEAN DATA

pairs = to_pairs(doc)

clean_pairs = clean_pairs(pairs)

# for i in range(100):
# 	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))

# print(clean_pairs[1])
# raw_target, raw_src = clean_pairs[1]
# print(raw_target)
# print(raw_src)

clean_eng = clean_pairs[:,0]
clean_fr = clean_pairs[:,1]

save_clean_data(clean_eng, 'data/clean_english.pkl')
save_clean_data(clean_fr, 'data/clean_french.pkl')
