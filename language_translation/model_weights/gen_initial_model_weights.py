import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from organise_data import org_data
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


"""Just a file to generate some model weights, keeps things more consistent to
use same weights"""

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


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

###########<---- MAIN ---->##############
## get data
x, y, all_pairs = get_data()

## PREP MODEL TOKENIZER

# prep x tokeniser
x_tokenizer = create_tokenizer(clean_x)
x_vocab_size = len(x_tokenizer.word_index) +1
x_length = max_length(clean_x)
print('X = ' + x[0] )
print('X Vocabulary Size: %d' % x_vocab_size)
print('X Max Length: %d' % (x_length))
# prep y tokeniser
y_tokenizer = create_tokenizer(clean_y)
y_vocab_size = len(y_tokenizer.word_index) +1
y_length = max_length(clean_y)
print('Y = ' + y[0])
print('Y Vocabulary Size: %d' % y_vocab_size)
print('Y Max Length: %d' % (y_length))


# define model
model = define_model(x_vocab_size, y_vocab_size, x_length, y_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.save_weights("./weights/init_weights" + str(0) + ".h5")
