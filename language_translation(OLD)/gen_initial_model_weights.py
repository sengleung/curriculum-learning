import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
from organise_data import get_data
from encode_data import create_tokenizer, max_length, encode_sequences, encode_output


"""Just a file to generate some model weights, keeps things more consistent to
use same weights"""

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
clean_x, clean_y, all_pairs = get_data()

print(clean_x[999])
print(len(clean_y[999].split()))
## PREP MODEL TOKENIZER

# prep x tokeniser
x_tokenizer = create_tokenizer(clean_x)
x_vocab_size = len(x_tokenizer.word_index) +1
x_length = max_length(clean_x)
# prep y tokeniser
y_tokenizer = create_tokenizer(clean_y)
y_vocab_size = len(y_tokenizer.word_index) +1
y_length = max_length(clean_y)


# define model
model = define_model(x_vocab_size, y_vocab_size, x_length, y_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.save_weights("./model_weights/weights/init_weights" + str(0) + ".h5")
