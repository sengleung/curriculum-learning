Running the language translator

--Clean the bilingual pairs
/clean_data/clean_data.py
This involves removing any obscure characters, removing punctuation and setting all the characters to lowercase.
This will output two files, clean_english.pkl and clean_french.pkl

--Generate weights for model (to be used as control for all tests)
/gen_initial_model_weights.py
This will define a learning model and save weights to be used for models that will be tested.
This is to add some continuity with testing.
Weights will be saved to /model_weights/weights

--Initialise and train model
This will be call a lot of functions before running the model on given data for a set number of epochs and
batchsizes, under a certain syllabus.
Model will be saved for evaluation in /models
The various functions used work as so:

encode_data.py
This encodes the input and output data to integers padded to the same length so the model can process them
properly.

ml_util.py
This provides functions that samples the data.
It also gives the option of separating data into test and training data by cutting off a percentage of
the total data.

model_trainer.py
This runs the model on the given syllabus.
The on_start_task function always certain operations to be done before each task is run.
The training data is split into N subsets and is supplied to the model to learn with a bias towards subset X
on task X.

organise_data.py
This has functions to:
* Load the clean data - not sorted or split
* Split the data - split base dataset to test and training data by a given %
* Sort the data - sort the training data by average length of the bilingual pair

prep_model.py
Defines the model before it is to be trained.
This includes specifying the type of input and output it will see.

weigted tast syllabus
This defines the syllabus that will be used to train the model.
