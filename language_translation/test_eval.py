from eval_model import eval_model
from organise_data import get_data, split_data
from numpy import array



x, y, all_pairs = get_data()
test_pairs, training_pairs = split_data(x, y)
tp_list = list(training_pairs)
tp_list.sort(key=(lambda z: (len(z[0])+len(z[1]))/2))
sorted_training_pairs = array(tp_list)
eval_model(test_pairs, sorted_training_pairs, all_pairs)
