from .data import tag, unzip

import struct as st
import numpy as np
import random

sets = {
    'balanced' : {
        'train_images' : './datasets/emnist-balanced-train-images-idx3-ubyte',
        'train_labels' : './datasets/emnist-balanced-train-labels-idx1-ubyte',
        'test_images' : './datasets/emnist-balanced-test-images-idx3-ubyte',
        'test_labels' : './datasets/emnist-balanced-test-labels-idx1-ubyte',
        'classes': 47
    },
    'byclass' : {
        'train_images' : './datasets/emnist-byclass-train-images-idx3-ubyte',
        'train_labels' : './datasets/emnist-byclass-train-labels-idx1-ubyte',
        'test_images' : './datasets/emnist-byclass-test-images-idx3-ubyte',
        'test_labels' : './datasets/emnist-byclass-test-labels-idx1-ubyte',
        'classes': 62
    },
    'digits' : {
        'train_images' : './datasets/emnist-mnist-train-images-idx3-ubyte',
        'train_labels' : './datasets/emnist-mnist-train-labels-idx1-ubyte',
        'test_images' : './datasets/emnist-mnist-test-images-idx3-ubyte',
        'test_labels' : './datasets/emnist-mnist-test-labels-idx1-ubyte',
        'classes': 10
    },
    'letters' : {
        'train_images' : './datasets/emnist-letters-train-images-idx3-ubyte',
        'train_labels' : './datasets/emnist-letters-train-labels-idx1-ubyte',
        'test_images' : './datasets/emnist-letters-test-images-idx3-ubyte',
        'test_labels' : './datasets/emnist-letters-test-label-idx1-ubyte',
        'classes': 37
    }
}

def mean_sort(xs, ys, classes, sections, shuffle_sections=True):

    #If optimization needed:
    # Change from doing 'classes' amount of passes to single pass
    sorted_classes = dict()
    image_shape = xs[0].shape;

    for label in range(0, classes):
        #example[0] is its index, example[1] is the image
        #filter is filtering out all the digits that are the correct label
        image_collection = list(map( lambda x: x[1], filter(
            lambda x: ys[x[0]] == label, enumerate(xs)
        )))

        #Calculate our mean image
        #float 64 needed so no overflow in addition
        mean = np.zeros(shape=image_shape, dtype=np.float64)
        for image in range(0,len(image_collection)):
            mean = np.add(mean, np.asarray(image_collection[image]))
        mean = mean / len(image_collection)

        #Sort based on their difference score
        image_collection.sort(key=(lambda image: np.sum(np.absolute(image - mean))))

        #Insert into our dict
        #sorted_classes = {
        #   label0 : [images],
        #   label1 : [images],
        #   ...
        #}
        sorted_classes[label] = image_collection

    #Label items with their class (x,y) and split into chunks
    #{
    #   label0 :[chunk0 , chunk1 , ... , chunk_sections ]
    #   label1 :[chunk0 , chunk1 , ... , chunk_sections ]
    #}
    # chunk_x = [(x,y) , (x,y)]
    for label in sorted_classes.keys():
        sorted_classes[label] = tag(sorted_classes[label], label)
        sorted_classes[label] = np.array_split(sorted_classes[label], sections)

    #Create a super section comprising of each classes section
    #[
    #   section0 :[all chunk 0's] , optionally shuffled
    #   section1 :[all chunk 1's] , optionally shuffled
    #]
    difficulty_tiered_list = list()
    for chunk in range(0,sections):
        super_section = list()
        for label in sorted_classes.keys():
            super_section.extend(sorted_classes[label][chunk])
        if shuffle_sections:
            random.shuffle(super_section)
        difficulty_tiered_list.extend(super_section)

    #[ (x,y) , (x,y) , ... , (x,y) ] sorted by difficulty
    return unzip(difficulty_tiered_list);


#https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
#Guide on unpacking EMNIST data

#Loads an idx stored set of images
#   filename:
#       The file from which to load from
#
#   returns:
#       np array of all the images in the file
def load_idx_images(filename, amount):
    #print("Reading in " + filename)
    image_file = open(filename, 'rb')
    image_file.seek(0)

    #https://docs.python.org/2/library/struct.html
    magic = st.unpack('>4B', image_file.read(4))
    img_count = st.unpack('>I', image_file.read(4))[0]
    rows = st.unpack('>I', image_file.read(4))[0]
    columns =st.unpack('>I', image_file.read(4))[0]

    if amount > 0:
        img_count = amount

    bytes_per_pixel = 1 #should use magic number to work this out
    channels_per_pixel = 1; #Number of channels (1 for greyscale)
    total_bytes = img_count*rows*columns*bytes_per_pixel*channels_per_pixel

    images = np.zeros( (img_count, rows, columns, channels_per_pixel) )

    #print("Unpacking " + str(total_bytes) + " bytes ")
    image_bytes = st.unpack('>' + 'B'*total_bytes, image_file.read(total_bytes))

    #read in and correct sideways images, probably a way to read in correctly
    #print("Reshaping")
    images = np.asarray(image_bytes).astype(np.float16).reshape((img_count, rows, columns, channels_per_pixel))
    images /= 255
    #print("Rotating images")
    images = np.transpose(images, (0,2,1,3))
    return images

#Loads an idx stored set of labels
#   filename:
#       The file from which to load from
#
#   returns:
#       np array of all the labels in the file
def load_idx_labels(filename, amount):
    #print("Reading in " + filename)
    label_file = open(filename, 'rb')
    label_file.seek(0)

    #https://docs.python.org/2/library/struct.html
    magic = st.unpack('>4B', label_file.read(4))
    label_count = st.unpack('>I', label_file.read(4))[0]

    if amount > 0:
        label_count = amount

    bytes_per_label = 1 #should use magic number to work this out
    total_bytes = label_count*bytes_per_label

    labels = np.zeros(label_count)
    label_bytes = st.unpack('>' + 'B'*total_bytes, label_file.read(total_bytes))
    #print("Unpacking labels")
    labels = np.asarray(label_bytes) #probably has to be considered based on byte size

    return labels

def get(dataset, amount=-1):
    data_collection = dict()
    data_collection['x'] = load_idx_images(dataset['train_images'], amount)
    data_collection['y'] = load_idx_labels(dataset['train_labels'], amount)
    data_collection['test_x'] = load_idx_images(dataset['test_images'], amount)
    data_collection['test_y'] = load_idx_labels(dataset['test_labels'], amount)
    return data_collection, dataset['classes']
