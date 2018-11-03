import numpy as np
import emnist_file_loader as efl
import random
from ml_util import *

def emnist_difficulty_sort(classes, sections, shuffle_sections=True):
    #If optimization needed:
    # Change from doing 'classes' amount of passes to single pass
    def sort_data(examples,labels):
        sorted_digits = dict()
        image_shape = examples[0].shape;

        for digit in range(0, classes):
            #Filter all the images of a certain digit -> [images]
            #example[0] is its index, example[1] is the image
            image_collection = list(
                map(
                    lambda example: example[1],
                    filter(
                        lambda example: labels[example[0]] == digit
                        , enumerate(examples)
                    )
                )
            )

            #Calculate our mean image
            #float 64 needed so no overflow in addition
            mean = np.zeros(shape=image_shape, dtype=np.float32)
            for image in range(0,len(image_collection)):
                mean = np.add(mean, np.asarray(image_collection[image]))
            mean = mean / len(image_collection)

            #Sort based on their difference score
            image_collection.sort(key=(lambda image: np.sum(np.absolute(image - mean))))

            #Insert into our dict
            #{ digit_class : [images] , digit_class : [images] }
            sorted_digits[digit] = image_collection

        #Label items with their class (x,y) and split into chunks
        #{ 13 :[ [(x,y),(x,y)] , [(x,y), (x,y)] ... ]}
        for class_label in sorted_digits.keys():
            sorted_digits[class_label] = tag(sorted_digits[class_label], class_label)
            sorted_digits[class_label] = np.array_split(sorted_digits[class_label], sections)

        #Create a super section comprising of each classes section
        difficulty_tiered_list = list()
        for section in range(0,sections):
            super_section = list()
            for key in sorted_digits.keys():
                super_section.extend(sorted_digits[key][section])
            if shuffle_sections:
                random.shuffle(super_section)
            difficulty_tiered_list.extend(super_section)

        return difficulty_tiered_list;

    return sort_data
