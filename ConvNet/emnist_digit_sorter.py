import numpy as np
import emnist_file_loader as efl
import matplotlib.pyplot as plt

#Definitaly faster ways to do this but ehhh, only if we need
def emnist_digit_sort_by_mean_diff(examples, labels, classes):

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

        #Add their difference score (image, difference)
        # digit_collection = list(map(
        #     lambda digit: (digit[0], np.sum(np.absolute(digit[0] - mean)))
        #     ,digit_collection
        # ))

        #Sort based on their difference score
        image_collection.sort(key=(lambda image: np.sum(np.absolute(image - mean))))

        #Insert into our dict
        #{ digit_class : [images] , digit_class : [images] }
        sorted_digits[digit] = image_collection

    return sorted_digits;
