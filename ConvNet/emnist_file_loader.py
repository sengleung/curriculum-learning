import struct as st
import numpy as np

#https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
#Guide on unpacking EMNIST data

#Loads an idx stored set of images
#   filename:
#       The file from which to load from
#
#   returns:
#       np array of all the images in the file
def load_idx_images(filename, amount):
    print("Reading in " + filename)
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

    print("Unpacking " + str(total_bytes) + " bytes ")
    image_bytes = st.unpack('>' + 'B'*total_bytes, image_file.read(total_bytes))

    #read in and correct sideways images, probably a way to read in correctly
    print("Reshaping")
    images = np.asarray(image_bytes).astype(np.float16).reshape((img_count, rows, columns, channels_per_pixel))
    images /= 255
    print("Rotating images")
    images = np.transpose(images, (0,2,1,3))
    return images

#Loads an idx stored set of labels
#   filename:
#       The file from which to load from
#
#   returns:
#       np array of all the labels in the file
def load_idx_labels(filename, amount):
    print("Reading in " + filename)
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
    print("Unpacking labels")
    labels = np.asarray(label_bytes) #probably has to be considered based on byte size

    return labels
