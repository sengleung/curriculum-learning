import numpy as np
import matplotlib.pyplot as plt

def show_image(image, label):
    print("Label : " + str(label))
    rescaled = (image * 255).astype(np.uint8)
    plt.imshow(rescaled.reshape(28,28))
    plt.show()
