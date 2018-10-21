import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')

img_rows, img_cols = 28, 28
shape = (img_rows, img_cols, 1)

X_train = np.array(data_train.iloc[:, 1:])
y_train = to_categorical(np.array(data_train.iloc[:, 0]))
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = shape, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.4))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, epochs = 20)

predicted_classes = classifier.predict_classes(X_test)

real_classes = np.zeros(10000)
y_test = y_test.astype('int')
for i in range(0,10000):
    for j in range(0, 10):
        if(y_test[i][j] == 1):
            real_classes[i] = (j)
        
correct = 0
for x in range(0,12000):
    if(predicted_classes[x] == real_classes[x]):
        correct += 1
        
print('The accuracy is', correct/10000)