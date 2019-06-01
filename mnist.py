from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Convolution2D
from keras.layers import Flatten, MaxPooling1D
from keras.layers import MaxPooling2D, SpatialDropout2D, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist

import numpy as np


def convert_to_one_of_k(Y, k):
    '''Takes an n by 1 array of classification outputs and returns an n by k
    array, with the second dimension taking a value of 1 at the assigned class
    and a 0 elsewhere.

    k is the number of categories used in classification.

    i.e. [0, 2, 1] becomes [[1,0,0], [0,0,2], [0,1,0]]
    '''
    out = []
    for sample in Y:
        y = np.zeros(k)
        y[sample] = 1
        out.append(y)
    return np.array(out)


np.random.seed(1337)

(xTrn, yTrn), (xTst, yTst) = mnist.load_data()

yCatTrn = convert_to_one_of_k(yTrn, 10)
yCatTst = convert_to_one_of_k(yTst, 10)

xTrn = xTrn.astype('float32')
xTst = xTst.astype('float32')

xTrn /= 255
xTst /= 255

# Create the model
# 2D Convolutional Model
model2D = Sequential([
    Convolution2D(32, 3, 3, input_shape=(28, 28, 1), activation='relu'),
    # SpatialDropout2D(0.8),
    Convolution2D(32, 3, 3, activation='relu'),
    # SpatialDropout2D(0.5),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model2D.compile(optimizer='adadelta', loss='categorical_crossentropy')

# Test the model on the test data before and after fitting
correct2D = 0
incorrect2D = 0

# Make predictions for the 2D Convolutional model
print("Running Predictions...")
predictions2D = model2D.predict(np.expand_dims(xTst, -1))

for i, prediction in enumerate(predictions2D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct2D += 1
    else:
        incorrect2D += 1

print("""\nBefore
Correct: {0}\t\tIncorrect: {1}
Percent Correct: {2}%\n""".
      format(correct2D, incorrect2D,
             100 * correct2D / (correct2D + incorrect2D)))

# Fit the model
print("Fitting 2D Model...")
model2D.fit(np.expand_dims(xTrn, -1), yCatTrn, nb_epoch=12, batch_size=128)

correct1D = 0
incorrect1D = 0
correct2D = 0
incorrect2D = 0

# Make predictions for the 2D Convolutional model
print("Running Predictions...")
predictions2D = model2D.predict(np.expand_dims(xTst, -1))

for i, prediction in enumerate(predictions2D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct2D += 1
    else:
        incorrect2D += 1

print("""\nAfter
Correct: {0}\t\tIncorrect: {1}
Percent Correct: {2}%\n""".
      format(correct2D, incorrect2D,
             100 * correct2D / (correct2D + incorrect2D)))
