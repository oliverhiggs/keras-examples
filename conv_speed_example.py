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


(xTrn, yTrn), (xTst, yTst) = mnist.load_data()

# np.expand_dims is required to introduce the 'channel' axis, which is used in
# Convolutional models
xFlatTrn = np.array([np.expand_dims(sample.flatten(), -1) for sample in xTrn])
xFlatTst = np.array([np.expand_dims(sample.flatten(), -1) for sample in xTst])

yCatTrn = convert_to_one_of_k(yTrn, 10)
yCatTst = convert_to_one_of_k(yTst, 10)

# Create the model
# # 2D Convolutional Model
# model2D = Sequential([
#     Convolution2D(64, 2, 2, input_shape=(28, 28, 1), activation='sigmoid'),
#     # SpatialDropout2D(0.5),
#     MaxPooling2D(pool_size=(2, 2)),
#     # Convolution2D(10, 2, 2, activation='sigmoid'),
#     # # SpatialDropout2D(0.5),
#     # MaxPooling2D(pool_size=(3, 3)),
#     Flatten(),
#     Dense(10, activation='softmax')])


# 1D Convolutional Model
model1D = Sequential([
    Convolution1D(64, 4, input_shape=(784, 1), activation='sigmoid'),
    MaxPooling1D(pool_length=4),
    # Convolution1D(10, 4, activation='sigmoid'),
    # MaxPooling1D(pool_length=9),
    Flatten(),
    Dense(10, activation='softmax')])

# 2D Convolutional Model emulating 1D Model
model2D = Sequential([
    Convolution2D(64, 4, 1, input_shape=(784, 1, 1), activation='sigmoid'),
    MaxPooling2D(pool_size=(4, 1)),
    # Convolution2D(10, 4, 1, activation='sigmoid'),
    # MaxPooling2D(pool_size=(9, 1)),
    Flatten(),
    Dense(10, activation='softmax')])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model1D.compile(optimizer=sgd, loss='binary_crossentropy')
model2D.compile(optimizer=sgd, loss='binary_crossentropy')

# Test the model on the test data before and after fitting
correct1D = 0
incorrect1D = 0

correct2D = 0
incorrect2D = 0

# Make predictions for both 2D Convolutional model and 1D Convolutional model
print("Running 1D Predictions...")
predictions1D = model1D.predict(xFlatTst)
print("Running 2D Predictions...")
predictions2D = model2D.predict(np.expand_dims(xFlatTst, -1))

for i, prediction in enumerate(predictions1D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct1D += 1
    else:
        incorrect1D += 1

for i, prediction in enumerate(predictions2D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct2D += 1
    else:
        incorrect2D += 1

print("""\nBefore
1D Correct: {0}\t\t2D Correct: {1}
1D Incorrect: {2}\t\t2D Incorrect: {3}
1D Percent Correct: {4}%\t2D Percent Correct: {5}%\n""".
      format(correct1D, correct2D, incorrect1D, incorrect2D,
             100 * correct1D / (correct1D + incorrect1D),
             100 * correct2D / (correct2D + incorrect2D)))

# Fit the model
print("Fitting 1D Model...")
# model1D.fit(xFlatTrn, yCatTrn, nb_epoch=1, batch_size=32)
print("Fitting 2D Model...")
model2D.fit(np.expand_dims(xFlatTrn, -1), yCatTrn, nb_epoch=1, batch_size=32)

correct1D = 0
incorrect1D = 0
correct2D = 0
incorrect2D = 0

# Make predictions for both 2D Convolutional model and 1D Convolutional model
print("Running 1D Predictions...")
predictions1D = model1D.predict(xFlatTst)
print("Running 2D Predictions...")
predictions2D = model2D.predict(np.expand_dims(xFlatTst, -1))

for i, prediction in enumerate(predictions1D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct1D += 1
    else:
        incorrect1D += 1

for i, prediction in enumerate(predictions2D):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct2D += 1
    else:
        incorrect2D += 1

print("""\nAfter
1D Correct: {0}\t\t2D Correct: {1}
1D Incorrect: {2}\t\t2D Incorrect: {3}
1D Percent Correct: {4}%\t2D Percent Correct: {5}%\n""".
      format(correct1D, correct2D, incorrect1D, incorrect2D,
             100 * correct1D / (correct1D + incorrect1D),
             100 * correct2D / (correct2D + incorrect2D)))
