from keras.models import Sequential
from keras.layers import Dense, Activation
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

xFlatTrn = np.array([sample.flatten() for sample in xTrn])
xFlatTst = np.array([sample.flatten() for sample in xTst])

yCatTrn = convert_to_one_of_k(yTrn, 10)
yCatTst = convert_to_one_of_k(yTst, 10)

# Create the model
model = Sequential([
    Dense(200, input_dim=784),
    Activation('sigmoid'),
    Dense(20),
    Activation('sigmoid'),
    Dense(10),
    Activation('softmax')])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

# Test the model on the test data before and after fitting
correct = 0
incorrect = 0
predictions = model.predict(xFlatTst)
for i, prediction in enumerate(predictions):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct += 1
    else:
        incorrect += 1

print("\nBefore\nCorrect: {0}\nIncorrect: {1}\nPercent Correct: {2}%\n".
      format(correct, incorrect, 100 * correct / (correct + incorrect)))

# Fit the model
model.fit(xFlatTrn, yCatTrn)

correct = 0
incorrect = 0
predictions = model.predict(xFlatTst)
for i, prediction in enumerate(predictions):
    if np.argmax(prediction) == np.argmax(yCatTst[i]):
        correct += 1
    else:
        incorrect += 1

print("\nAfter\nCorrect: {0}\nIncorrect: {1}\nPercent Correct: {2}%\n".
      format(correct, incorrect, 100 * correct / (correct + incorrect)))
