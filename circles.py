from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.optimizers import SGD

from vis_tools import progress_plot

import numpy as np


def generate_circle_data(radii, numPoints, maxPerterbutation):
    ''' Create training cases for a circle configuration
    Points are drawn randomly from circles of different radii with each being
    assigned to a different class
    The points will be perturbed randomly from the circle from which they are
    drawn
    '''
    # begin with an empty array for inputs and classes
    X = []
    Y = []

    for i in range(numPoints):
        # Select a class at random
        outClass = np.random.randint(0, len(radii))
        # Select an angle at random
        angle = np.random.rand() * 2 * np.pi
        pertubations = maxPerterbutation * (np.random.rand(2) - 0.5)
        X.append(radii[outClass] * np.array([np.cos(angle) + pertubations[0],
                                             np.sin(angle) + pertubations[1]]))
        y = np.zeros(len(radii))
        y[outClass] = 1
        Y.append(y)

    return (np.array(X), np.array(Y))


radii = [1, 2, 3]
numPoints = 200
maxPerterbutation = 0.5

# Create the neural network model
model = Sequential([
    Dense(10, input_dim=2),
    Activation('sigmoid'),
    Dense(10),
    Activation('sigmoid'),
    Dense(len(radii)),
    Activation('softmax')])

# Compile the model with an optimiser
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

X, Y = generate_circle_data(radii, numPoints, maxPerterbutation)

# Create a map of points to plot the function stored in the neural network
bottomLeft = [-(max(radii) + 1), -(max(radii) + 1)]
topRight = [max(radii) + 1, max(radii) + 1]

# Create a plot to visualise the progress of training
epochList = [1, 10, 20, 50, 100, 300, 400, 500, 2000]
batchSize = 10

progress_plot(model, X, Y, bottomLeft, topRight, epochList, batchSize,
              verbose=True)
