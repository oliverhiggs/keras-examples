from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

from vis_tools import progress_plot

import numpy as np

# Create the neural network model
model = Sequential([
    Dense(2, input_dim=2),
    Activation('sigmoid'),
    Dense(2),
    Activation('softmax')])

# Compile the model with an optimiser
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

# Create training cases for an XOR function
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Create a map of points to plot the function stored in the neural network
bottomLeft = [-0.5, -0.5]
topRight = [1.5, 1.5]

# Create a plot to visualise the progress of training
epochList = [100, 200, 300, 400, 500, 1000]
batchsize = 4

progress_plot(model, x, y, bottomLeft, topRight, epochList, batchsize)
