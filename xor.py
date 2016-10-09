from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt


def progress_plot(model, xTrn, yTrn, axisWidth, epochList):
    ''' A function to plot the progress of training a neural network for a
    binary classification problem'''

    X = [[x1, x2] for x2 in np.linspace(-(axisWidth - 1) / 2,
                                        1 + (axisWidth - 1) / 2, 201)
         for x1 in np.linspace(-(axisWidth - 1) / 2,
                               1 + (axisWidth - 1) / 2, 201)]

    # The figure will have a width equal to ceiling(sqrt(number))
    plotWidth = int(np.sqrt(len(epochList)))
    if plotWidth**2 < len(epochList):
        plotWidth += 1

    # The figure has a height equal to the ceiling of the number of axes
    # divided by the width
    plotHeight = (len(epochList) - 1) // plotWidth + 1

    fig, axs = plt.subplots(plotHeight, plotWidth, squeeze=False)

    epochList.sort()

    trained = 0

    for i in range(0, len(epochList)):
        axs[int(i / plotWidth), i % plotWidth].set_xlim(-(axisWidth - 1) / 2,
                                                        1 + (axisWidth - 1) /
                                                        2)
        axs[int(i / plotWidth), i % plotWidth].set_ylim(-(axisWidth - 1) / 2,
                                                        1 + (axisWidth - 1) /
                                                        2)
        axs[int(i / plotWidth), i % plotWidth].set_title("{0:d} Epochs".
                                                         format(epochList[i]))

        # Train the network
        model.fit(xTrn, yTrn, batch_size=4, nb_epoch=epochList[i] - trained,
                  verbose=0)
        trained = epochList[i]

        # Run the trained neural network and colour points on a sliding scale
        # with blue for 0 and red for 1
        Y = model.predict(X)
        c = [(Yi[0], 0, Yi[1]) for Yi in Y]

        # Plot the output
        Xt = np.transpose(X)
        axs[int(i / plotWidth), i % plotWidth].scatter(Xt[0], Xt[1], color=c)

        # Add marks at the training points
        for j in range(len(xTrn)):
            if yTrn[j][0] == 1:
                markColour = 'k'
            else:
                markColour = 'w'
            axs[int(i / plotWidth), i % plotWidth].scatter(xTrn[j][0],
                                                           xTrn[j][1],
                                                           color=markColour,
                                                           marker='o', s=50)

    # Remove all the unused axes
    for i in range(len(epochList), plotWidth * plotHeight):
        fig.delaxes(axs[int(i / plotWidth), i % plotWidth])

    fig.set_size_inches(min(5 * plotWidth, 20),
                        min(5 * plotHeight, 20),
                        forward=True)
    model.predict(xTrn)
    plt.show()


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
axisWidth = 3

# Create a plot to visualise the progress of training
epochList = [100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 5000, 10000]

progress_plot(model, x, y, axisWidth, epochList)
